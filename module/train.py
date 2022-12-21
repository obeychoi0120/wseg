import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import functional as tvf
import wandb

import os
import random
import numpy as np
from copy import deepcopy


from util import pyutils
from data.augmentation.randaugment import tensor_augment_list

from module.validate import validate
from module.loss import adaptive_min_pooling_loss, get_er_loss, get_eps_loss, get_contrast_loss
from module.ssl import cutmix, EMA, apply_strong_tr, get_ssl_loss

def train_cls(train_loader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls')
    timer = pyutils.Timer("Session started: ")
    # DataLoader
    loader_iter = iter(train_loader)
    
    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    tscalar = {}
    val_freq = max_step // args.val_times ### validation logging
    # Iter
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, label = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                img_id, img, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred = model(img)

            # Classification loss
            loss = F.multilabel_soft_margin_loss(pred[:, :-1], label)
            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss_cls')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (tscalar['train/lr']), flush=True)
                
                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                    
                tscalar.clear()

            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


def train_seam(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')
    tb_writer = SummaryWriter(args.log_folder)
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    print(args)
    ### validation logging
    val_num = 10 # 10 times
    val_freq = max_step // val_num

    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, label = next(loader_iter)

            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)

            pred1, cam1, pred_rv1, cam_rv1 = model(img)
            pred2, cam2, pred_rv2, cam_rv2 = model(img2)

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)

            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # total loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss = loss_cls + loss_er + loss_ecr

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tblog
            tb_dict = {}
            for k in avg_meter.get_keys():
                tb_dict['train/' + k] = avg_meter.pop(k)
            tb_dict['train/lr'] = optimizer.param_groups[0]['lr']
            
            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (tb_dict['train/loss_cls']),
                      'Loss_ER: %.4f' % (tb_dict['train/loss_er']),
                      'Loss_ECR:%.4f' % (tb_dict['train/loss_ecr']),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
            # Validate 10 times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                if val_dataloader is not None:
                    # loss_, mAP, mean_acc, mean_precision, mean_recall, mean_f1, corrects, precision, recall, f1
                    tb_dict['val/loss'], tb_dict['val/mAP'], tb_dict['val/mean_acc'], tb_dict['val/mean_precision'], \
                    tb_dict['val/mean_recall'], tb_dict['val/mean_f1'], acc, precision, recall, f1 = validate(model, val_dataloader, iteration, args) ###
                
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

            # tblog update
            for k, value in tb_dict.items():
                tb_writer.add_scalar(k, value, iteration)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


def train_eps(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    # DataLoader
    loader_iter = iter(train_dataloader)

    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    tscalar = {}
    val_freq = max_step // args.val_times ### validation logging
    # Iter
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred, cam = model(img)

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam,
                                                              saliency,
                                                              label,
                                                              args.tau,
                                                              args.alpha,
                                                              intermediate=True,
                                                              num_class=args.num_sample)
            loss = loss_cls + loss_sal

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (tscalar['train/lr']), flush=True)
                
                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)

            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
            timer.reset_stage()

            
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


def train_contrast(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr')
    # tb_writer = SummaryWriter(args.log_folder)
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)

    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times ### validation logging
    
    gamma = 0.10
    print(args)
    print('Using Gamma:', gamma)

    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
            pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)

            # Classification loss 1
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

            # Classification loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)

            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)

            # loss cls = cam cls loss + cam_cv cls loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.

            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            loss = loss_cls + loss_sal + loss_nce #+ loss_er + loss_ecr

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ## Logging 
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                # Print Logs
                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                      'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                      'Fin:%s' % (timer.str_est_finish()), flush=True)

                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                
                tscalar.clear()

            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


def train_cls_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_ssl']
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')
        log_keys.append('mt_mask_ratio')
    if 2 in args.ssl_type:
        log_keys.append('loss_pmt')
    if 3 in args.ssl_type:
        log_keys.append('loss_pl')
        log_keys.append('mask_ratio')
    if 4 in args.ssl_type:
        log_keys.append('loss_con')
    if 5 in args.ssl_type:
        log_keys.append('loss_cdc')
        log_keys.append('ratio_cdc_pos')
        log_keys.append('ratio_cdc_neg')
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")

    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###
    
    # EMA
    #ema_model = deepcopy(model)
    ema = EMA(model, args.ema_m)
    ema.register()

    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    tscalar = {}
    val_freq = max_step // args.val_times ### validation logging

    # Iter
    print(args)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img_w, img_s, ops, label = next(lb_loader_iter)
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img_w, img_s, ops, label = next(lb_loader_iter)
            B = len(img_id)

            ### Unlabeled ###
            if train_ulb_dataloader:
                try:
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                except:
                    ulb_loader_iter = iter(train_ulb_dataloader)        ###
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)

                # Concat Image lb & ulb ###
                img_id = img_id + ulb_img_id
                img_w = torch.cat([img_w, ulb_img_w], dim=0)
                img_s = torch.cat([img_s, ulb_img_s], dim=0)
                # Concat Strong Aug. options ###
                for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                    ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                    ops[i][1] = torch.cat([v, ulb_v], dim=0)

            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pred = model(img_w[:B])

            ### Teacher (for ulb)
            ema.apply_shadow()
            with torch.no_grad():
                ulb_pred1, ulb_cam1 = model(img_w, forward_cam=True)  ###
                ulb_cam1[:B,:-1] *= label[:,:,None,None]

                ### Apply strong transforms to pseudo-label(pixelwise matching with ulb_cam2) ###
                if args.ulb_aug_type == 'strong':
                    ulb_cam1_s = apply_strong_tr(ulb_cam1, ops, strong_transforms=strong_transforms)
                    # ulb_cam_rv1_s = apply_strong_tr(ulb_cam_rv1, ops2, strong_transforms=strong_transforms)
                else: # weak aug
                    ulb_cam1_s = ulb_cam1
                
                ### Cutmix 
                if args.use_cutmix:
                    img_s, ulb_cam1_s = cutmix(img_s, ulb_cam1_s)
                    # ulb_img2, ulb_cam1_s, ulb_feat1_s = cutmix(ulb_img2, ulb_cam1_s, ulb_feat1_s)

                ### Make strong augmented (transformed) prediction for MT ###
                if 1 in args.ssl_type:
                    ulb_pred1_s = F.avg_pool2d(ulb_cam1_s, kernel_size=(ulb_cam1_s.size(-2), ulb_cam1_s.size(-1)), padding=0)
                    ulb_pred1_s = ulb_pred1_s.view(ulb_pred1_s.size(0), -1)
                else:
                    ulb_pred1_s = ulb_pred1
                ### Make masks for pixel-wise MT ###
                mask_s = torch.ones_like(ulb_cam1)
                if 2 in args.ssl_type or 4 in args.ssl_type :
                    mask_s = apply_strong_tr(mask_s, ops, strong_transforms=strong_transforms)

            ema.restore()
            ###

            ### Student (for ulb)
            ulb_pred2, ulb_cam2 = model(img_s, forward_cam=True) ###

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

            ###########           Semi-supervsied Learning Loss           ###########
            ssl_pack = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2[:,:-1], cam_t=ulb_cam1_s[:,:-1], mask=mask_s)
            loss_ssl = ssl_pack['loss_ssl']

            loss = loss_cls + loss_ssl

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_ssl': loss_ssl.item()})

            if 1 in args.ssl_type:
                avg_meter.add({'loss_mt'      : ssl_pack['loss_mt'].item(),
                               'mt_mask_ratio': ssl_pack['mask_mt'].item()})
            if 2 in args.ssl_type:
                avg_meter.add({'loss_pmt'     : ssl_pack['loss_pmt'].item()})
            if 3 in args.ssl_type:
                avg_meter.add({'loss_pl'      : ssl_pack['loss_pl'].item(),
                               'mask_ratio'   : ssl_pack['mask_pl'].item()})
            if 4 in args.ssl_type:
                avg_meter.add({'loss_con'     : ssl_pack['loss_con'].item()})
            if 5 in args.ssl_type:
                avg_meter.add({'loss_cdc'     : ssl_pack['loss_cdc'].item(),
                               'cdc_pos_ratio': ssl_pack['mask_cdc_pos'].item(),
                               'cdc_neg_ratio': ssl_pack['mask_cdc_neg'].item()})
            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update() ########

            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                # Print Logs
                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')), end=' ')
                # SSL Losses
                for k, v in ssl_pack.items():
                    print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')                    
                print('imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s'      % (timer.str_est_finish()),
                      'lr: %.4f'    % (tscalar['train/lr']), flush=True)
                
                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                    
                tscalar.clear()

            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### SEAM + semi-supervsied learning ###
def train_seam_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_ssl', 'p_cutoff']
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')
        log_keys.append('mt_mask_ratio')
    if 2 in args.ssl_type:
        log_keys.append('loss_pmt')
    if 3 in args.ssl_type:
        log_keys.append('loss_pl')
        log_keys.append('mask_ratio')
        log_keys.append('p_cutoff')
    if 4 in args.ssl_type:
        log_keys.append('loss_con')
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")

    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###

    # EMA
    #ema_model = deepcopy(model)
    ema = EMA(model, args.ema_m)
    ema.register()

    ### Model Watch (log_freq=val_freq)
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times

    print(args)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img_w, img_s, ops, label = next(lb_loader_iter)
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img_w, img_s, ops, label = next(lb_loader_iter)
            B = len(img_id)

            ### Unlabeled ###
            if train_ulb_dataloader:
                try:
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                except:
                    ulb_loader_iter = iter(train_ulb_dataloader)        ###
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)

                # Concat Image lb & ulb ###
                img_id = img_id + ulb_img_id
                img_w = torch.cat([img_w, ulb_img_w], dim=0)
                img_s = torch.cat([img_s, ulb_img_s], dim=0)
                # Concat Strong Aug. options ###
                for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                    ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                    ops[i][1] = torch.cat([v, ulb_v], dim=0)

            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            img2 = F.interpolate(img_w[:B], size=(128, 128), mode='bilinear', align_corners=True)

            # Forward (only labeled)
            pred1, cam1, pred_rv1, cam_rv1 = model(img_w[:B])
            pred2, cam2, pred_rv2, cam_rv2 = model(img2)

            ### Teacher (for ulb)
            ema.apply_shadow()
            with torch.no_grad():
                pred_w, _, pred_rv_w, cam_w = model(img_w)  ###
                # Make CAM (use label)
                cam_w[:B,:-1] *= label[:,:,None,None]

                ### Apply strong transforms to pseudo-label(pixelwise matching with ulb_cam2) ###
                if args.ulb_aug_type == 'strong':
                    cam_s_t = apply_strong_tr(cam_w, ops, strong_transforms=strong_transforms)
                    # cam_rv_s_t = apply_strong_tr(cam_rv_w, ops, strong_transforms=strong_transforms)
                else: # weak aug
                    cam_s_t = cam_w
                
                ### Cutmix 
                if args.use_cutmix:
                    img_s, cam_s_t = cutmix(img_s, cam_s_t)
                    # img_s, cam_s_t, feat_s = cutmix(ulb_img2, cam_s_t, ulb_feat1_s)

                ### Make strong augmented (transformed) prediction for MT ###
                if 1 in args.ssl_type:
                    pred_s_t = F.avg_pool2d(cam_s_t, kernel_size=(cam_s_t.size(-2), cam_s_t.size(-1)), padding=0)
                    pred_s_t = pred_s_t.view(pred_s_t.size(0), -1)
                else:
                    pred_s_t = pred_w
                ### Make masks for pixel-wise MT ###
                mask_s = torch.ones_like(cam_w)
                if 2 in args.ssl_type or 4 in args.ssl_type :
                    mask_s = apply_strong_tr(mask_s, ops, strong_transforms=strong_transforms)

            ema.restore()
            ###

            ### Student (for ulb)
            pred_s, _, pred_rv_s, cam_s = model(img_s) ###

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)

            bg_score = torch.ones((B, 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # total loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.

            ###########           Semi-supervsied Learning Loss           ###########
            ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=pred_s, pred_t=pred_s_t, cam_s=cam_s, cam_t=cam_s_t, mask=mask_s)
            loss_ssl = ssl_pack['loss_ssl']
            
            loss = loss_cls + loss_er + loss_ecr + loss_ssl # NO saliency for SEAM
            # SEAM Logging
            # Logging AVGMeter
            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_ssl': loss_ssl.item(),
                           'p_cutoff': cutoff_value
                           })
            if 1 in args.ssl_type:
                avg_meter.add({'loss_mt'      : ssl_pack['loss_mt'].item(),
                               'mt_mask_ratio': ssl_pack['mask_mt'].item()})
            if 2 in args.ssl_type:
                avg_meter.add({'loss_pmt'     : ssl_pack['loss_pmt'].item()})
            if 3 in args.ssl_type:
                avg_meter.add({'loss_pl'      : ssl_pack['loss_pl'].item(),
                               'mask_ratio'   : ssl_pack['mask_pl'].item()})
            if 4 in args.ssl_type:
                avg_meter.add({'loss_con'     : ssl_pack['loss_con'].item()})
            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update() ########

            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                # Print Logs
                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                      'p_cutoff:%.4f' % (avg_meter.get('p_cutoff')),
                       end=' ')
                # SSL Losses
                for k, v in ssl_pack.items():
                    print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')
                print('imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (tscalar['train/lr']), flush=True)
                
                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)

                tscalar.clear()
            
            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
                    
                    # ### EMA model ###
                    # ema.apply_shadow() ###
                    # print()
                    # print('Validating Teacher(EMA) Model... ')
                    # validate(args, model, val_dataloader, iteration, tag='val_ema')
                    # ema.restore() ###

            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### EPS + semi-supervsied learning ###
def train_eps_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_ssl']
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')
        log_keys.append('mt_mask_ratio')
    if 2 in args.ssl_type:
        log_keys.append('loss_pmt')
    if 3 in args.ssl_type:
        log_keys.append('loss_pl')
        log_keys.append('mask_ratio')
    if 4 in args.ssl_type:
        log_keys.append('loss_con')
    if 5 in args.ssl_type:
        log_keys.append('loss_cdc')
        log_keys.append('ratio_cdc_pos')
        log_keys.append('ratio_cdc_neg')
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")
    
    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###

    # EMA
    #ema_model = deepcopy(model)
    ema = EMA(model, args.ema_m)
    ema.register()

    ### Model Watch (log_freq=val_freq)
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times

    print(args)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
            B = len(img_id)

            ### Unlabeled ###
            if train_ulb_dataloader:
                try:
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                except:
                    ulb_loader_iter = iter(train_ulb_dataloader)        ###
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)

                # Concat Image lb & ulb ###
                img_id = img_id + ulb_img_id
                img_w = torch.cat([img_w, ulb_img_w], dim=0)
                img_s = torch.cat([img_s, ulb_img_s], dim=0)
                # Concat Strong Aug. options ###
                for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                    ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                    ops[i][1] = torch.cat([v, ulb_v], dim=0)

            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pred, cam = model(img_w[:B])

            ### Teacher (for ulb)
            ema.apply_shadow()
            with torch.no_grad():
                ulb_pred1, ulb_cam1 = model(img_w)  ###
                ulb_cam1[:B,:-1] *= label[:,:,None,None]

                ### Apply strong transforms to pseudo-label(pixelwise matching with ulb_cam2) ###
                if args.ulb_aug_type == 'strong':
                    ulb_cam1_s = apply_strong_tr(ulb_cam1, ops, strong_transforms=strong_transforms)
                    # ulb_cam_rv1_s = apply_strong_tr(ulb_cam_rv1, ops2, strong_transforms=strong_transforms)
                else: # weak aug
                    ulb_cam1_s = ulb_cam1
                
                ### Cutmix 
                if args.use_cutmix:
                    img_s, ulb_cam1_s = cutmix(img_s, ulb_cam1_s)
                    # ulb_img2, ulb_cam1_s, ulb_feat1_s = cutmix(ulb_img2, ulb_cam1_s, ulb_feat1_s)

                ### Make strong augmented (transformed) prediction for MT ###
                if 1 in args.ssl_type:
                    ulb_pred1_s = F.avg_pool2d(ulb_cam1_s, kernel_size=(ulb_cam1_s.size(-2), ulb_cam1_s.size(-1)), padding=0)
                    ulb_pred1_s = ulb_pred1_s.view(ulb_pred1_s.size(0), -1)
                else:
                    ulb_pred1_s = ulb_pred1
                ### Make masks for pixel-wise MT ###
                mask_s = torch.ones_like(ulb_cam1)
                if 2 in args.ssl_type or 4 in args.ssl_type :
                    mask_s = apply_strong_tr(mask_s, ops, strong_transforms=strong_transforms)

            ema.restore()
            ###

            ### Student (for ulb)
            ulb_pred2, ulb_cam2 = model(img_s) ###

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam,
                                                              saliency,
                                                              label,
                                                              args.tau,
                                                              args.alpha,
                                                              intermediate=True,
                                                              num_class=args.num_sample)

            ###########           Semi-supervsied Learning Loss           ###########
            ssl_pack = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2, cam_t=ulb_cam1_s, mask=mask_s)
            loss_ssl = ssl_pack['loss_ssl']

            loss = loss_cls + loss_sal + loss_ssl


            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_ssl': loss_ssl.item()})
            if 1 in args.ssl_type:
                avg_meter.add({'loss_mt'      : ssl_pack['loss_mt'].item(),
                               'mt_mask_ratio': ssl_pack['mask_mt'].item()})
            if 2 in args.ssl_type:
                avg_meter.add({'loss_pmt'     : ssl_pack['loss_pmt'].item()})
            if 3 in args.ssl_type:
                avg_meter.add({'loss_pl'      : ssl_pack['loss_pl'].item(),
                               'mask_ratio'   : ssl_pack['mask_pl'].item()})
            if 4 in args.ssl_type:
                avg_meter.add({'loss_con'     : ssl_pack['loss_con'].item()})
            if 5 in args.ssl_type:
                avg_meter.add({'loss_cdc'     : ssl_pack['loss_cdc'].item(),
                               'cdc_pos_ratio': ssl_pack['mask_cdc_pos'].item(),
                               'cdc_neg_ratio': ssl_pack['mask_cdc_neg'].item()})
            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update() ########

            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']
                
                # Print Logs
                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')), end=' ')
                # SSL Losses
                for k, v in ssl_pack.items():
                    print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')
                print('imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s'      % (timer.str_est_finish()),
                      'lr: %.4f'    % (tscalar['train/lr']), flush=True)

                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)

                tscalar.clear()

            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 

            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### contrast + semi-supervised learning ###
def train_contrast_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr', 'loss_ssl', 'p_cutoff']
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')  # loss of Mean Teacher
        log_keys.append('mt_mask_ratio')
    if 2 in args.ssl_type:
        log_keys.append('loss_pmt')
    if 3 in args.ssl_type:
        log_keys.append('loss_pl')
        log_keys.append('mask_ratio')
        log_keys.append('p_cutoff')
    if 4 in args.ssl_type:
        log_keys.append('loss_con')
    if 5 in args.ssl_type:
        log_keys.append('loss_cdc')
        log_keys.append('ratio_cdc_pos')
        log_keys.append('ratio_cdc_neg')
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")

    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###
    
    # EMA
    #ema_model = deepcopy(model)
    ema = EMA(model, args.ema_m)
    ema.register()
    
    ### Model Watch (log_freq=val_freq)
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times

    gamma = 0.10
    print(args)
    print('Using Gamma:', gamma)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            # Labeled
            try:
                img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
            B = len(img_id)

            ### Unlabeled ###
            if train_ulb_dataloader:
                try:
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                except:
                    ulb_loader_iter = iter(train_ulb_dataloader)        ###
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)

                # Concat Image lb & ulb ###
                img_id = img_id + ulb_img_id
                img_w = torch.cat([img_w, ulb_img_w], dim=0)
                img_s = torch.cat([img_s, ulb_img_s], dim=0)
                # Concat Strong Aug. options ###
                for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                    ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                    ops[i][1] = torch.cat([v, ulb_v], dim=0)
                
            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # Resize
            img2 = F.interpolate(img_w[:B], size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            # Forward (only labeled)
            pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img_w[:B]) # Whole Images (for SSL)
            pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)
            
            ### Teacher Model
            ema.apply_shadow()
            with torch.no_grad():
                pred_w, cam_w, pred_rv_w, cam_rv_w, feat_w = model(img_w) # Whole Images (for SSL)
                # Make CAM (use label)
                cam_w[:B,:-1] *= label[:,:,None,None]

                ### Apply strong transforms to pseudo-label(pixelwise matching with ulb_cam2) ###
                if args.ulb_aug_type == 'strong':
                    cam_s_t = apply_strong_tr(cam_w, ops, strong_transforms=strong_transforms)
                    # cam_rv_s_t = apply_strong_tr(cam_rv_w, ops, strong_transforms=strong_transforms)
                    # if 5 in args.ssl_type:
                    #     feat_s_t = apply_strong_tr(feat_w, ops, strong_transforms=strong_transforms)
                else: # weak aug
                    cam_s_t = cam_w
                
                ### Cutmix 
                if args.use_cutmix:
                    img_s, cam_s_t = cutmix(img_s, cam_s_t)
                    # img_s, cam_s_t, feat_s = cutmix(ulb_img2, cam_s_t, ulb_feat1_s)

                ### Make strong augmented (transformed) prediction for MT ###
                if 1 in args.ssl_type:
                    pred_s_t = F.avg_pool2d(cam_s_t, kernel_size=(cam_s_t.size(-2), cam_s_t.size(-1)), padding=0)
                    pred_s_t = pred_s_t.view(pred_s_t.size(0), -1)
                else:
                    pred_s_t = pred_w
                ### Make masks for pixel-wise MT ###
                mask_s = torch.ones_like(cam_w)
                if 2 in args.ssl_type or 4 in args.ssl_type :
                    mask_s = apply_strong_tr(mask_s, ops, strong_transforms=strong_transforms)

            ema.restore()
            ###

            ### Student
            if 5 not in args.ssl_type:
                pred_s, cam_s, _, _, feat_s = model(img_s) ###
            else:
                pred_s, cam_s, _, _, feat_low_s, feat_s = model(img_s, require_feats_high=True)  ###


            # Classification & EPS loss 1
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

            # Classification & EPS loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

            # Classification & EPS loss (rv)
            bg_score = torch.ones((B, 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
            
            # Classification & EPS loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            # SEAM Losses
            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # PPC loss
            loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)

            ###########           Semi-supervsied Learning Loss           ###########
            ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=pred_s, pred_t=pred_s_t, cam_s=cam_s, cam_t=cam_s_t, feat_s=feat_s, feat_t=None, mask=mask_s)
            loss_ssl = ssl_pack['loss_ssl']

            loss = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce + loss_ssl
            
            # Logging AVGMeter
            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er' : loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_ssl': loss_ssl.item(),
                           'p_cutoff': cutoff_value
                           })
            if 1 in args.ssl_type:
                avg_meter.add({'loss_mt'      : ssl_pack['loss_mt'].item(),
                               'mt_mask_ratio': ssl_pack['mask_mt'].item()})
            if 2 in args.ssl_type:
                avg_meter.add({'loss_pmt'     : ssl_pack['loss_pmt'].item()})
            if 3 in args.ssl_type:
                avg_meter.add({'loss_pl'      : ssl_pack['loss_pl'].item(),
                               'mask_ratio'   : ssl_pack['mask_pl'].item()})
            if 4 in args.ssl_type:
                avg_meter.add({'loss_con'     : ssl_pack['loss_con'].item()})
            if 5 in args.ssl_type:
                avg_meter.add({'loss_cdc'     : ssl_pack['loss_cdc'].item(),
                               'cdc_pos_ratio': ssl_pack['mask_cdc_pos'].item(),
                               'cdc_neg_ratio': ssl_pack['mask_cdc_neg'].item()})
            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update() ########
                    
            # Logging
            if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                timer.update_progress(optimizer.global_step / max_step)
                tscalar['train/lr'] = optimizer.param_groups[0]['lr']
                
                # Print Logs
                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                      'Loss_Nce:%.4f' % (avg_meter.get('loss_nce')),
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                      'p_cutoff:%.4f' % (avg_meter.get('p_cutoff')),
                       end=' ')
                # SSL Losses
                for k, v in ssl_pack.items():
                    print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')
                print('imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s'      % (timer.str_est_finish()),
                      'lr: %.4f'    % (tscalar['train/lr']), flush=True)

                ### wandb logging Scalars, Histograms, Images ###
                if args.use_wandb:
                    wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                    wandb.log({k: v for k, v in tscalar.items()}, step=iteration)

                tscalar.clear()
            
            # Validate K times
            current_step = optimizer.global_step-(max_step % val_freq)
            if current_step and current_step % val_freq == 0:
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate(args, model, val_dataloader, iteration, tag='val') 
                    
                    # ### EMA model ###
                    # ema.apply_shadow() ###
                    # print()
                    # print('Validating Teacher(EMA) Model... ')
                    # validate(args, model, val_dataloader, iteration, tag='val_ema')
                    # ema.restore() ###

            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))
