import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import functional as tvf
import wandb

import os
import re 
import random
import numpy as np
from glob import glob
from copy import deepcopy
import natsort as nsort

from util import pyutils
from data.augmentation.randaugment import tensor_augment_list

from module.validate import *
from module.loss import adaptive_min_pooling_loss, get_er_loss, get_eps_loss, get_contrast_loss
from module.ssl import *
from module.helper import calc_score, get_masks_by_confidence

# Control Randomness\
random_seed = 7
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

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
    # tb_writer = SummaryWriter(args.log_folder)
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    print(args)
    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times ### validation logging

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

            # loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # total loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
            loss = loss_cls + loss_er + loss_ecr

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
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
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
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

            # loss = loss_cls + loss_sal + loss_nce + loss_er + loss_ecr
            loss = loss_cls + loss_nce + loss_er + loss_ecr

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

def train_cls2(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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
            ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2[:,:-1], cam_t=ulb_cam1_s[:,:-1], mask=mask_s)
            loss_ssl = ssl_pack['loss_ssl']

            loss = loss_cls
            masks, _max_probs = get_masks_by_confidence(cam=ulb_cam1_s[:,:-1])
                
            loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-7)
            loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-7)
            loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-7)
            loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-7)
            loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-7)
            loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-7)

            ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()

            # Logging AVGMeter
            avg_meter.add({
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_ssl': loss_ssl.item(),
                'loss_ssl_1': loss_ssl_1.item(),
                'loss_ssl_2': loss_ssl_2.item(),
                'loss_ssl_3': loss_ssl_3.item(),
                'loss_ssl_4': loss_ssl_4.item(),
                'loss_ssl_5': loss_ssl_5.item(),
                'loss_ssl_6': loss_ssl_6.item(),
                'mask_1' : masks[0].float().mean().item(),
                'mask_2' : masks[1].float().mean().item(),
                'mask_3' : masks[2].float().mean().item(),
                'mask_4' : masks[3].float().mean().item(),
                'mask_5' : masks[4].float().mean().item(),
                'mask_6' : masks[5].float().mean().item(),
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
                      'Loss_Cls:%.4f' % (avg_meter.get('loss_cls'))
                      )

                print('                 Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                    'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                    'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                    'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                    'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                    'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                )
                print('                 mask1:%.4f' % (avg_meter.get('mask_1')),
                    'mask2:%.4f' % (avg_meter.get('mask_2')),
                    'mask3:%.4f' % (avg_meter.get('mask_3')),
                    'mask4:%.4f' % (avg_meter.get('mask_4')),
                    'mask5:%.4f' % (avg_meter.get('mask_5')),
                    'mask6:%.4f' % (avg_meter.get('mask_6')),
                    'p_cutoff:%.4f' % cutoff_value,
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

            if args.test: 
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
        
            else:
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
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
            
            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))

### SEAM + semi-supervsied learning ###
def train_seam2(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls',  'loss_er', 'loss_ecr', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")

    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###

    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
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
                pred_w, cam_w, pred_rv_w, cam_rv_w = model(img_w)  ###
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
            loss = loss_cls + loss_er + loss_ecr

            # Logging by confidence range
            masks, _max_probs = get_masks_by_confidence(cam=cam_s_t)

            loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-5)
            loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-5)
            loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-5)
            loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-5)
            loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-5)
            loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-5)
            
            ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()
            
            avg_meter.add({
                'loss': loss.item(),
                'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(),
                'loss_ecr': loss_ecr.item(),
                'loss_ssl': loss_ssl.item(),
                'loss_ssl_1': loss_ssl_1.item(),
                'loss_ssl_2': loss_ssl_2.item(),
                'loss_ssl_3': loss_ssl_3.item(),
                'loss_ssl_4': loss_ssl_4.item(),
                'loss_ssl_5': loss_ssl_5.item(),
                'loss_ssl_6': loss_ssl_6.item(),
                'mask_1' : masks[0].mean().item(),
                'mask_2' : masks[1].mean().item(),
                'mask_3' : masks[2].float().mean().item(),
                'mask_4' : masks[3].float().mean().item(),
                'mask_5' : masks[4].float().mean().item(),
                'mask_6' : masks[5].float().mean().item(),
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
                      'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')))
                print('                 Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                      'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                      'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                      'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                      'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                      'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                )
                print('                 masks[0]:%.4f' % (avg_meter.get('mask_1')),
                      'mask_1:%.4f' % (avg_meter.get('mask_2')),
                      'mask_2:%.4f' % (avg_meter.get('mask_3')),
                      'mask_3:%.4f' % (avg_meter.get('mask_4')),
                      'mask_4:%.4f' % (avg_meter.get('mask_5')),
                      'mask_5:%.4f' % (avg_meter.get('mask_6')),
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
                    np_hist = np.histogram(_max_probs.detach().cpu().numpy(), range=[0.0, 1.0], bins=100)
                    wandb.log({'train/mask_hist': wandb.Histogram(np_histogram=np_hist, num_bins=100)})

                tscalar.clear()

            if args.test: 
                # Save intermediate model
                model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                torch.save(model.module.state_dict(), model_path)
                print(f'Model {model_path} Saved.')

                # Validation
                if val_dataloader is not None:
                    print('Validating Student Model... ')
                    validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
            else:
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
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))

def train_eps2(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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
    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    else:
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
                ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2, cam_t=ulb_cam1_s, mask=mask_s)  
                loss_ssl = ssl_pack['loss_ssl']
                loss = loss_cls + loss_sal 

                # Logging by confidence range
                masks, _max_probs = get_masks_by_confidence(cam=ulb_cam1_s)

                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-7)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-7)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-7)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-7)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-7)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-7)

                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()

                avg_meter.add({
                    'loss': loss.item(),
                    'loss_cls': loss_cls.item(),
                    'loss_sal': loss_sal.item(),
                    'loss_ssl': loss_ssl.item(),
                    'loss_ssl_1': loss_ssl_1.item(),
                    'loss_ssl_2': loss_ssl_2.item(),
                    'loss_ssl_3': loss_ssl_3.item(),
                    'loss_ssl_4': loss_ssl_4.item(),
                    'loss_ssl_5': loss_ssl_5.item(),
                    'loss_ssl_6': loss_ssl_6.item(),
                    'mask_1' : masks[0].mean().item(),
                    'mask_2' : masks[1].mean().item(),
                    'mask_3' : masks[2].float().mean().item(),
                    'mask_4' : masks[3].float().mean().item(),
                    'mask_5' : masks[4].float().mean().item(),
                    'mask_6' : masks[5].float().mean().item(),
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
                    print(
                        'Iter:%5d/%5d' % (iteration, args.max_iters),
                        'Loss:%.4f' % (avg_meter.get('loss')),
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                        'Loss_Sal:%.4f' % (avg_meter.get('loss_sal'))
                    )
                    print(
                        'Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print(
                        'mask1:%.4f' % (avg_meter.get('mask_1')),
                        'mask2:%.4f' % (avg_meter.get('mask_2')),
                        'mask3:%.4f' % (avg_meter.get('mask_3')),
                        'mask4:%.4f' % (avg_meter.get('mask_4')),
                        'mask5:%.4f' % (avg_meter.get('mask_5')),
                        'mask6:%.4f' % (avg_meter.get('mask_6')),
                        end=' ')
                    # SSL Losses
                    for k, v in ssl_pack.items():
                        print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')
                    print(
                        'imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                        'Fin:%s'      % (timer.str_est_finish()),
                        'lr: %.4f'    % (tscalar['train/lr']), flush=True)

                    ### wandb logging Scalars, Histograms, Images ###
                    if args.use_wandb:
                        wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                        wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                        # np_hist = np.histogram(_max_probs.detach().cpu().numpy(), range=[0.0, 1.0], bins=100)
                        # wandb.log({'train/mask_hist': wandb.Histogram(np_histogram=np_hist, num_bins=100)})
                    tscalar.clear()

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val')

                timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))

### contrast + semi-supervised learning ###
def train_contrast2(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6' ]
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')  # loss of Mean Teacher
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
    
    if args.val_only:
        print("Val-only mode.")
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state
        print('weight path: ', weight_path)
        
        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    else:
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
                # divide by confidence range
                ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=pred_s, pred_t=pred_s_t, cam_s=cam_s, cam_t=cam_s_t,  mask=mask_s)
                loss_ssl = ssl_pack['loss_ssl']
                loss = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce

                masks, _max_probs = get_masks_by_confidence(cam=cam_s_t)

                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-7)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-7)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-7)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-7)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-7)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-7)

                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()
                
                # Logging AVGMeter
                avg_meter.add({'loss': loss.item(),
                            'loss_ssl': loss_ssl.item(),
                            'loss_ssl_1': loss_ssl_1.item(),
                            'loss_ssl_2': loss_ssl_2.item(),
                            'loss_ssl_3': loss_ssl_3.item(),
                            'loss_ssl_4': loss_ssl_4.item(),
                            'loss_ssl_5': loss_ssl_5.item(),
                            'loss_ssl_6': loss_ssl_6.item(),
                            'mask_1' : masks[0].mean().item(),
                            'mask_2' : masks[1].mean().item(),
                            'mask_3' : masks[2].float().mean().item(),
                            'mask_4' : masks[3].float().mean().item(),
                            'mask_5' : masks[4].float().mean().item(),
                            'mask_6' : masks[5].float().mean().item(),
                            'loss_cls': loss_cls.item(),
                            'loss_sal': loss_sal.item(),
                            'loss_nce': loss_nce.item(),
                            'loss_er' : loss_er.item(),
                            'loss_ecr': loss_ecr.item(),
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
                    )
                    print('                 Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print('                 mask1:%.4f' % (avg_meter.get('mask_1')),
                        'mask2:%.4f' % (avg_meter.get('mask_2')),
                        'mask3:%.4f' % (avg_meter.get('mask_3')),
                        'mask4:%.4f' % (avg_meter.get('mask_4')),
                        'mask5:%.4f' % (avg_meter.get('mask_5')),
                        'mask6:%.4f' % (avg_meter.get('mask_6')),
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
                        np_hist = np.histogram(_max_probs.detach().cpu().numpy(), range=[0.0, 1.0], bins=100)
                        wandb.log({'train/mask_hist': wandb.Histogram(np_histogram=np_hist, num_bins=100)})

                    tscalar.clear()

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 

                timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))

def train_cls_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    
    log_keys = ['loss', 'loss_cls', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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

    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    
    else:
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
                ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2[:,:-1], cam_t=ulb_cam1_s[:,:-1], mask=mask_s)
                loss_ssl = ssl_pack['loss_ssl']

                loss = loss_cls + loss_ssl
                masks, _max_probs = get_masks_by_confidence(cam=ulb_cam1_s[:,:-1])
                    
                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-7)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-7)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-7)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-7)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-7)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-7)

                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()

                # Logging AVGMeter
                avg_meter.add({
                    'loss': loss.item(),
                    'loss_cls': loss_cls.item(),
                    'loss_ssl': loss_ssl.item(),
                    'loss_ssl_1': loss_ssl_1.item(),
                    'loss_ssl_2': loss_ssl_2.item(),
                    'loss_ssl_3': loss_ssl_3.item(),
                    'loss_ssl_4': loss_ssl_4.item(),
                    'loss_ssl_5': loss_ssl_5.item(),
                    'loss_ssl_6': loss_ssl_6.item(),
                    'mask_1' : masks[0].float().mean().item(),
                    'mask_2' : masks[1].float().mean().item(),
                    'mask_3' : masks[2].float().mean().item(),
                    'mask_4' : masks[3].float().mean().item(),
                    'mask_5' : masks[4].float().mean().item(),
                    'mask_6' : masks[5].float().mean().item(),
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
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls'))
                        )

                    print('                 Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print('                 mask1:%.4f' % (avg_meter.get('mask_1')),
                        'mask2:%.4f' % (avg_meter.get('mask_2')),
                        'mask3:%.4f' % (avg_meter.get('mask_3')),
                        'mask4:%.4f' % (avg_meter.get('mask_4')),
                        'mask5:%.4f' % (avg_meter.get('mask_5')),
                        'mask6:%.4f' % (avg_meter.get('mask_6')),
                        'p_cutoff:%.4f' % cutoff_value,
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

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
            
                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
                
                timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### SEAM + semi-supervsied learning ###
def train_seam_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls',  'loss_er', 'loss_ecr', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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
    avg_meter = pyutils.AverageMeter(*log_keys) ###
    timer = pyutils.Timer("Session started: ")

    # DataLoader
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None ###
    strong_transforms = tensor_augment_list() ###

    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val')
    else:   
        # EMA
        # ema_model = deepcopy(model)
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
                pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img_w[:B])
                pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)

                ### Teacher (for ulb)
                ema.apply_shadow()
                with torch.no_grad():
                    pred_w, cam_w, pred_rv_w, cam_rv_w, feat_w = model(img_w)
                    # Make CAM (use label)
                    cam_w[:B,:-1] *= label[:,:,None,None]

                    ### Apply strong transforms to pseudo-label(pixelwise matching with ulb_cam2) ###
                    if args.ulb_aug_type == 'strong':
                        cam_s_t = apply_strong_tr(cam_w, ops, strong_transforms=strong_transforms)
                        feat_s_t = apply_strong_tr(feat_w, ops, strong_transforms=strong_transforms)
                        # cam_rv_s_t = apply_strong_tr(cam_rv_w, ops, strong_transforms=strong_transforms)
                    else: # weak aug
                        cam_s_t = cam_w
                        feat_s_t = feat_w
                    ### Cutmix 
                    if args.use_cutmix:
                        img_s, cam_s_t, feat_s_t = cutmix(img_s, cam_s_t, feat_s_t)
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
                pred_s, _, pred_rv_s, cam_s, feat_s = model(img_s) ###

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

                # Logging by confidence range
                masks, max_probs = get_masks_by_confidence(cam=cam_s_t)

                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-5)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-5)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-5)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-5)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-5)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-5)
                
                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()
                
                ############       Pseudo Label Propagation      ############
                # get k anchors: top-k highest confidence
                # pred_w:       (B, 21)
                # cam_s:        (B, 21, 56, 56)
                # max_probs:    (B, 56, 56) - 가장 높은 class confidence
                # feat_s_t:     (B, 128, 56, 56)

                if args.anchor_k or args.anchor_thr:
                    nn_masks = torch.zeros_like(feat_s_t, dtype=torch.long)
                    feat_s = torch.softmax(feat_s, dim=1)
                    feat_s_t = torch.softmax(feat_s_t, dim=1)

                    for max_prob in max_probs:
                        if args.anchor_k:
                            anc_values, _ = max_prob.flatten().topk(k=args.anchor_k)
                        elif args.anchor_thr:
                            anc_values = max_prob.flatten()
                            # anc_values = max_prob.flatten().sort(descending=True, stable=True).values
                            anc_values = anc_values[anc_values > args.anchor_thr]

                        # 전체 Batch, Channel의 feature space에서 anc와 가장 가까운 top L 개를 뽑음
                        for anc in anc_values:
                            dist = torch.abs(feat_s_t - anc).flatten()
                            # dist = torch.linalg.vector_norm(feat_s_t - anc, dim=0)    # (, 3136)
                            dis, idx = dist.topk(args.nn_l, largest=False)
                            nn_idx = np.array(np.unravel_index(idx.cpu().numpy(), feat_s_t.shape)).T                          
                            
                            for idx in nn_idx:
                                n, c, h, w = idx[0], idx[1], idx[2], idx[3]
                                nn_masks[n, c, h, w] = 1  

                    # import code; code.interact(local=vars())
                    # loss_feat, _, _, _, _ = consistency_loss(feat_s, feat_s_t, 'ce', args.T, cutoff_value, args.soft_label)
                    # loss_feat = ce_loss(feat_s, feat_s_t, reduction='none') * nn_masks
                    loss_feat = F.mse_loss(feat_s, feat_s_t, reduction='none') * nn_masks
                    loss_feat = loss_feat.sum() / (nn_masks.sum() + 1e-6)

                else:
                    loss_feat = torch.tensor(0)

                loss = loss_cls + loss_er + loss_ecr + loss_ssl + loss_feat

                avg_meter.add({
                    'loss': loss.item(),
                    'loss_cls': loss_cls.item(),
                    'loss_er': loss_er.item(),
                    'loss_ecr': loss_ecr.item(),
                    'loss_ssl': loss_ssl.item(),
                    'loss_ssl_1': loss_ssl_1.item(),
                    'loss_ssl_2': loss_ssl_2.item(),
                    'loss_ssl_3': loss_ssl_3.item(),
                    'loss_ssl_4': loss_ssl_4.item(),
                    'loss_ssl_5': loss_ssl_5.item(),
                    'loss_ssl_6': loss_ssl_6.item(),
                    'mask_1' : masks[0].mean().item(),
                    'mask_2' : masks[1].mean().item(),
                    'mask_3' : masks[2].float().mean().item(),
                    'mask_4' : masks[3].float().mean().item(),
                    'mask_5' : masks[4].float().mean().item(),
                    'mask_6' : masks[5].float().mean().item(),
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
                    print(
                        'Iter:%5d/%5d' % (iteration, args.max_iters),
                        'Loss:%.4f' % (avg_meter.get('loss')),
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                        'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                        'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')))
                    print(
                        'Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print(
                        'mask_1:%.4f' % (avg_meter.get('mask_1')),
                        'mask_2:%.4f' % (avg_meter.get('mask_2')),
                        'mask_3:%.4f' % (avg_meter.get('mask_3')),
                        'mask_4:%.4f' % (avg_meter.get('mask_4')),
                        'mask_5:%.4f' % (avg_meter.get('mask_5')),
                        'mask_6:%.4f' % (avg_meter.get('mask_6')),
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

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
        
                timer.reset_stage()
                
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### EPS + semi-supervsied learning ###
def train_eps_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_ssl', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6']
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

    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    else:
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

                ### Student (for ulb)
                ulb_pred2, ulb_cam2 = model(img_s) ###

                # Classification loss
                loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

                loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)

                ###########           Semi-supervsied Learning Loss           ###########
                ssl_pack, cutoff_value = get_ssl_loss(args, iteration, pred_s=ulb_pred2, pred_t=ulb_pred1_s, cam_s=ulb_cam2, cam_t=ulb_cam1_s, mask=mask_s)  
                loss_ssl = ssl_pack['loss_ssl']

                # Logging by confidence range
                masks, max_probs = get_masks_by_confidence(cam=ulb_cam1_s)

                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-7)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-7)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-7)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-7)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-7)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-7)

                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()

                ############       Pseudo Label Propagation      ############
                # get k anchors: top-k highest confidence
                # pred_w:       (B, 21)
                # cam_s:        (B, 21, 56, 56)
                # max_probs:    (B, 56, 56) - 가장 높은 class confidence
                # feat_s_t:     (B, 128, 56, 56)

                if args.anchor_k or args.anchor_thr:
                    nn_masks = torch.zeros_like(feat_s_t, dtype=torch.long)
                    feat_s = torch.softmax(feat_s, dim=1)
                    feat_s_t = torch.softmax(feat_s_t, dim=1)

                    for max_prob in max_probs:
                        if args.anchor_k:
                            anc_values, _ = max_prob.flatten().topk(k=args.anchor_k)
                        elif args.anchor_thr:
                            anc_values = max_prob.flatten()
                            # anc_values = max_prob.flatten().sort(descending=True, stable=True).values
                            anc_values = anc_values[anc_values > args.anchor_thr]

                        # 전체 Batch, Channel의 feature space에서 anc와 가장 가까운 top L 개를 뽑음
                        for anc in anc_values:
                            dist = torch.abs(feat_s_t - anc).flatten()
                            # dist = torch.linalg.vector_norm(feat_s_t - anc, dim=0)    # (, 3136)
                            dis, idx = dist.topk(args.nn_l, largest=False)
                            nn_idx = np.array(np.unravel_index(idx.cpu().numpy(), feat_s_t.shape)).T                          
                            
                            for idx in nn_idx:
                                n, c, h, w = idx[0], idx[1], idx[2], idx[3]
                                nn_masks[n, c, h, w] = 1  

                    # import code; code.interact(local=vars())
                    # loss_feat, _, _, _, _ = consistency_loss(feat_s, feat_s_t, 'ce', args.T, cutoff_value, args.soft_label)
                    # loss_feat = ce_loss(feat_s, feat_s_t, reduction='none') * nn_masks

                    loss_feat = F.mse_loss(feat_s, feat_s_t, reduction='none') * nn_masks
                    loss_feat = loss_feat.sum() / (nn_masks.sum() + 1e-6)

                else:
                    loss_feat = torch.tensor(0)

                loss = loss_cls + loss_sal + loss_ssl + loss_feat

                avg_meter.add({'loss': loss.item(),
                            'loss_ssl': loss_ssl.item(),
                            'loss_ssl_1': loss_ssl_1.item(),
                            'loss_ssl_2': loss_ssl_2.item(),
                            'loss_ssl_3': loss_ssl_3.item(),
                            'loss_ssl_4': loss_ssl_4.item(),
                            'loss_ssl_5': loss_ssl_5.item(),
                            'loss_ssl_6': loss_ssl_6.item(),
                            'mask_1' : masks[0].mean().item(),
                            'mask_2' : masks[1].mean().item(),
                            'mask_3' : masks[2].float().mean().item(),
                            'mask_4' : masks[3].float().mean().item(),
                            'mask_5' : masks[4].float().mean().item(),
                            'mask_6' : masks[5].float().mean().item(),
                            'loss_cls': loss_cls.item(),
                            'loss_sal': loss_sal.item(),
                            })
                if 1 in args.ssl_type:
                    avg_meter.add({'loss_mt'      : ssl_pack['loss_mt'].item(),
                                    'mt_mask_ratio': ssl_pack['mask_mt'].item()})
                if 2 in args.ssl_type:
                    avg_meter.add({'loss_pmt'     : ssl_pack['loss_pmt'].item()})
                if 3 in args.ssl_type:
                    avg_meter.add({
                        'loss_pl'      : ssl_pack['loss_pl'].item(),
                        'mask_ratio'   : ssl_pack['mask_pl'].item()
                        })
                if 4 in args.ssl_type:
                    avg_meter.add({'loss_con'     : ssl_pack['loss_con'].item()})
                if 5 in args.ssl_type:
                    avg_meter.add({
                        'loss_cdc'     : ssl_pack['loss_cdc'].item(),
                        'cdc_pos_ratio': ssl_pack['mask_cdc_pos'].item(),
                        'cdc_neg_ratio': ssl_pack['mask_cdc_neg'].item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update()

                # Logging
                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']
                    
                    # Print Logs
                    print(
                        'Iter:%5d/%5d' % (iteration, args.max_iters),
                        'Loss:%.4f' % (avg_meter.get('loss')),
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                        'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')))
                    print(
                        'Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print(
                        'mask1:%.4f' % (avg_meter.get('mask_1')),
                        'mask2:%.4f' % (avg_meter.get('mask_2')),
                        'mask3:%.4f' % (avg_meter.get('mask_3')),
                        'mask4:%.4f' % (avg_meter.get('mask_4')),
                        'mask5:%.4f' % (avg_meter.get('mask_5')),
                        'mask6:%.4f' % (avg_meter.get('mask_6')),
                        end=' ')
                    # SSL Losses
                    for k, v in ssl_pack.items():
                        print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')
                    print(
                        'imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                        'Fin:%s'      % (timer.str_est_finish()),
                        'lr: %.4f'    % (tscalar['train/lr']), flush=True)

                    ### wandb logging Scalars, Histograms, Images ###
                    if args.use_wandb:
                        wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                        wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                        # np_hist = np.histogram(_max_probs.detach().cpu().numpy(), range=[0.0, 1.0], bins=100)
                        # wandb.log({'train/mask_hist': wandb.Histogram(np_histogram=np_hist, num_bins=100)})
                        
                    tscalar.clear()

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 

                timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))


### contrast + semi-supervised learning ###


def train_contrast_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    # torch.autograd.set_detect_anomaly(True)
    log_keys = ['loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr', 'loss_ssl', 'loss_feat', \
                'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', \
                'loss_ssl_1', 'loss_ssl_2', 'loss_ssl_3', 'loss_ssl_4', 'loss_ssl_5', 'loss_ssl_6' ]
    if 1 in args.ssl_type:
        log_keys.append('loss_mt')  # loss of Mean Teacher
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
    strong_transforms = tensor_augment_list() 
    if args.val_only:
        print("Val-only mode.")

        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
            
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state

        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 
    
    else:
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
                        feat_s_t = apply_strong_tr(feat_w, ops, strong_transforms=strong_transforms)
                        # cam_rv_s_t = apply_strong_tr(cam_rv_w, ops, strong_transforms=strong_transforms)
                        # if 5 in args.ssl_type:
                    else: # weak aug
                        cam_s_t = cam_w
                        feat_s_t = feat_w
                    ### Cutmix 
                    if args.use_cutmix:
                        img_s, cam_s_t, feat_s_t = cutmix(img_s, cam_s_t, feat_s_t)
                        # img_s, cam_s_t, feat_s = cutmix(ulb_img2, cam_s_t, ulb_feat1_s)

                    ### Make strong augmented (transformed) prediction for MT ###
                    if 1 in args.ssl_type:
                        pred_s_t = F.avg_pool2d(cam_s_t, kernel_size=(cam_s_t.size(-2), cam_s_t.size(-1)), padding=0)
                        pred_s_t = pred_s_t.view(pred_s_t.size(0), -1)
                    else:
                        pred_s_t = pred_w
                    ### Make masks for pixel-wise MT ###
                    mask_s = torch.ones_like(cam_w)
                    if 2 in args.ssl_type or 4 in args.ssl_type:
                        mask_s = apply_strong_tr(mask_s, ops, strong_transforms=strong_transforms)

                ema.restore()
                ###

                ### Student
                if 5 not in args.ssl_type:
                    pred_s, cam_s, pred_rv_s, cam_rv_s, feat_s = model(img_s) ###
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
                # divide by confidence range
                # cam_s, cam_s_t

                ssl_pack, cutoff_value = get_ssl_loss(args, iteration, \
                    pred_s=pred_s, pred_t=pred_s_t, cam_s=cam_s, cam_t=cam_s_t, mask=mask_s)
                loss_ssl = ssl_pack['loss_ssl']
                masks, max_probs = get_masks_by_confidence(cam=cam_s_t)

                loss_ssl_1 = (ssl_pack['loss_ce'] * masks[0]).sum() / (masks[0].sum() + 1e-6)
                loss_ssl_2 = (ssl_pack['loss_ce'] * masks[1]).sum() / (masks[1].sum() + 1e-6)
                loss_ssl_3 = (ssl_pack['loss_ce'] * masks[2]).sum() / (masks[2].sum() + 1e-6)
                loss_ssl_4 = (ssl_pack['loss_ce'] * masks[3]).sum() / (masks[3].sum() + 1e-6)
                loss_ssl_5 = (ssl_pack['loss_ce'] * masks[4]).sum() / (masks[4].sum() + 1e-6)
                loss_ssl_6 = (ssl_pack['loss_ce'] * masks[5]).sum() / (masks[5].sum() + 1e-6)

                ssl_pack['loss_ce'] = ssl_pack['loss_ce'].mean()
                
                ############       Pseudo Label Propagation      ############
                # get k anchors: top-k highest confidence
                # pred_w:       (B, 21)
                # cam_s:        (B, 21, 56, 56)
                # max_probs:    (B, 56, 56) - 가장 높은 class confidence
                # feat_s_t:     (B, 128, 56, 56)

                if args.anchor_k or args.anchor_thr:
                    nn_masks = torch.zeros_like(feat_s_t, dtype=torch.long)
                    feat_s = torch.softmax(feat_s, dim=1)
                    feat_s_t = torch.softmax(feat_s_t, dim=1)

                    for max_prob in max_probs:
                        if args.anchor_k:
                            anc_values, _ = max_prob.flatten().topk(k=args.anchor_k)
                        elif args.anchor_thr:
                            anc_values = max_prob.flatten()
                            # anc_values = max_prob.flatten().sort(descending=True, stable=True).values
                            anc_values = anc_values[anc_values > args.anchor_thr]

                        # 전체 Batch, Channel의 feature space에서 anc와 가장 가까운 top L 개를 뽑음
                        for anc in anc_values:
                            dist = torch.abs(feat_s_t - anc).flatten()
                            # dist = torch.linalg.vector_norm(feat_s_t - anc, dim=0)    # (, 3136)
                            dis, idx = dist.topk(args.nn_l, largest=False)
                            nn_idx = np.array(np.unravel_index(idx.cpu().numpy(), feat_s_t.shape)).T                          
                            
                            for idx in nn_idx:
                                n, c, h, w = idx[0], idx[1], idx[2], idx[3]
                                nn_masks[n, c, h, w] = 1  

                    # import code; code.interact(local=vars())
                    # loss_feat, _, _, _, _ = consistency_loss(feat_s, feat_s_t, 'ce', args.T, cutoff_value, args.soft_label)
                    # loss_feat = ce_loss(feat_s, feat_s_t, reduction='none') * nn_masks

                    loss_feat = F.mse_loss(feat_s, feat_s_t, reduction='none') * nn_masks
                    loss_feat = loss_feat.sum() / (nn_masks.sum() + 1e-6)

                else:
                    loss_feat = torch.tensor(0)

                loss = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce + loss_ssl + loss_feat
        
                # Logging AVGMeter
                avg_meter.add({
                        'loss': loss.item(),
                        'loss_ssl': loss_ssl.item(),
                        'loss_ssl_1': loss_ssl_1.item(),
                        'loss_ssl_2': loss_ssl_2.item(),
                        'loss_ssl_3': loss_ssl_3.item(),
                        'loss_ssl_4': loss_ssl_4.item(),
                        'loss_ssl_5': loss_ssl_5.item(),
                        'loss_ssl_6': loss_ssl_6.item(),
                        'mask_1' : masks[0].float().mean().item(),
                        'mask_2' : masks[1].float().mean().item(),
                        'mask_3' : masks[2].float().mean().item(),
                        'mask_4' : masks[3].float().mean().item(),
                        'mask_5' : masks[4].float().mean().item(),
                        'mask_6' : masks[5].float().mean().item(),
                        'loss_cls': loss_cls.item(),
                        'loss_sal': loss_sal.item(),
                        'loss_nce': loss_nce.item(),
                        'loss_er' : loss_er.item(),
                        'loss_ecr': loss_ecr.item(),
                        'loss_feat': loss_feat.item()
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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update()
                
                # Logging
                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']
                    
                    # Print Logs
                    print(
                        'Iter:%5d/%5d' % (iteration, args.max_iters),
                        'Loss:%.4f' % (avg_meter.get('loss')),
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                        'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                        'Loss_Nce:%.4f' % (avg_meter.get('loss_nce')),
                        'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                        'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                        'Loss_Feat:%.4f' % (avg_meter.get('loss_feat'))
                    )
                    print(
                        'Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                    )
                    print(
                        'mask1:%.4f' % (avg_meter.get('mask_1')),
                        'mask2:%.4f' % (avg_meter.get('mask_2')),
                        'mask3:%.4f' % (avg_meter.get('mask_3')),
                        'mask4:%.4f' % (avg_meter.get('mask_4')),
                        'mask5:%.4f' % (avg_meter.get('mask_5')),
                        'mask6:%.4f' % (avg_meter.get('mask_6')),
                        end=' ')
                    # SSL Losses
                    for k, v in ssl_pack.items():
                        print(f'{k.replace(" ","_")}: {v.item():.4f}', end=' ')

                    print(
                        'imps:%.1f'   % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                        'Fin:%s'      % (timer.str_est_finish()),
                        'lr: %.4f'    % (tscalar['train/lr']), flush=True)

                    ### wandb logging Scalars, Histograms, Images ###
                    if args.use_wandb:
                        wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                        wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                        # np_hist = np.histogram(_max_probs.detach().cpu().numpy(), range=[0.0, 1.0], bins=100)
                        # wandb.log({'train/mask_hist': wandb.Histogram(np_histogram=np_hist, num_bins=100)})

                    tscalar.clear()

                if args.test: 
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')

                    # Validation
                    if val_dataloader is not None:
                        print('Validating Student Model... ')
                        validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 

                else:
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
                            validate_acc_by_class(args, model, val_dataloader, iteration, tag='val') 

                timer.reset_stage()
                
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint.pth'))