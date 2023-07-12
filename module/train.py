import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import functional as tvf
import wandb
from chainercv.evaluations import calc_semantic_segmentation_confusion

import os
import re 
import random
import numpy as np
from glob import glob
from copy import deepcopy
import natsort as nsort
import pdb
from util import pyutils
from data.augmentation.randaugment import tensor_augment_list
from module.loss import adaptive_min_pooling_loss, get_er_loss, get_eps_loss, get_contrast_loss, max_norm, balanced_cross_entropy
from module.validate import *
from module.ssl import *
from module.helper import get_avg_meter, get_masks_by_confidence, patchfy, merge_patches, resize_labels
import cv2

def train_ppc(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = get_avg_meter(args=args)
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

    if args.val_only:
        print("Val-only mode.")
        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        # weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state
        for weight in weight_path:
            print(f'Loading {weight}')
            model.module.load_state_dict(torch.load(weight), strict=True)
            model.eval()
            tmp = weight[-10:]
            # pdb.set_trace()
            try:
                iteration = int(re.sub(r'[^0-9]', '', tmp))
            except:
                iteration = 10000
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate(args, model, val_dataloader, iteration, tag='val')
    
    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                try:
                    img_id, img, sal, label = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_dataloader)
                    img_id, img, sal, label = next(loader_iter)
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                sal = sal.cuda(non_blocking=True)
                                    
                B = img.shape[0]

                img_w2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
                sal2 = F.interpolate(sal, size=(128, 128), mode='bilinear', align_corners=True)
                pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
                pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img_w2)                        
                loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                bg_score = torch.ones((img.shape[0], 1)).cuda()
                label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                
                # SEAM Losses
                loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                
                # EPS+PPC Losses
                loss_sal, _, _, _ = get_eps_loss(cam1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal2, _, _, _ = get_eps_loss(cam2, sal2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, sal2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.
                loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)
                loss_sup = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce
                avg_meter.add({
                    'loss_cls': loss_cls.item(),
                    'loss_er': loss_er.item(),
                    'loss_ecr': loss_ecr.item(),
                    'loss_nce': loss_nce.item(),
                    'loss_sal': loss_sal.item(),
                    'loss_sup': loss_sup.item()
                    })

                optimizer.zero_grad()
                loss_sup.backward()
                optimizer.step()
                
                # Logging 
                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                    print('Iter:%5d/%5d' % (iteration, args.max_iters),
                          'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                          'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                          'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                          'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                          'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                          'Loss_SUP:%.4f' % (avg_meter.get('loss_sup')),
                          'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                          'ETA: %s' % (timer.get_est_remain()), 
                          flush=True)

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



def train_ppc_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):        
    
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None
    strong_transforms = tensor_augment_list(args.use_geom_augs)

    print('Using Geometric Transforms') if args.use_geom_augs else print("NOT using Geometric Transforms")
    print(f'Using {args.n_strong_augs} RandAugs.')
    print(f'Using {args.patch_k}x{args.patch_k} patches.') if args.patch_k else print('NOT using patches.')

    # EMA
    if args.use_ema:
        print('EMA enabled')
        ema = EMA(model, args.ema_m)
        ema.register()
    else:
        print('EMA disabled')
        ema = None
    
    print("CutMix enabled") if args.use_cutmix else print("CutMix disabled")

    if args.recon_lambda != 0.0:
        print(f"Recon loss ratio: {args.recon_lambda}")
    if args.bdry_lambda != 0.0:
        print(f"Boundary loss ratio: {args.bdry_lambda}")
    if args.screg_lambda != 0.0:
        print(f"Scale Regularization loss ratio: {args.screg_lambda}")

    ### Train Scalars, Histograms, Images ###
    avg_meter = get_avg_meter(args=args)
    timer = pyutils.Timer("Session started: ")

    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times ### validation logging
    gamma = 0.10

    if args.val_only:
        print("Val-only mode.")
        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        for weight in weight_path:
            print(f'Loading {weight}')
            model.module.load_state_dict(torch.load(weight), strict=True)
            model.eval()
            tmp = weight[-10:]
            try:
                iteration = int(re.sub(r'[^0-9]', '', tmp))
            except:
                iteration = 10000
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate(args, model, val_dataloader, iteration, tag='val')
    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                ######## Dataloads ########
                try:
                    img_id, img_w, sal, img_s, _, tr_ops, randaug_ops, label = next(lb_loader_iter)
                except:
                    lb_loader_iter = iter(train_dataloader)
                    img_id, img_w, sal, img_s, _, tr_ops, randaug_ops, label = next(lb_loader_iter)
                B = len(img_id)
                if train_ulb_dataloader:
                    try:
                        ulb_img_id, ulb_img_w, ulb_img_s, ulb_tr_ops, ulb_randaug_ops = next(ulb_loader_iter)
                    except:
                        ulb_loader_iter = iter(train_ulb_dataloader)       
                        ulb_img_id, ulb_img_w, ulb_img_s, ulb_tr_ops, ulb_randaug_ops = next(ulb_loader_iter)
                        
                    # Concat Image lb & ulb
                    img_id = img_id + ulb_img_id
                    img_w = torch.cat([img_w, ulb_img_w], dim=0)
                    img_s = torch.cat([img_s, ulb_img_s], dim=0)
                    
                    # Concat Strong Aug. options
                    for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(randaug_ops, ulb_randaug_ops)):
                        randaug_ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                        randaug_ops[i][1] = torch.cat([v, ulb_v], dim=0)

                    for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(tr_ops, ulb_tr_ops)):
                        tr_ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                        tr_ops[i][1] = torch.cat([v, ulb_v], dim=0)

                img_w = img_w.cuda(non_blocking=True)
                img_s = img_s.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                sal = sal.cuda(non_blocking=True)

                ######## Supervised Losses ########                

                img_w2 = F.interpolate(img_w, size=(128, 128), mode='bilinear', align_corners=True)
                pred1, cam1, _, cam_rv1, feat1 = model(img_w)     # 112, 56, 56, 56
                pred2, cam2, _, cam_rv2, feat2 = model(img_w2)    # 32, 16, 16, 16
                loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                bg_score = torch.ones((img_w.shape[0], 1)).cuda()
                label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                
                # SEAM Losses
                loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                
                # EPS+PPC Losses
                # resblock7(conv6)
                loss_sal, _, _, _ = get_eps_loss(cam1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal2, _, _, _ = get_eps_loss(cam2, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)[0]
                loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2. 

                loss_sup = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce
                avg_meter.add({
                    'loss_cls': loss_cls.item(),
                    'loss_er': loss_er.item(),
                    'loss_ecr': loss_ecr.item(),
                    'loss_nce': loss_nce.item(),
                    'loss_sal': loss_sal.item(),
                    'loss_sup': loss_sup.item()
                    })
                '''
                #######     Pseudo Label Propagation    #######
                pred_w:       (B, 21)
                cam_s:        (B, 21, 56, 56)
                max_probs:    (B, 56, 56) - 가장 높은 class confidence
                feat_tr_t:     (B, 128, 56, 56)
                '''
                ################################ Teacher ################################
                if ema is not None:
                    ema.apply_shadow()
                
                with torch.no_grad():
                    class_mask = label.unsqueeze(2).unsqueeze(3)
                    cam_w = model.module.forward_cam(img_w)
                    cam_w[:B, :-1, :, :] *= class_mask  # ignore activation of non-existing classes
                    conf_value, pl_w = cam_w.softmax(dim=1).max(dim=1)

                    if args.use_cutmix:
                        ### TODO: low_scale일때 원래와 동일하도록 구현
                        # geom 없을 때만 사용 가능
                        img_s, cam_w = cutmix(img_s, cam_w)

                    pl_mask = conf_value.ge(args.p_cutoff).float()
                    avg_meter.add({'mask_ratio': pl_mask.mean().item()})

                    if args.bdry:
                        kernel = np.ones((args.bdry_size, args.bdry_size), np.int8)
                        pl_mask_bdry = np.zeros_like(pl_mask.cpu().numpy())

                        # if args.bdry_method == 'fg':
                        pl_mask_fg = cam_w.softmax(dim=1)[:, :-1].max(dim=1).values.ge(args.p_cutoff).float()
                        for i in range(len(pl_mask_fg)):
                            pl_mask_bdry[i] = cv2.dilate(pl_mask_fg[i].cpu().numpy(), kernel, iterations=1) - pl_mask_fg[i].cpu().numpy()

                        pl_mask_bdry = torch.tensor(pl_mask_bdry).cuda()
                        avg_meter.add({'bdry_ratio': pl_mask_bdry.mean().item()})

                if ema is not None:
                    ema.restore()
                
                ################################ Student ################################

                semcon_criterion = nn.CrossEntropyLoss(reduction='none')
                screg_criterion = nn.L1Loss(reduction='mean')
                recon_criterion = nn.CrossEntropyLoss(reduction='mean')

                cam_s = model.module.forward_cam(img_s)
                cam_s[:B, :-1, :, :] *= class_mask

                # Hard Label CE Loss on PL region
                loss_semcon1 = (semcon_criterion(cam_s, pl_w) * pl_mask).mean()
                loss_semcon2 = torch.tensor(0.0)
                loss_recon = torch.tensor(0.0)
                loss_screg = torch.tensor(0.0)

                loss = loss_sup + loss_semcon1 

                if args.screg_lambda != 0.0:
                    # Scale Regularization Loss
                    size_list = [128, 224, 336]
                    res = size_list[random.randrange(len(size_list))]
                    img_s2 = F.interpolate(img_s, size=(res, res), mode='bilinear', align_corners=True)
                    cam_s2 = model.module.forward_cam(img_s2)
                    cam_s2[:B, :-1, :, :] *= class_mask
                    cam_s_res = F.interpolate(cam_s, size=cam_s2.size(-1), mode='bilinear', align_corners=True)
                    loss_screg = screg_criterion(max_norm(cam_s2), max_norm(cam_s_res))
                    loss += args.screg_lambda*loss_screg

                if args.bdry:
                    # Hard Label CE Loss on Boundary region
                    loss_semcon2 = (semcon_criterion(cam_s, pl_w) * pl_mask_bdry).mean()
                    loss += args.bdry_lambda*loss_semcon2

                if args.recon_lambda != 0.0:
                    img_s_p = patchfy(img_s, args.patch_k)
                    cam_s_p = model.module.forward_cam(img_s_p)
                    cam_s_p = F.relu(cam_s_p, inplace=True)
                    cam_s_p = merge_patches(cam_s_p, args.patch_k, img_s.shape[0])
                    cam_s_p[:B, :-1, :, :] *= class_mask
                    pred_s_p = cam_s_p.detach().max(dim=1)[1]
                    # Recon Loss for local details
                    loss_recon = recon_criterion(cam_s, pred_s_p)
                    loss += args.recon_lambda*loss_recon
                    
                masks = get_masks_by_confidence(cam=cam_w)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ema is not None:
                    ema.update()
                
                avg_meter.add({
                    'loss': loss.item(),
                    'loss_semcon': loss_semcon1.item(),
                    'loss_semcon2': loss_semcon2.item(),
                    'loss_recon': loss_recon.item(),
                    'loss_screg': loss_screg.item(),
                    'mask_1' : masks[0].mean().item(),
                    'mask_2' : masks[1].mean().item(),
                    'mask_3' : masks[2].mean().item(),
                    'mask_4' : masks[3].mean().item(),
                    'mask_5' : masks[4].mean().item(),
                    'mask_6' : masks[5].mean().item(),
                })          

                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']
                    print('Iter:%5d/%5d' % (iteration, args.max_iters), 
                          'Loss:%.4f' % (avg_meter.get('loss')),
                          'Loss_CLS:%.4f' % (avg_meter.get('loss_cls')),
                          'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                          'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                          'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                          'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                          'Loss_SUP: %.4f' % (avg_meter.get('loss_sup')),
                          'Loss_SemCon1: %.4f' % (avg_meter.get('loss_semcon')),
                          'Loss_SemCon2: %.4f' % (avg_meter.get('loss_semcon2')),
                        #   'Loss_Recon: %.4f' % (avg_meter.get('loss_recon')),
                        #   'Loss_ScReg: %.4f' % (avg_meter.get('loss_screg')),
                          'conf_1:%.4f' % (avg_meter.get('mask_1')),
                          'conf_2:%.4f' % (avg_meter.get('mask_2')),
                          'conf_3:%.4f' % (avg_meter.get('mask_3')),
                          'conf_4:%.4f' % (avg_meter.get('mask_4')),
                          'conf_5:%.4f' % (avg_meter.get('mask_5')),
                          'conf_6:%.4f' % (avg_meter.get('mask_6')),
                          'mask_ratio:%.4f' % (avg_meter.get('mask_ratio')),
                          'bdry_ratio:%.4f' % (avg_meter.get('bdry_ratio')),
                          'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                          'ETA: %s' % (timer.get_est_remain()), 
                          flush=True)

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