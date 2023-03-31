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
from module.loss import adaptive_min_pooling_loss, get_er_loss, get_eps_loss, get_contrast_loss
from module.validate import *
from module.ssl import *
from module.helper import get_avg_meter, Attn, Self_Attn, ssl_dataiter

# Control Randomness
random_seed = 42 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def train(train_dataloader, val_dataloader, model, optimizer, max_step, args):
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
        weight_path = nsort.natsorted(weight_path)[1:]  # excludes last state
        for weight in weight_path:
            print(f'Loading {weight}')
            model.load_state_dict(torch.load(weight), strict=False)
            model.eval()
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate(args, model, val_dataloader, iteration, tag='val')
    
    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                # Forward
                if args.network_type in ['cls', 'seam']:
                    try:
                        img_id, img, label = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(train_dataloader)
                        img_id, img, label = next(loader_iter)
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    if args.network_type == 'seam':
                        pred1, cam1, pred_rv1, cam_rv1 = model(img)
                    else:
                        pred1 = model(img)

                elif args.network_type in ['eps', 'contrast']:
                    try:
                        img_id, img, saliency, label = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(train_dataloader)
                        img_id, img, saliency, label = next(loader_iter)
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    saliency = saliency.cuda(non_blocking=True)
                    if args.network_type == 'contrast':
                        pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
                    else:
                        pred1, cam1 = model(img)
                                    
                loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                if args.network_type == 'cls':
                    loss = loss_cls
                    avg_meter.add({'loss_cls': loss_cls.item(),
                                'loss': loss.item()})

                if args.network_type == 'eps':
                    loss_sal, _, _, _ = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss = loss_cls + loss_sal
                    avg_meter.add({'loss_cls': loss_cls.item(),
                                'loss_sal': loss_sal.item(), 
                                'loss': loss.item()})
                    
                elif args.network_type in ['seam', 'contrast']:
                    # Additional forward for SEAM and PPC
                    img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
                    if args.network_type == 'contrast':
                        pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)
                    else:
                        pred2, cam2, pred_rv2, cam_rv2 = model(img2)
                    
                    # Classification Losses for SEAM and PPC
                    bg_score = torch.ones((img.shape[0], 1)).cuda()
                    label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                    loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                    loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                    loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                    # SEAM Losses
                    loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                    loss = loss_cls + loss_er + loss_ecr
                    avg_meter.add({'loss_cls': loss_cls.item(),
                                'loss_er': loss_er.item(),
                                'loss_ecr': loss_ecr.item()})
                    # EPS+PPC Losses
                    if args.network_type == 'contrast':
                        saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)
                        loss_sal, _, _, _ = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                        loss_sal2, _, _, _ = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                        loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                        loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                        loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.
                        loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)
                        loss = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce
                        avg_meter.add({'loss_nce': loss_nce.item(),
                                    'loss_sal': loss_sal.item()})
                    avg_meter.add({'loss': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Logging 
                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                    print('Iter:%5d/%5d' % (iteration, args.max_iters), 
                        'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')), end=' ')
                    if args.network_type == 'seam':
                        print('Loss_ER:%.4f' % (avg_meter.get('loss_er')), 
                            'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')), end=' ')
                    elif args.network_type == 'eps':
                        print('Loss_Sal:%.4f' % (avg_meter.get('loss_sal')), end=' ')
                    elif args.network_type == 'contrast':
                        print('Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                            'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                            'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                            'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                            end=' ')
                    print('Loss:%.4f' % (avg_meter.get('loss')),'ETA: %s' % (timer.get_est_remain()), flush=True)

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

def train_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = get_avg_meter(args=args)
    timer = pyutils.Timer("Session started: ")
    strong_transforms = tensor_augment_list()

    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = max_step // args.val_times ### validation logging
    gamma = 0.10
    print(args)
    
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None

    if args.use_attn:
        with torch.no_grad(): 
            feat_attn = Attn(4096, 21).cuda()

    # EMA
    if args.use_ema:
        print('EMA enabled at training')
        ema = EMA(model, args.ema_m)
        ema.register()
    else:
        print('EMA disabled')
        ema = None

    print("Using CutMix") if args.use_cutmix else print("CutMix disabled")

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
            model.eval()
            tmp = weight[-10:]
            iteration = int(re.sub(r'[^0-9]', '', tmp) )
            # Validation
            if val_dataloader is not None:
                print('Validating Student Model... ')
                validate(args, model, val_dataloader, iteration, tag='val')
    
    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                ######## Dataloads ########
                if args.network_type in ['eps', 'contrast']:
                    try:
                        img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
                    except:
                        lb_loader_iter = iter(train_dataloader)
                        img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
                    if train_ulb_dataloader:
                        try:
                            ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                        except:
                            ulb_loader_iter = iter(train_ulb_dataloader)       
                            ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                        
                        # Concat Image lb & ulb
                        img_id = img_id + ulb_img_id
                        img_w = torch.cat([img_w, ulb_img_w], dim=0)
                        img_s = torch.cat([img_s, ulb_img_s], dim=0)
                        # Concat Strong Aug. options
                        for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                            ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                            ops[i][1] = torch.cat([v, ulb_v], dim=0)

                    img_w = img_w.cuda(non_blocking=True)
                    img_s = img_s.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    saliency = saliency.cuda(non_blocking=True)
                
                elif args.network_type in ['cls', 'seam']:
                    try:
                        img_id, img_w, img_s, ops, label = next(lb_loader_iter)
                    except:
                        lb_loader_iter = iter(train_dataloader)
                        img_id, img_w, img_s, ops, label = next(lb_loader_iter)
                    
                    if ulb_loader_iter:
                        try:
                            ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                        except:
                            ulb_loader_iter = iter(train_ulb_dataloader)  
                            ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                        
                        # Concat Image lb & ulb
                        img_id = img_id + ulb_img_id
                        img_w = torch.cat([img_w, ulb_img_w], dim=0)
                        img_s = torch.cat([img_s, ulb_img_s], dim=0)
                        # Concat Strong Aug. options
                        for i, ((idx, v), (ulb_idx, ulb_v)) in enumerate(zip(ops, ulb_ops)):
                            ops[i][0] = torch.cat([idx, ulb_idx], dim=0)
                            ops[i][1] = torch.cat([v, ulb_v], dim=0)

                    img_w = img_w.cuda(non_blocking=True)
                    img_s = img_s.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)

                ######## Supervised Losses ########
                B = img_w.shape[0]
                if args.network_type == 'cls':
                    pred1, cam1 = model(img_w, forward_cam=True)
                    loss_sup = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    avg_meter.add({'loss_cls': loss_cls.item()})
                
                elif args.network_type == 'eps':
                    pred1, cam1 = model(img_w)
                    loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    loss_sal, _, _, _ = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss_sup = loss_cls + loss_sal
                    avg_meter.add({'loss_cls': loss_cls.item(), 'loss_sal': loss_sal.item()})
                
                elif args.network_type == 'seam':
                    img2 = F.interpolate(img_w, size=(128, 128), mode='bilinear', align_corners=True)
                    pred1, cam1, pred_rv1, cam_rv1 = model(img_w)
                    pred2, cam2, pred_rv2, cam_rv2 = model(img2)
                    loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                    bg_score = torch.ones((B, 1)).cuda()
                    label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                    loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                    loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                    loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                    loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                    loss_sup = loss_cls + loss_er + loss_ecr
                    avg_meter.add({'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item()})

                elif args.network_type == 'contrast':
                    img2 = F.interpolate(img_w, size=(128, 128), mode='bilinear', align_corners=True)
                    saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)
                    pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img_w)
                    pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)                   
                    loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                    bg_score = torch.ones((B, 1)).cuda()
                    label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                    loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                    loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                    loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                    loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                    loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)[0]
                    loss_sal, _, _, _ = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss_sal2, _, _, _ = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_sample)
                    loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.
                    loss_sup = loss_cls + loss_er + loss_ecr + loss_nce + loss_sal

                    avg_meter.add({
                        'loss_cls': loss_cls.item(),
                        'loss_er': loss_er.item(),
                        'loss_ecr': loss_ecr.item(),
                        'loss_nce': loss_nce.item(),
                        'loss_sal': loss_sal.item()
                        })
                    
                '''
                #######     Pseudo Label Propagation    #######
                pred_w:       (B, 21)
                cam_s:        (B, 21, 56, 56)
                max_probs:    (B, 56, 56) - 가장 높은 class confidence
                feat_tr_t:     (B, 128, 56, 56)
                '''
                ######## Teacher ########
                if ema is not None:
                    ema.apply_shadow()
                with torch.no_grad():
                    if args.network_type == 'contrast':
                        pred_w, cam_w, pred_rv_w, cam_rv_w, feat_w = model(img_w)
                        cam_w[:, :-1] *= label[:,:,None,None]
                        # Geometric Matching 
                        if args.ulb_aug_type == 'strong':
                            cam_tr = apply_strong_tr(cam_w, ops, strong_transforms=strong_transforms)
                            feat_tr = apply_strong_tr(feat_w, ops, strong_transforms=strong_transforms)
                        else: # weak aug
                            cam_tr = cam_w
                            feat_tr = feat_w
                        if args.use_cutmix:
                            img_s, cam_tr, feat_tr = cutmix(img_s, cam_tr, feat_tr)

                    elif args.network_type in ['cls', 'eps', 'seam']:
                        if args.network_type == 'seam':
                            pred_w, cam_w, pred_rv_w, cam_rv_w = model(img_w)
                        else:
                            pred_w, cam_w = model(img_w)
                        cam_w[:, :-1] *= label[:,:,None,None]
                        # Geometric Matching 
                        if args.ulb_aug_type == 'strong':
                            cam_tr = apply_strong_tr(cam_w, ops, strong_transforms=strong_transforms) if args.ulb_aug_type == 'strong' else cam1
                        if args.use_cutmix:
                            img_s, cam_tr = cutmix(img_s, cam_tr)
                if ema is not None:
                    ema.restore()

                mask1 = torch.max(torch.softmax(cam_tr, dim=1), dim=1).values.ge(args.p_cutoff).float()
                
                if args.use_attn:
                    mask_focal = focal_mask(feat_tr, args.focal_p)
                    attn, _ = feat_attn(feat_tr, feat_tr, cam_tr, mask_focal, args)
                    mask2 = torch.max(attn, dim=1).values.ge(args.attn_cutoff).float()
                    conf_mask = torch.maximum(mask1, mask2) # Union of 2 masks
                    avg_meter.add({
                        'attn_max'   : attn.max().item(),
                        'mask_ratio' : mask1.mean().item(),
                        'mask2_ratio': mask2.mean().item(),       # attn mask ratio
                        'maskU_ratio': conf_mask.mean().item(),
                        })
                else:
                    conf_mask = mask1
                    avg_meter.add({'mask_ratio': conf_mask.mean().item()})
                
                ######## Student ########
                if args.network_type == 'cls':
                    pred_s, cam_s = model(img_s, forward_cam=True)
                elif args.network_type == 'eps':
                    pred_s, cam_s = model(img_s)
                elif args.network_type == 'seam':
                    pred_s, cam_s, pred_rv_s, cam_rv_s = model(img_s)
                elif args.network_type == 'contrast':
                    pred_s, cam_s, pred_rv_s, cam_rv_s, feat_s = model(img_s)

                # divide by confidence range
                ssl_pack = get_ssl_loss(args, iteration, cam_s=cam_s, cam_t=cam_tr, mask=conf_mask)
                loss_ssl = ssl_pack['loss_ssl']
                masks = get_masks_by_confidence(cam=cam_tr)
                loss_ssl_1 = (ssl_pack['loss_org'] * masks[0]).sum() / (masks[0].sum() + 1e-6)
                loss_ssl_2 = (ssl_pack['loss_org'] * masks[1]).sum() / (masks[1].sum() + 1e-6)
                loss_ssl_3 = (ssl_pack['loss_org'] * masks[2]).sum() / (masks[2].sum() + 1e-6)
                loss_ssl_4 = (ssl_pack['loss_org'] * masks[3]).sum() / (masks[3].sum() + 1e-6)
                loss_ssl_5 = (ssl_pack['loss_org'] * masks[4]).sum() / (masks[4].sum() + 1e-6)
                loss_ssl_6 = (ssl_pack['loss_org'] * masks[5]).sum() / (masks[5].sum() + 1e-6)
                ssl_pack['loss_org'] = ssl_pack['loss_org'].mean()
                
                loss = loss_sup + loss_ssl
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update()

                # Logging 
                avg_meter.add({
                    'loss': loss.item(),
                    'loss_ssl': loss_ssl.item(),
                    'loss_ssl_1': loss_ssl_1.item(),
                    'loss_ssl_2': loss_ssl_2.item(),
                    'loss_ssl_3': loss_ssl_3.item(),
                    'loss_ssl_4': loss_ssl_4.item(),
                    'loss_ssl_5': loss_ssl_5.item(),
                    'loss_ssl_6': loss_ssl_6.item(),
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
                        'Loss_CLS:%.4f' % (avg_meter.get('loss_cls')), end=' ')
                    if args.network_type == 'seam':
                        print('Loss_ER:%.4f' % (avg_meter.get('loss_er')), 
                            'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')), end=' ')
                    elif args.network_type == 'eps':
                        print('Loss_Sal:%.4f' % (avg_meter.get('loss_sal')), end=' ')
                    elif args.network_type == 'contrast':
                        print('Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                            'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                            'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                            'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                            end=' ')
                    print(
                        'Loss_SSL: %.4f' % (avg_meter.get('loss_ssl')),
                        'Loss_SSL1:%.4f' % (avg_meter.get('loss_ssl_1')),
                        'Loss_SSL2:%.4f' % (avg_meter.get('loss_ssl_2')),
                        'Loss_SSL3:%.4f' % (avg_meter.get('loss_ssl_3')),
                        'Loss_SSL4:%.4f' % (avg_meter.get('loss_ssl_4')),
                        'Loss_SSL5:%.4f' % (avg_meter.get('loss_ssl_5')),
                        'Loss_SSL6:%.4f' % (avg_meter.get('loss_ssl_6')),
                        'conf_1:%.4f' % (avg_meter.get('mask_1')),
                        'conf_2:%.4f' % (avg_meter.get('mask_2')),
                        'conf_3:%.4f' % (avg_meter.get('mask_3')),
                        'conf_4:%.4f' % (avg_meter.get('mask_4')),
                        'conf_5:%.4f' % (avg_meter.get('mask_5')),
                        'conf_6:%.4f' % (avg_meter.get('mask_6')),
                        'mask1:%.4f' % (avg_meter.get('mask_ratio')), 
                        end=' ')
                    if args.use_attn:
                        print(
                            'mask2:%.4f' % (avg_meter.get('mask2_ratio')),
                            'maskU:%.4f' % (avg_meter.get('maskU_ratio')),
                            'attn_max:%.4f' % (avg_meter.get('attn_max')), 
                            end=' ')    
                    print('ETA: %s' % (timer.get_est_remain()), flush=True)

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