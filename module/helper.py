import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from chainercv.evaluations import calc_semantic_segmentation_confusion
import math
from util import pyutils

def calc_score(pred, gt, mask=None, num_cls=21):
    if mask is None:
        mask = torch.ones_like(gt)
        total = mask.nelement()
    else:
        total = mask.sum()
        pred = torch.where(mask, pred, torch.ones_like(mask) * 255.)
        gt = torch.where(mask, gt.float(), torch.ones_like(mask) * 255.)
        
    class_mask = gt.lt(num_cls)
    correct = (pred == gt) * class_mask
    class_bg_mask = gt.not_equal(0) * class_mask
    correct_nobg = correct * class_bg_mask
    # Accuracy
    Acc      = (correct.sum() / class_mask.sum()).item()
    Acc_nobg = (correct_nobg.sum() / class_bg_mask.sum()).item()
    P, T, TP = [], [] ,[]
    IoU, Precision, Recall = [], [], []
    for c in range(num_cls):
        P.append(((pred == c) * class_mask).sum().item())
        T.append(((gt == c) * class_mask).sum().item())
        TP.append(((gt == c) * correct).sum().item())
        
        IoU.append(TP[c] / (T[c] + P[c] - TP[c] + 1e-10))
        Precision.append(TP[c] / (P[c] + 1e-10))
        Recall.append(TP[c] / (T[c] + 1e-10))
    
    mmIoU = sum(IoU)/len(np.unique(gt))
    mmIoU_nobg = sum(IoU[1:])/(len(np.unique(gt)-1))

    return total, Acc, Acc_nobg, P, T, TP, IoU, Precision, Recall, mmIoU, mmIoU_nobg

def align_with_strongcrop(args, img, target_img, tr_ops, is_cam=False):
    target_size = target_img.shape[-1]
    if is_cam==True:
        # upsample to weak crop size
        img = F.interpolate(img, size=(args.crop_size, args.crop_size), mode='bilinear', align_corners=False)
    
    box = torch.stack(tr_ops[-1]).permute(1, 0)
    img_cont = torch.ones(size=(args.batch_size, img.size(1), args.strong_crop_size, args.strong_crop_size), dtype=img.dtype)
    for i in range(len(img_cont)):
        pos = box[i]
        img_cont[i, :, pos[0]:pos[1], pos[2]:pos[3]] = img[i, :, pos[4]:pos[5], pos[6]:pos[7]]
    
    if is_cam==True:
        # shrink to original size
        return F.interpolate(img_cont, size=(target_size, target_size), mode='bilinear', align_corners=False)
    else:
        return img_cont

def get_masks_by_confidence(cam):
    '''
    input: normalized class prob map [B, C, H, W]
    output: list of masks [B, H, W] which belongs to confidence range of 
    [, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 0.95), [0.95, 0.99), [0.99, ]
    '''
    masks = []
    cam = torch.softmax(cam, dim=1)
    _max_probs, _max_idx = torch.max(cam, dim=1)
    masks.append(_max_probs.lt(0.4).float())
    masks.append(torch.logical_and(_max_probs.ge(0.4), _max_probs.lt(0.6)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.6), _max_probs.lt(0.8)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.8), _max_probs.lt(0.95)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.95), _max_probs.lt(0.99)).float())
    masks.append(_max_probs.ge(0.99).float())
    return masks
    
def get_anc_mask(cam_tea, anchor_k):
    pseudo_label = torch.softmax(cam_tea, dim=1)
    max_probs, _ = torch.max(pseudo_label, dim=1)
    anc_mask = torch.zeros_like(max_probs, dtype=torch.long)
    for i in range(len(max_probs)):
        max_prob = max_probs[i]
        anc_values, anc_idx = max_prob.flatten().topk(anchor_k)
        anc_idx = np.array(np.unravel_index(anc_idx.cpu().numpy(), max_prob.shape)).T
        for idx in anc_idx:
            h, w = idx[0], idx[1]
            anc_mask[i, h, w] = 1
    return anc_mask
                    
def calc_metrics_val(cams, labels): 
    confusion = calc_semantic_segmentation_confusion(cams, labels)
    gtj = confusion.sum(axis=1)     # P
    resj = confusion.sum(axis=0)    # T
    gtjresj = np.diag(confusion)    # TP    
    denominator = gtj + resj - gtjresj

    precision = gtjresj / (gtj + 1e-10) # TP / TP+FP
    recall = gtjresj / (resj + 1e-10)   # TP / TP+FN
    iou = gtjresj / denominator
    acc_total = gtjresj.sum() / confusion.sum()

    return precision, recall, iou, acc_total, confusion

def calc_acc_byclass(cams, labels): 
    confusion = calc_semantic_segmentation_confusion(cams, labels)
    gtjresj = np.diag(confusion)
    acc_by_class = gtjresj / (confusion.sum(axis=1) + 1e-10)
    acc_total = gtjresj.sum() / confusion.sum()
    return acc_by_class, acc_total, confusion

def get_avg_meter(args):
    log_keys = ['loss_cls', 'loss_sup']
    if args.network_type == 'seam':
        log_keys.extend(['loss_er', 'loss_ecr'])
    elif args.network_type == 'eps':
        log_keys.extend(['loss_sal'])
    elif args.network_type == 'contrast':
        log_keys.extend(['loss_er', 'loss_ecr','loss_sal', 'loss_nce'])
    if args.mode in ['v2', 'ssl']:
        log_keys.extend([
            'loss', 'loss_semcon','loss_viewcon', 'mask_ratio', \
            'loss_semcon_1', 'loss_semcon_2', 'loss_semcon_3', 'loss_semcon_4', 'loss_semcon_5', 'loss_semcon_6',\
            'loss_viewcon_1', 'loss_viewcon_2', 'loss_viewcon_3', 'loss_viewcon_4', 'loss_viewcon_5', 'loss_viewcon_6', \
            'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6'])
    avg_meter = pyutils.AverageMeter(*log_keys)
    return avg_meter
    
def ssl_dataiter(train_dataloader, train_ulb_dataloader, args):
    assert args.network_type in ['cls', 'eps', 'seam', 'contrast']
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) if train_ulb_dataloader else None
    if args.network_type in ['eps', 'contrast']:
        try:
            img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
        except StopIteration:
            lb_loader_iter = iter(train_dataloader)
            img_id, img_w, saliency, img_s, _, ops, label = next(lb_loader_iter)
            
            if train_ulb_dataloader:
                try:
                    ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
                except StopIteration:
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
        return img_id, img_w, img_s, ops, saliency, label

    elif args.network_type in ['cls', 'seam']:
        try:
            img_id, img_w, img_s, ops, label = next(lb_loader_iter)
        except StopIteration:
            lb_loader_iter = iter(train_dataloader)
            img_id, img_w, img_s, ops, label = next(lb_loader_iter)
        
        if ulb_loader_iter:
            try:
                ulb_img_id, ulb_img_w, ulb_img_s, ulb_ops = next(ulb_loader_iter)
            except StopIteration:
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
        return img_id, img_w, img_s, ops, label