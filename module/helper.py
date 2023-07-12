import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from chainercv.evaluations import calc_semantic_segmentation_confusion
import math
from util import pyutils
from PIL import Image

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

def get_masks_by_confidence(cam):
    '''
    input: CAM before softmax [B, C, H, W]
    output: list of masks [B, H, W] which belongs to confidence range of 
    [, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 0.95), [0.95, 0.99), [0.99, ]
    '''
    masks = []
    # cam = torch.softmax(cam, dim=1)
    _max_probs = cam.softmax(dim=1).max(dim=1).values
    masks.append(_max_probs.lt(0.4).float())
    masks.append(torch.logical_and(_max_probs.ge(0.4), _max_probs.lt(0.6)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.6), _max_probs.lt(0.8)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.8), _max_probs.lt(0.95)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.95), _max_probs.lt(0.99)).float())
    masks.append(_max_probs.ge(0.99).float())
    return masks
                    
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
        log_keys.extend(['loss_er', 'loss_ecr', 'loss_er_s', 'loss_ecr_s', \
                         'loss_er_ws', 'loss_ecr_ws'])
    elif args.network_type == 'eps':
        log_keys.extend(['loss_sal'])
    elif args.network_type == 'contrast':
        log_keys.extend(['loss_er', 'loss_ecr', 'loss_nce', 'loss_sal', \
                         ])
    if args.mode in ['v2', 'ssl']:
        log_keys.extend([
            'loss', 'loss_semcon', 'loss_semcon2', 'loss_recon', 'loss_screg', \
            'mask_1', 'mask_2', 'mask_3', 'mask_4', 'mask_5', 'mask_6', \
            'mask_ratio', 'bdry_ratio'
            ])
    avg_meter = pyutils.AverageMeter(*log_keys)
    return avg_meter
    
def patchfy(img, patch_k):
    h, w = img.size()[-2:]
    h_patch_size = h // patch_k
    w_patch_size = w // patch_k
    patches = []
    for p in torch.split(img, h_patch_size, dim=2):
        for patch in torch.split(p, w_patch_size, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)

def merge_patches(patch, patch_k, batch_size):
    patch_list = list(torch.split(patch, batch_size, dim=0))
    idx = 0
    ext_h_list = []

    for _ in range(patch_k):
        ext_w_list = []
        for _ in range(patch_k):
            ext_w_list.append(patch_list[idx])
            idx += 1
        ext_h_list.append(torch.cat(ext_w_list, dim=3))
    
    return torch.cat(ext_h_list, dim=2)

def patch_with_tr(img, patch_k, tr):
    patches = []
    randaug_ops = []
    for p in np.split(img, patch_k, axis=0):
        for patch in np.split(p, patch_k, axis=1):
            patch = Image.fromarray(patch)
            patch, aug_ops = tr(patch)
            patches.append(np.asarray(patch))
            randaug_ops.append(aug_ops)
    return patches, randaug_ops

def merge_patches_np(patch_list, patch_k):
    idx = 0
    ext_h_list = []

    for _ in range(patch_k):
        ext_w_list = []
        for _ in range(patch_k):
            ext_w_list.append(patch_list[idx])
            idx += 1
        ext_h_list.append(np.concatenate(ext_w_list, axis=1))
    
    return np.concatenate(ext_h_list, axis=0)

### AMN
def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = np.array(new_labels)
    new_labels = torch.LongTensor(new_labels)
    return new_labels