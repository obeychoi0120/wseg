import numpy as np
import torch
from chainercv.evaluations import calc_semantic_segmentation_confusion

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
    masks = []
    _pseudo_label = torch.softmax(cam, dim=1)
    _max_probs, _ = torch.max(_pseudo_label, dim=1)
    masks.append(_max_probs.lt(0.4).float())
    masks.append(torch.logical_and(_max_probs.ge(0.4), _max_probs.lt(0.6)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.6), _max_probs.lt(0.8)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.8), _max_probs.lt(0.95)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.95), _max_probs.lt(0.99)).float())
    masks.append(_max_probs.ge(0.99).float())
    return masks, _max_probs

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
    



    confusion.sum(axis=0)
    confusion.sum(axis=1)