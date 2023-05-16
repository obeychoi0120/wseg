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

def gauss(x, mean, std):
    return 1 / (math.sqrt(2 * math.pi * pow(std, 2))) * pow(math.e, -1 * (pow(x - mean, 2) / (2 * pow(std, 2))))

def return_gau_mask():
    mean = 0
    std = 2
    size_matrix = 5
    hw = 56 * 56
    double_matrix = np.zeros([hw, hw])
    i = 0
    j = 0
    for row_idx in range(hw):
        if j > 55:
            i += 1
            j = 0

        min_i = max(i - (size_matrix // 2), 0)
        max_i = min(i + (size_matrix // 2), 55)
        min_j = max(j - (size_matrix // 2), 0)
        max_j = min(j + (size_matrix // 2), 55)

        for ii in range(min_i, max_i + 1):
            for jj in range(min_j, max_j + 1):
                double_matrix[row_idx][ii * 56 + jj] = gauss(math.sqrt(pow(ii - i, 2) + pow(jj - j, 2)), mean, std)
        j += 1

    return double_matrix

class Attn(nn.Module):
    """ Feat-CAM attention Layer"""
    def __init__(self):
        super(Attn, self).__init__()
        #self.query_conv = nn.Conv2d(in_channels = qk_dim , out_channels = qk_dim//8 , kernel_size= 1)
        #self.key_conv = nn.Conv2d(in_channels = qk_dim , out_channels = qk_dim//8 , kernel_size= 1)
        #self.value_conv = nn.Conv2d(in_channels = v_dim , out_channels = v_dim , kernel_size= 1)
        self.softmax  = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.gau_kernel = torch.tensor(return_gau_mask()).to('cuda').float()
        self.focal_mask = focal_mask(56, 64)
    def forward(self, q, k, v, args):
        """
            inputs :
                x : input feature maps(B X C X H X W)
                v : input CAM         (B X 21 X H X W)
            returns :
                out : self attention value
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = v.size()
        mask        = torch.max(v, dim=1).values.ge(args.p_cutoff).float()         # B, H, W
        mask_q      = (1 - mask).unsqueeze(dim=1).view(B, -1, H*W).permute(0,2,1)  # B, HW, 1
        mask_k      = mask.unsqueeze(dim=1).view(B, -1, H*W)                       # B, 1, HW
        mask_qk    = torch.bmm(mask_q, mask_k)                                     # B, HW, HW
        proj_query  = q.view(B, -1, H*W).permute(0,2,1)                 # B, C, HW
        proj_query  = F.normalize(proj_query, dim=1)                    # B, HW, C
        proj_key    = k.view(B, -1, H*W)                   # B, C, HW
        proj_key    = F.normalize(proj_key, dim=1)
        energy      = torch.bmm(proj_query, proj_key) / args.attn_tau
        if args.attn_type=='e':
            attention = self.softmax(energy)    # naive attn
        elif args.attn_type=='et':
            attention = self.softmax(energy*mask_qk)   # et
        # elif args.attn_type=='ef':
        #     attention = self.softmax(energy*self.focal_mask) # ef
        # elif args.attn_type=='e-f':
        #     attention = self.softmax(energy)*focal_mask # e-f
        # elif args.attn_type=='etf':
        #     attention = self.softmax(energy*mask_qk*focal_mask)    # etf
        # elif args.attn_type=='et-f':
        #     attention = self.softmax(energy*mask_qk)*focal_mask    # et-f
        elif args.attn_type=='gau':
            attention = self.softmax(energy*self.gau_kernel)

        proj_value  = v.view(B, -1, H*W)           # B, 21, HW
        out         = torch.bmm(proj_value, attention)
        out = out.view(B, -1, H, W)
        
        return out, attention

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, gamma=0.0, conv=False, remove_self_corr=False):
        super(Self_Attn, self).__init__()
        self.conv=conv
        self.remove_self_corr = remove_self_corr
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.softmax  = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        if not self.conv:
            self.query_conv = nn.Identity()
            self.key_conv = nn.Identity()
            self.value_conv = nn.Identity()

        m_batchsize, C, height, width = x.size()
        N = height*width
        proj_query  = self.query_conv(x).view(m_batchsize,-1,N)              # B, HW, C
        proj_key    = self.key_conv(x).view(m_batchsize,-1,N)                # B, C, HW
        energy      = torch.bmm(proj_query.permute(0,2,1), proj_key)                         
        attention   = self.softmax(energy * torch.logical_not(torch.eye(N, N)).float().cuda())
        proj_value  = self.value_conv(x).view(m_batchsize,-1,N)              # B, 21, HW
        out = torch.bmm(proj_value, attention)
        out = self.relu(out)
        out = out.view(m_batchsize,C,height,width)                           # B, 21, H, W
        y = self.gamma*out + x                                              
        return y, out, x, self.gamma

def focal_mask(feat_size, pow):
    src_i       = torch.arange(feat_size, device='cuda').repeat_interleave(feat_size)
    diff_i      = src_i.repeat(feat_size*feat_size, 1) - src_i.unsqueeze(1)
    src_j       = torch.arange(feat_size, device='cuda').repeat(feat_size)
    diff_j      = src_j.repeat(feat_size*feat_size, 1) - src_j.unsqueeze(1)
    diff        = (torch.sqrt(diff_i ** 2 + diff_j ** 2) / (feat_size*feat_size)).unsqueeze(0)
    mask = (1.0 - diff) ** pow 
    return mask

def get_avg_meter(args):
    log_keys = ['loss_cls', 'loss_sup']
    if args.network_type == 'seam':
        log_keys.extend(['loss_er', 'loss_ecr', 'loss_er_s', 'loss_ecr_s'])
    elif args.network_type == 'eps':
        log_keys.extend(['loss_sal'])
    elif args.network_type == 'contrast':
        log_keys.extend(['loss_er', 'loss_ecr', 'loss_er_s', 'loss_ecr_s', 'loss_nce', 'loss_sal', 'loss_sal_s'])
    if args.mode in ['v2', 'ssl']:
        log_keys.extend([
            'loss', 'loss_semcon','loss_semcon_1', 'loss_semcon_2', 'loss_semcon_3', 'loss_semcon_4', 'loss_semcon_5', 'loss_semcon_6', \
            'mask_1', 'mask_2', 'mask_3','mask_4', 'mask_5', 'mask_6', 'mask_ratio'])
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