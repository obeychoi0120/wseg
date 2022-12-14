import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as tvf

import numpy as np
from PIL import Image


###########           Semi-supervsied Learning           ###########
def get_ssl_loss(args, iteration, pred_s=None, pred_t=None, cam_s=None, cam_t=None, feat_s=None, feat_t=None, mask=None):
    losses = {'loss_ssl': 0}

    #######                1. Logit MSE(L2) loss                 #######
    if 1 in args.ssl_type:
        pl_t = torch.sigmoid(pred_t)
        # Calculate loss by threholded class (pos neg both)
        if args.mt_p:
            class_mask = pl_t.le(1 - args.mt_p) | pl_t.ge(args.mt_p)
            losses['mask_mt'] = class_mask.float().mean()
        else:
            class_mask = torch.ones_like(pl_t)
            losses['mask_mt'] = torch.zeros(1)

        # loss_mt = consistency_loss(torch.sigmoid(pred_s), torch.sigmoid(pl_t), 'L2')
        losses['loss_mt'] = consistency_loss(torch.sigmoid(pred_s), pl_t, 'L2', mask=class_mask) # (optional) w.geometry tr.

        mt_warmup = float(np.clip(iteration / (args.mt_warmup * args.max_iters + 1e-9), 0., 1.))
        losses['loss_ssl'] += losses['loss_mt'] * args.mt_lambda * mt_warmup                    

    ######             2. Pixel-wise CAM MSE(L2) loss             ######
    if 2 in args.ssl_type:
        losses['loss_pmt'] = consistency_loss(torch.softmax(pred_s,dim=1), torch.softmax(pred_t,dim=1), 'L2', mask=mask)
        losses['loss_ssl'] += losses['loss_pmt'] * args.ssl_lambda

    ######  3. Pixel-wise CAM pseudo-labeling(FixMatch, CE) loss  ######
    ######                  loss_ssl만 사용                        ######
    if 3 in args.ssl_type:
        ratio = float(np.clip((iteration/args.max_iters)+1e-9 , 0., 1.))    # 0~1 
        cutoff = args.p_cutoff - ratio*(args.p_cutoff - args.min_p_cutoff)  # min_p_cutoff: 0.8
        losses['loss_pl'], losses['mask_pl'], _, pseudo_label = consistency_loss(cam_s, cam_t, 'ce', args.T, cutoff, args.soft_label)
        losses['loss_ssl'] += losses['loss_pl'] * args.ssl_lambda

    ######           4. T(f(x)) <=> f(T(x)) InfoNCE loss          ######
    if 4 in args.ssl_type:
        losses['loss_con'] = consistency_cam_loss(torch.softmax(cam_s, dim=1), torch.softmax(cam_t, dim=1), mask)
        
        warmup = float(np.clip(iteration / (args.mt_warmup * args.max_iters + 1e-9), 0., 1.))
        losses['loss_ssl'] += losses['loss_con'] * args.ssl_lambda * warmup

    ######      5. Class Discriminative(Divide) Contrastive loss       ######
    if 5 in args.ssl_type:
        losses['loss_cdc'], losses['mask_cdc_pos'], losses['mask_cdc_neg'] = class_discriminative_contrastive_loss(cam_s, feat_s, args.p_cutoff, inter=args.cdc_inter, temperature=args.cdc_T, normalize=args.cdc_norm)
        losses['loss_ssl'] += losses['loss_cdc'] * args.cdc_lambda

    return losses, cutoff
    

def consistency_loss(logits_s, logits_t, name='L2', T=1.0, p_cutoff=0.0, use_soft_label=False, mask=None):
    logits_t = logits_t.detach()
    if name == 'L2':
        assert logits_s.size() == logits_t.size()
        if mask is not None:
            masked_loss = F.mse_loss(logits_s, logits_t, reduction='none') * mask
            return masked_loss.mean()
        else:
            return F.mse_loss(logits_s, logits_t, reduction='mean')

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_t, dim=1)
        max_probs, max_idx = torch.max(pseudo_label, dim=1)
        mask = max_probs.ge(p_cutoff).float()   # greater or equal than
        #mask = torch.where(max_idx < 20, mask, torch.zeros_like(mask)) # ignore background confidence
        select = max_probs.ge(p_cutoff).long()
        # strong_prob, strong_idx = torch.max(torch.softmax(logits_s, dim=-1), dim=-1)
        # strong_select = strong_prob.ge(p_cutoff).long()
        # select = select * strong_select * (strong_idx == max_idx)
        if not use_soft_label:
            masked_loss = ce_loss(logits_s, max_idx, use_soft_label, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_t / T, dim=1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_soft_label) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()


def ce_loss(preds, targets, use_soft_label=False, reduction='none'):
    if not use_soft_label:
        log_pred = F.log_softmax(preds, dim=1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert preds.shape == targets.shape
        log_pred = F.log_softmax(preds, dim=1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class NCESoftmaxLoss(nn.Module): ### useless
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self, T=0.01):
        super(NCESoftmaxLoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # pdb.set_trace()
        bsz = x.shape[0]
        x = x.squeeze()
        x = torch.div(x, self.T)
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


def transform_cam(cam, i, j, h, w, hor_flip, args): ### useless
    gpu_batch_len = cam.size(0)
    cam_size = cam.size(-1)
    aug_cam = torch.zeros_like(cam).cuda()
    scale = cam.size(-1) / args.crop_size
    for b in range(gpu_batch_len):
        # convert orig_gradcam_mask to image
        orig_gcam = cam[b]
        orig_gcam = orig_gcam[:, int(i[b]*scale): int(i[b]*scale) + int(h[b]*scale), 
                                 int(j[b]*scale): int(j[b]*scale) + int(w[b]*scale)]
        # We use torch functional to resize without breaking the graph
        orig_gcam = orig_gcam.unsqueeze(0)
        orig_gcam = F.interpolate(orig_gcam, size=cam_size, mode='bilinear')
        orig_gcam = orig_gcam.squeeze()
        if hor_flip[b]:
            orig_gcam = orig_gcam.flip(-1)
        aug_cam[b, :, :] = orig_gcam
    return aug_cam


# functional form of NCESoftmaxLoss
def nce_softmax_loss(x, T=0.01):
    bsz = x.shape[0]
    x = x.squeeze()
    x = torch.div(x, T)
    label = torch.zeros([bsz]).cuda().long()
    loss = F.cross_entropy(x, label)
    return loss


def cam_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def consistency_cam_loss(cam_from_aug, augmented_cam, mask=None):
    if mask is not None:
        cam_from_aug = cam_from_aug * mask
        augmented_cam = augmented_cam * mask
    cam_from_aug = cam_normalize(cam_from_aug.flatten(1))
    augmented_cam = cam_normalize(augmented_cam.flatten(1))

    pos = (cam_from_aug * augmented_cam).sum(1)
    bsize = pos.shape[0]
    neg = torch.mm(cam_from_aug, cam_from_aug.transpose(1, 0))
    neg = neg[(1-torch.eye(bsize)).bool()].view(-1, bsize-1)
    out = torch.cat((pos.view(bsize, 1), neg), dim=1)
    
    return nce_softmax_loss(out)


# Implementation from https://fyubang.com/2019/06/01/ema/
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def apply_strong_tr(img, ops, strong_transforms=None, fill_background=False):
    b, c, h, w = img.size()

    ops = torch.stack([torch.stack(op,dim=0) for op in ops], dim=0) # (N_transforms, 2(i,v), Batch_size)
    img = img.detach().clone()
    for idxs, vals in ops:
        for i, (idx, val) in enumerate(zip(idxs, vals)):
            idx, val = int(idx.item()), val.item()
            kwargs = strong_transforms[idx](img[i], val)
            if fill_background:
                # replace fillcolor into fill after 0.10
                kwargs['fillcolor'] = torch.zeros_like(img[i,:,0,0])
                kwargs['fillcolor'][-1] = img[i].max()
            # reample: NEAREST or BILINEAR, replace resample into interpolation(:InterpolationMode) after 0.10
            img[i,:] = tvf.affine(img[i], resample=Image.BILINEAR, **kwargs) 
    return img


def rand_bbox(size, l):
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - l)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    

def cutmix(img_ulb, target, mask=None):
    mix_img_ulb = img_ulb.clone()
    mix_target = target.clone()
    if mask is not None:
        mix_mask = mask.clone()

    x_r = img_ulb.size(-1) / target.size(-1)
    y_r = img_ulb.size(-2) / target.size(-2)
    # cam size == feat size
    
    u_rand_index = torch.randperm(img_ulb.size()[0])[:img_ulb.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(img_ulb.size(), l=np.random.beta(4, 4))

    u_bbx1_t, u_bby1_t, u_bbx2_t, u_bby2_t = (u_bbx1//x_r).astype(np.int32), (u_bby1//y_r).astype(np.int32), \
                                             (u_bbx2//x_r).astype(np.int32), (u_bby2//y_r).astype(np.int32)

    for i in range(0, mix_img_ulb.size(0)):
        mix_img_ulb[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            img_ulb[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_target[i, :, u_bbx1_t[i]:u_bbx2_t[i], u_bby1_t[i]:u_bby2_t[i]] = \
            target[u_rand_index[i], :, u_bbx1_t[i]:u_bbx2_t[i], u_bby1_t[i]:u_bby2_t[i]]
        
        if mask is not None:
            mix_mask[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del img_ulb, target

    if mask is not None:
        return mix_img_ulb, mix_target, mix_mask
    else:    
        return mix_img_ulb, mix_target


def class_discriminative_contrastive_loss(cam, feat, p_cutoff=0., inter=False, temperature=0.07, eps=1e-9, normalize=True):
    B, FS, H, W = feat.size()
    if inter:
        cam = cam.permute(0,2,3,1).reshape(B*H*W, cam.size(1)).detach()
        feat = feat.permute(1,0,2,3).view(FS, -1)
    else:
        # cam = cam.permute(0,2,3,1).view(B, H*W, cam.size(1)).detach()
        # feat = feat.permute(0,2,3,1).view(B, H*W, FS)
        cam = cam.view(B, cam.size(1), -1).detach()
        feat = feat.view(B, FS, -1)

    if normalize:
        feat = F.normalize(feat)

    # Calculate Feature Similarity
    # feature_contrast = torch.div( torch.matmul(feat, feat.transpose(-2,-1)), temperature )
    feature_contrast = torch.div(torch.matmul(feat.transpose(-2,-1), feat), temperature)
    
    # Normalize
    features_max, _ = torch.max(feature_contrast, dim=-1, keepdim=True)
    logits = feature_contrast - features_max.detach()

    # Make Pseudo-labels, Masks
    with torch.no_grad():
        # Make pseudo label
        pseudo_label = torch.softmax(cam, dim=-2)
        max_probs, max_idx = torch.max(pseudo_label, dim=-2, keepdim=True)

        # Thresholding Mask
        th_mask_1d = max_probs.ge(p_cutoff).float()
        th_mask = torch.matmul(th_mask_1d.transpose(-2,-1), th_mask_1d)

        # Remove Correlation of Background pixels
        bg_mask_1d = (max_idx == (cam.size(-2) - 1)).float()
        bg_mask = torch.matmul(bg_mask_1d.transpose(-2,-1), bg_mask_1d).logical_not()

        # # Class Mask (Same class positive, other negative)
        class_mask_1d = torch.zeros_like(cam).scatter(-2, max_idx, 1.)
        class_mask = torch.matmul(class_mask_1d.transpose(-2,-1), class_mask_1d)
        # class_mask = torch.matmul(cam.transpose(-2,-1), cam).ge(p_cutoff*p_cutoff).float()

        # Ignore self-similarity
        if inter:
            self_mask = torch.ones_like(logits).scatter(-1, 
                                                        torch.arange(B*H*W).view(-1, 1).to(logits.device), 
                                                        0)      
        else:
            self_mask = torch.ones_like(logits).scatter(-1, 
                                                        torch.arange(H*W).view(-1, 1).repeat(B,1,1).to(logits.device), 
                                                        0)      
        # Final Masks
        logit_mask = self_mask * th_mask * bg_mask
        mask = logit_mask * class_mask


    # Log_prob (for all Similairites)
    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(-1, keepdim=True) + eps)

    # # compute mean of log-likelihood (Positive)
    pos_mean_log_prob = (mask * log_prob).sum(-1) / (mask.sum(-1) + eps)

    # # Final loss
    loss = - pos_mean_log_prob # (temperature / base_temperature)
    return loss.mean(), mask.mean().detach(), logit_mask.mean().detach()