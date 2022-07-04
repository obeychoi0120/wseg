import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import functional as tvf
from eps import get_eps_loss
from util import pyutils
import random
import numpy as np

from module.validate import validate


def max_norm(p, e=1e-5):
    if p.dim() == 3:
        C, H, W = p.size()
        # p = F.relu(p)
        max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
        # min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
        # p = F.relu(p-min_v-e)/(max_v-min_v+e)
        p = p / (max_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
        min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
        p = F.relu(p-min_v-e)/(max_v-min_v+e)
        # p = p / (max_v + e)
    return p


def get_contrast_loss(cam1, cam2, f_proj1, f_proj2, label, gamma=0.05, bg_thres=0.05):

    n1, c1, h1, w1 = cam1.shape
    n2, c2, hw, w2 = cam2.shape
    assert n1 == n2

    bg_score = torch.ones((n1, 1)).cuda()
    label = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)
    f_proj1 = F.interpolate(f_proj1, size=(128//8, 128//8), mode='bilinear', align_corners=True)
    cam1 = F.interpolate(cam1, size=(128//8, 128//8), mode='bilinear', align_corners=True)

    with torch.no_grad():

        fea1 = f_proj1.detach()
        c_fea1 = fea1.shape[1]
        cam_rv1_down = F.relu(cam1.detach())

        n1, c1, h1, w1 = cam_rv1_down.shape
        max1 = torch.max(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        min1 = torch.min(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        cam_rv1_down[cam_rv1_down < min1 + 1e-5] = 0.
        norm_cam1 = (cam_rv1_down - min1 - 1e-5) / (max1 - min1 + 1e-5)
        # norm_cam1 = cam_rv1_down / (max1 + 1e-5)
        cam_rv1_down = norm_cam1
        cam_rv1_down[:, -1, :, :] = bg_thres
        scores1 = F.softmax(cam_rv1_down * label, dim=1)

        pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
        n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape
        scores1 = scores1.transpose(0, 1)  # (21, N, H/8, W/8)
        fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)  # (nhw, 128)
        top_values, top_indices = torch.topk(cam_rv1_down.transpose(0, 1).reshape(c_sc1, -1),
                                             k=h_sc1*w_sc1//16, dim=-1)
        prototypes1 = torch.zeros(c_sc1, c_fea1).cuda()  # [21, 128]
        for i in range(c_sc1):
            top_fea = fea1[top_indices[i]]
            prototypes1[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
        prototypes1 = F.normalize(prototypes1, dim=-1)

        # For target
        fea2 = f_proj2.detach()
        c_fea2 = fea2.shape[1]

        cam_rv2_down = F.relu(cam2.detach())
        n2, c2, h2, w2 = cam_rv2_down.shape
        max2 = torch.max(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        min2 = torch.min(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        cam_rv2_down[cam_rv2_down < min2 + 1e-5] = 0.
        norm_cam2 = (cam_rv2_down - min2 - 1e-5) / (max2 - min2 + 1e-5)

        # max norm
        cam_rv2_down = norm_cam2
        cam_rv2_down[:, -1, :, :] = bg_thres

        scores2 = F.softmax(cam_rv2_down * label, dim=1)
        # pseudo_label2
        pseudo_label2 = scores2.argmax(dim=1, keepdim=True)

        n_sc2, c_sc2, h_sc2, w_sc2 = scores2.shape
        scores2 = scores2.transpose(0, 1)  # (21, N, H/8, W/8)
        fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)  # (N*C*H*W)
        top_values2, top_indices2 = torch.topk(cam_rv2_down.transpose(0, 1).reshape(c_sc2, -1), k=h_sc2*w_sc2//16, dim=-1)
        prototypes2 = torch.zeros(c_sc2, c_fea2).cuda()

        for i in range(c_sc2):
            top_fea2 = fea2[top_indices2[i]]
            prototypes2[i] = torch.sum(top_values2[i].unsqueeze(-1) * top_fea2, dim=0) / torch.sum(top_values2[i])

        # L2 Norm
        prototypes2 = F.normalize(prototypes2, dim=-1)

    # Contrast Loss
    n_f, c_f, h_f, w_f = f_proj1.shape
    f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f*h_f*w_f, c_f)
    f_proj1 = F.normalize(f_proj1, dim=-1)
    pseudo_label1 = pseudo_label1.reshape(-1)
    positives1 = prototypes2[pseudo_label1]
    negitives1 = prototypes2
    n_f, c_f, h_f, w_f = f_proj2.shape
    f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f*h_f*w_f, c_f)
    f_proj2 = F.normalize(f_proj2, dim=-1)
    pseudo_label2 = pseudo_label2.reshape(-1)
    positives2 = prototypes1[pseudo_label2]  # (N, 128)
    negitives2 = prototypes1
    A1 = torch.exp(torch.sum(f_proj1 * positives1, dim=-1) / 0.1)
    A2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1))/0.1), dim=-1)
    loss_nce1 = torch.mean(-1 * torch.log(A1 / A2))

    A3 = torch.exp(torch.sum(f_proj2 * positives2, dim=-1) / 0.1)
    A4 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1))/0.1), dim=-1)
    loss_nce2 = torch.mean(-1 * torch.log(A3 / A4))

    loss_cross_nce = gamma * (loss_nce1 + loss_nce2) / 2

    A1_view1 = torch.exp(torch.sum(f_proj1 * positives2, dim=-1) / 0.1)
    A2_view1 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives2.transpose(0, 1))/0.1), dim=-1)
    loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view1 / A2_view1))

    A3_view2 = torch.exp(torch.sum(f_proj2 * positives1, dim=-1) / 0.1)
    A4_view2 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives1.transpose(0, 1))/0.1), dim=-1)

    loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

    loss_cross_nce2 = gamma * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2

    positives_intra1 = prototypes1[pseudo_label1]
    negitives_intra1 = prototypes1
    similarity_intra1 = (torch.sum(f_proj1 * positives_intra1, dim=-1) + 1) / 2.

    A1_intra_view1 = torch.exp(torch.sum(f_proj1 * positives_intra1, dim=-1) / 0.1)
    neg_scores = torch.matmul(f_proj1, negitives_intra1.transpose(0, 1))  # (n*h*w, 21)
    with torch.no_grad():
        random_indices = torch.tensor([random.sample(range(21), 10) for _ in range(n_f * h_f * w_f)]).long()

    with torch.no_grad():
        _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
        lower_indices = lower_indices[:, 3:]

    negitives_intra1 = negitives_intra1.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
    random_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), random_indices]
    lower_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), lower_indices]
    negitives_intra1 = torch.cat([positives_intra1.unsqueeze(1), lower_negitives_intra1], dim=1)
    A2_intra_view1 = torch.sum(torch.exp(torch.matmul(f_proj1.unsqueeze(1), negitives_intra1.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)

    loss_intra_nce1 = torch.zeros(1).cuda()
    C = 0
    exists = np.unique(pseudo_label1.cpu().numpy()).tolist()
    for i_ in range(21):  # for each class
        if not i_ in exists:
            continue
        C += 1
        A1_intra_view1_class = A1_intra_view1[pseudo_label1 == i_]
        A2_intra_view1_class = A2_intra_view1[pseudo_label1 == i_]
        similarity_intra1_class = similarity_intra1[pseudo_label1 == i_]

        len_class = A1_intra_view1_class.shape[0]
        if len_class < 2:
            continue

        with torch.no_grad():
            random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
        random_A1_intra_view1 = A1_intra_view1_class[random_indices]  # (n, hw//2)
        random_A2_intra_view1 = A2_intra_view1_class[random_indices]

        with torch.no_grad():
            _, lower_indices = torch.topk(similarity_intra1_class, k=int(len_class * 0.6), largest=False)
            lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

        lower_A1_intra_view1 = A1_intra_view1_class[lower_indices]
        lower_A2_intra_view1 = A2_intra_view1_class[lower_indices]

        A1_intra_view1_class = torch.cat([random_A1_intra_view1, lower_A1_intra_view1], dim=0)  # (hw)
        A2_intra_view1_class = torch.cat([random_A2_intra_view1, lower_A2_intra_view1], dim=0)
        A1_intra_view1_class = A1_intra_view1_class.reshape(-1)
        A2_intra_view1_class = A2_intra_view1_class.reshape(-1)
        loss_intra_nce1 += torch.mean(-1 * torch.log(A1_intra_view1_class / A2_intra_view1_class))

    # mea over classes
    loss_intra_nce1 = loss_intra_nce1 / C

    # for target
    positives_intra2 = prototypes2[pseudo_label2]
    negitives_intra2 = prototypes2
    similarity_intra2 = (torch.sum(f_proj2 * positives_intra2, dim=-1) + 1) / 2.

    A3_intra_view2 = torch.exp(torch.sum(f_proj2 * positives_intra2, dim=-1) / 0.1)
    neg_scores = torch.matmul(f_proj2, negitives_intra2.transpose(0, 1))  # (n*h*w, 21)

    with torch.no_grad():
        random_indices = torch.tensor([random.sample(range(21), 10) for _ in range(n_f * h_f * w_f)]).long()

    with torch.no_grad():
        _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
        lower_indices = lower_indices[:, 3:]

    negitives_intra2 = negitives_intra2.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
    random_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), random_indices]
    lower_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), lower_indices]
    negitives_intra2 = torch.cat([positives_intra2.unsqueeze(1), lower_negitives_intra2], dim=1)

    A4_intra_view2 = torch.sum(torch.exp(torch.matmul(f_proj2.unsqueeze(1), negitives_intra2.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)
    loss_intra_nce2 = torch.zeros(1).cuda()
    C = 0
    exists = np.unique(pseudo_label2.cpu().numpy()).tolist()
    for i_ in range(21):
        if not i_ in exists:
            continue
        C += 1
        A3_intra_view2_class = A3_intra_view2[pseudo_label2 == i_]
        A4_intra_view2_class = A4_intra_view2[pseudo_label2 == i_]
        similarity_intra2_class = similarity_intra2[pseudo_label2 == i_]
        len_class = A3_intra_view2_class.shape[0]

        if len_class < 2:
            continue

        with torch.no_grad():
            random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
        random_A3_intra_view2 = A3_intra_view2_class[random_indices]  # (n, hw//2)
        random_A4_intra_view2 = A4_intra_view2_class[random_indices]
        with torch.no_grad():  # lowest 50%
            _, lower_indices = torch.topk(similarity_intra2_class, k=int(len_class * 0.6), largest=False)
            lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

        lower_A3_intra_view2 = A3_intra_view2_class[lower_indices]
        lower_A4_intra_view2 = A4_intra_view2_class[lower_indices]
        A3_intra_view2_class = torch.cat([random_A3_intra_view2, lower_A3_intra_view2], dim=0)
        A4_intra_view2_class = torch.cat([random_A4_intra_view2, lower_A4_intra_view2], dim=0)
        A3_intra_view2_class = A3_intra_view2_class.reshape(-1)
        A4_intra_view2_class = A4_intra_view2_class.reshape(-1)

        loss_intra_nce2 += torch.mean(-1 * torch.log(A3_intra_view2_class / A4_intra_view2_class))

    loss_intra_nce2 = loss_intra_nce2 / C

    loss_intra_nce = gamma * (loss_intra_nce1 + loss_intra_nce2) / 2

    loss_nce = loss_cross_nce + loss_cross_nce2 + loss_intra_nce

    return loss_nce


def get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label):

    ns, cs, hs, ws = cam2.size()
    cam1 = F.interpolate(max_norm(cam1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    # cam1 = F.softmax(cam1, dim=1) * label
    # cam2 = F.softmax(cam2, dim=1) * label
    cam2 = max_norm(cam2) * label
    loss_er = torch.mean(torch.abs(cam1[:, :-1, :, :] - cam2[:, :-1, :, :]))

    cam1[:, -1, :, :] = 1 - torch.max(cam1[:, :-1, :, :], dim=1)[0]
    cam2[:, -1, :, :] = 1 - torch.max(cam2[:, :-1, :, :], dim=1)[0]
    cam_rv1 = F.interpolate(max_norm(cam_rv1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    cam_rv2 = max_norm(cam_rv2) * label
    tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
    tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
    loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr = loss_ecr1 + loss_ecr2

    return loss_er, loss_ecr


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance,
    # but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


def consistency_2d_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
        
    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=1)
        max_probs, max_idx = torch.max(pseudo_label, dim=1)
        mask = max_probs.ge(p_cutoff).float()
        mask = torch.where(max_idx < 20, mask, torch.zeros_like(mask)) # ignore background confidence
        select = max_probs.ge(p_cutoff).long()
        # strong_prob, strong_idx = torch.max(torch.softmax(logits_s, dim=-1), dim=-1)
        # strong_select = strong_prob.ge(p_cutoff).long()
        # select = select * strong_select * (strong_idx == max_idx)
        if use_hard_labels:
            masked_loss = ce_2d_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=1)
            masked_loss = ce_2d_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()


def ce_2d_loss(preds, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        log_pred = F.log_softmax(preds, dim=1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert preds.shape == targets.shape
        log_pred = F.log_softmax(preds, dim=1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class NCESoftmaxLoss(nn.Module): ###
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


def cam_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def transform_cam(cam, i, j, h, w, hor_flip, args):
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


def consistency_cam_loss(cam_from_aug, augmented_cam, criterion):
    cam_from_aug = cam_normalize(cam_from_aug.flatten(1))
    augmented_cam = cam_normalize(augmented_cam.flatten(1))

    pos = (cam_from_aug * augmented_cam).sum(1)
    bsize = pos.shape[0]
    neg = torch.mm(cam_from_aug, cam_from_aug.transpose(1, 0))
    neg = neg[(1-torch.eye(bsize)).bool()].view(-1, bsize-1)
    out = torch.cat((pos.view(bsize, 1), neg), dim=1)
    
    return criterion(out)


def train_cls(train_loader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
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
            loss = F.multilabel_soft_margin_loss(pred, label)
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_eps(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
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
                                                              intermediate=True)
            loss = loss_cls + loss_sal

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_contrast(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
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
            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True)

            # Classification loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)

            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True)

            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True)

            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)

            # loss cls = cam cls loss + cam_cv cls loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.

            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            loss = loss_cls + loss_sal + loss_nce + loss_er + loss_ecr

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'Loss_Nce:%.4f' % (avg_meter.pop('loss_nce')),
                      'Loss_ER: %.4f' % (avg_meter.pop('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.pop('loss_ecr')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
            # Validate 10 times
            if (optimizer.global_step-1) % (max_step // 10) == 0:
                validate(model, val_dataloader, iteration, args)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_contrast.pth'))


### contrast + semi-supervised learning ###
# T(f(x)) <=> f(T(x)) (CAM-wise paring)
def train_contrast_ssl(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr', 'loss_ssl') ###
    timer = pyutils.Timer("Session started: ")
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) ###
    gamma = 0.10
    contrastive_criterion = NCESoftmaxLoss(args.T).cuda() ###
    hor_flip_tr = torchvision.transforms.RandomHorizontalFlip() ###
    print(args)
    print('Using Gamma:', gamma)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(lb_loader_iter)
                ulb_img_id, ulb_img, _, _ = next(ulb_loader_iter)   ###
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(lb_loader_iter)
                ulb_loader_iter = iter(train_ulb_dataloader)        ###
                ulb_img_id, ulb_img, _, _ = next(ulb_loader_iter)   ###

            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            ulb_img = ulb_img.cuda(non_blocking=True)               ###

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
            pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)
            
            if iteration+1 >= args.warmup_iter:
                ulb_img = torch.cat([img, ulb_img], dim=0)
                
                # Augmentation
                i, j, h, w, hor_flip = [], [], [], [], []
                ulb_img_aug = torch.zeros_like(ulb_img).cuda(non_blocking=True)
                for idx, b_img in enumerate(ulb_img):
                    ti, tj, th, tw = torchvision.transforms.RandomResizedCrop.get_params(b_img, scale=(0.08, 1.0),
                                                                            ratio=(0.75, 1.3333333333333333))
                    ulb_img_aug[idx] = tvf.resized_crop(b_img, ti, tj, th, tw, size=(b_img.size(-2), b_img.size(-1)))
                    t_hor_flip = False
                    if random.random() > 0.5:
                        ulb_img_aug[idx] = hor_flip_tr(ulb_img_aug[idx])
                        t_hor_flip = True
                    i.append(ti)
                    j.append(tj)
                    h.append(th)
                    w.append(tw)
                    hor_flip.append(t_hor_flip)

                ulb_pred1, ulb_cam1, ulb_pred_rv1, ulb_cam_rv1, ulb_feat1 = model(ulb_img)  ###
                ulb_aug_pred1, ulb_aug_cam1, ulb_aug_pred_rv1, ulb_aug_cam_rv1, ulb_aug_feat1 = model(ulb_img_aug)  ###

                ulb_cam1_aug = transform_cam(ulb_cam1, i, j, h, w, hor_flip, args)

            # Classification loss 1
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True)

            # Classification loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True)
            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True)

            # Classification_rv loss
            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            # ER Loss
            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # Contrast Loss
            loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)

            # loss cls = cam cls loss + cam_cv cls loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            # Total Loss
            loss = loss_cls + loss_sal + loss_nce + loss_er + loss_ecr

            ### Semi-supervsied Learning ###
            if iteration+1 >= args.warmup_iter: ###
                loss_ssl = consistency_cam_loss(ulb_aug_cam1, ulb_cam1_aug, contrastive_criterion)
                loss += loss_ssl * args.ssl_lambda ###
            else:
                loss_ssl = torch.zeros(1)

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_ssl': loss_ssl.item()})
                            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'Loss_Nce:%.4f' % (avg_meter.pop('loss_nce')),
                      'Loss_ER: %.4f' % (avg_meter.pop('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.pop('loss_ecr')),
                      'Loss_SSL:%.4f' % (avg_meter.pop('loss_ssl')),    ###
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
            # Validate 10 times
            if (optimizer.global_step-1) % (max_step // 10) == 0:
                validate(model, val_dataloader, iteration, args)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_contrast.pth'))


# Low resolution CAM as Pseudo-label
def train_contrast_ssl_lowres(train_dataloader, train_ulb_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_nce', 'loss_er', 'loss_ecr', 'loss_ssl') ###
    timer = pyutils.Timer("Session started: ")
    lb_loader_iter = iter(train_dataloader)
    ulb_loader_iter = iter(train_ulb_dataloader) ###
    gamma = 0.10
    print(args)
    print('Using Gamma:', gamma)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(lb_loader_iter)
                ulb_img_id, ulb_img, _, _ = next(ulb_loader_iter)   ###
            except:
                lb_loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(lb_loader_iter)
                ulb_loader_iter = iter(train_ulb_dataloader)        ###
                ulb_img_id, ulb_img, _, _ = next(ulb_loader_iter)   ###

            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            ulb_img = ulb_img.cuda(non_blocking=True)               ###

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
            pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img2)
            
            if iteration+1 >= args.warmup_iter:
                ulb_img = torch.cat([img, ulb_img], dim=0)
                ulb_pred1, ulb_cam1, ulb_pred_rv1, ulb_cam_rv1, ulb_feat1 = model(ulb_img)  ###
                with torch.no_grad():
                    ulb_img2 = F.interpolate(ulb_img, size=(128, 128), mode='bilinear', align_corners=True) ### strong(?) aug
                    ulb_pred2, ulb_cam2, ulb_pred_rv2, ulb_cam_rv2, ulb_feat2 = model(ulb_img2) ###

                    ulb_cam2 = F.interpolate(ulb_cam2, size=(56, 56), mode='bilinear', align_corners=True) 

            # Classification loss 1
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha, intermediate=True)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True)

            # Classification loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha, intermediate=True)
            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True)

            # Classification_rv loss
            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            # ER Loss
            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            # Contrast Loss
            loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)

            # loss cls = cam cls loss + cam_cv cls loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            # Total Loss
            loss = loss_cls + loss_sal + loss_nce + loss_er + loss_ecr

            ### Semi-supervsied Learning ###
            if iteration+1 >= args.warmup_iter: ###
                loss_ssl, masked_pixel, selected_pixel, pseudo_lb = consistency_2d_loss(ulb_cam1, ulb_cam2, 'ce', 
                                                                            T=args.T, p_cutoff=args.p_cutoff, use_hard_labels=args.hard_label)
                loss += loss_ssl * args.ssl_lambda ###
            else:
                loss_ssl = torch.zeros(1)
                masked_pixel = torch.zeros(1)

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_ssl': loss_ssl.item()})
                            ###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'Loss_Nce:%.4f' % (avg_meter.pop('loss_nce')),
                      'Loss_ER: %.4f' % (avg_meter.pop('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.pop('loss_ecr')),
                      'Loss_SSL:%.4f' % (avg_meter.pop('loss_ssl')),    ###
                      'Mask_ratio:%.4f' % (1.0 - masked_pixel.detach()),  ###
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            
            # Validate 10 times
            if (optimizer.global_step-1) % (max_step // 10) == 0:
                validate(model, val_dataloader, iteration, args)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_contrast.pth'))
