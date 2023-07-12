import torch
from torch.nn import functional as F

import numpy as np
import random


#### PPC Loss ####
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
        random_indices = torch.tensor([random.sample(range(c1), 10) for _ in range(n_f * h_f * w_f)]).long() ### range(21 or 81)

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
    for i_ in range(c1):  # for each class (21 or 81)
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
        random_indices = torch.tensor([random.sample(range(c1), 10) for _ in range(n_f * h_f * w_f)]).long() ### range(21 or 81)

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
    for i_ in range(c1): ### range(21 or 81)
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

#### SEAM Loss ####
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
    loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=int(cs * hs * ws * 0.2), dim=-1)[0]) ### cs == 21 or 81
    loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=int(cs * hs * ws * 0.2), dim=-1)[0]) ### cs == 21 or 81
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


#### EPS Loss ####
def get_eps_loss(cam, saliency, label, tau, alpha, intermediate=True, num_class=21):
    """
    Get EPS loss for pseudo-pixel supervision from saliency map.
    Args:
        cam (tensor): response from model with float values.
        saliency (tensor): saliency map from off-the-shelf saliency model.
        label (tensor): label information.
        tau (float): threshold for confidence area
        alpha (float): blending ratio between foreground map and background map
        intermediate (bool): if True return all the intermediates, if not return only loss.
    Shape:
        cam (N, C, H', W') where N is the batch size and C is the number of classes.
        saliency (N, 1, H, W)
        label (N, C)
    """
    b, c, h, w = cam.size()
    saliency = F.interpolate(saliency, size=(h, w))

    label_map = label.view(b, num_class-1, 1, 1).expand(size=(b, num_class-1, h, w)).bool()

    # Map selection
    label_map_fg = torch.zeros(size=(b, num_class, h, w)).bool().cuda()
    label_map_bg = torch.zeros(size=(b, num_class, h, w)).bool().cuda()

    label_map_bg[:, num_class-1] = True
    label_map_fg[:, :-1] = label_map.clone()

    sal_pred = F.softmax(cam, dim=1)

    iou_saliency = (torch.round(sal_pred[:, :-1].detach()) * torch.round(saliency)).view(b, num_class-1, -1).sum(-1) / \
                   (torch.round(sal_pred[:, :-1].detach()) + 1e-04).view(b, num_class-1, -1).sum(-1)

    valid_channel = (iou_saliency > tau).view(b, num_class-1, 1, 1).expand(size=(b, num_class-1, h, w))

    label_fg_valid = label_map & valid_channel

    label_map_fg[:, :-1] = label_fg_valid
    label_map_bg[:, :-1] = label_map & (~valid_channel)

    # Saliency loss
    fg_map = torch.zeros_like(sal_pred).cuda()
    bg_map = torch.zeros_like(sal_pred).cuda()

    fg_map[label_map_fg] = sal_pred[label_map_fg]
    bg_map[label_map_bg] = sal_pred[label_map_bg]

    fg_map = torch.sum(fg_map, dim=1, keepdim=True)
    bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)
    sal_pred = fg_map * alpha + bg_map * (1 - alpha)

    loss = F.mse_loss(sal_pred, saliency)

    if intermediate:
        return loss, fg_map, bg_map, sal_pred
    else:
        return loss
    
### AMN 
def balanced_cross_entropy(logits, labels, one_hot_labels):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """

    N, C, H, W = logits.shape

    assert one_hot_labels.size(0) == N and one_hot_labels.size(1) == C, f'label tensor shape is {one_hot_labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    loss_structure = -torch.sum(log_logits * one_hot_labels, dim=1)  # (N)

    ignore_mask_bg = torch.zeros_like(labels)
    ignore_mask_fg = torch.zeros_like(labels)
    
    ignore_mask_bg[labels == 0] = 1
    ignore_mask_fg[(labels != 0) & (labels != 255)] = 1
    
    loss_bg = (loss_structure * ignore_mask_bg).sum() / ignore_mask_bg.sum()
    loss_fg = (loss_structure * ignore_mask_fg).sum() / ignore_mask_fg.sum()

    return (loss_bg+loss_fg)/2