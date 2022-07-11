import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from torch.nn import functional as F

from util import pyutils


def validate(model, data_loader, epoch, args):
    print('\nvalidating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss')
    model.eval()

    corrects = torch.tensor([0 for _ in range(20)])
    tot_cnt = 0.0
    with torch.no_grad():
        y_true = list()
        y_pred = list()
        for i, pack in enumerate(data_loader):
            if len(pack) == 3:
                _, img, label = pack
            elif len(pack) == 4:
                _, img, _, label = pack
            else:
                img, label = pack

            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            output = model(img)
            x = output[0]

            x = x[:, :-1]

            loss = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss': loss.item()})

            x_sig = torch.sigmoid(x)
            corrects += torch.round(x_sig).eq(label).sum(0).cpu()

            y_true.append(label.cpu())
            #y_pred.append(torch.round(x_sig).cpu())
            y_pred.append(x_sig.cpu())

            tot_cnt += label.size(0)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        corrects = corrects.float() / tot_cnt
        mean_acc = torch.mean(corrects).item() * 100.0

        # if not hasattr(args, 'cls_thr'):
        #     aps = []
        #     maps = []
        #     ths = [t/100 for t in range(5, 100, 5)]
        #     for th in ths:
        #         y_pred_th = (y_pred >= th).float()

        #         ap = AP(y_true.numpy(), y_pred_th.numpy()) * 100.0
        #         mAP = ap.mean()
        #         aps.append(ap)
        #         maps.append(mAP)

        #     max_idx = maps.index(max(maps))
        #     ap = aps[max_idx]
        #     mAP = maps[max_idx]
        #     args.cls_thr = ths[max_idx]

        #     print(f'Best classification threshold value is {args.cls_thr}.')
        
        y_pred = (y_pred >= 0.5).float()
        ap = AP(y_true.numpy(), y_pred.numpy()) * 100.0
        mAP = ap.mean()

        precision, recall, f1, _ = score(y_true.numpy(), y_pred.numpy(), average=None)

        precision = torch.tensor(precision).float()
        recall = torch.tensor(recall).float()
        f1 = torch.tensor(f1).float()
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        mean_f1 = f1.mean()

    model.train()
    loss_ = val_loss_meter.pop('loss')
    corrects, precision, recall, f1 = corrects.cpu().numpy(), precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy()
    print('loss:', loss_)
    #print("Epoch({:03d})\t".format(epoch))
    print("mAP: {:.2f}\t".format(mAP))
    print("MeanACC: {:.2f}\t".format(mean_acc))
    print("MeanPRE: {:.4f}\t".format(mean_precision))
    print("MeanREC: {:.4f}\t".format(mean_recall))
    print("MeanF1: {:.4f}\t".format(mean_f1))
    print("{:10s}: {}\t".format("ClassACC", " ".join(["{:.3f}".format(x) for x in corrects])))
    print("{:10s}: {}\t".format("PRECISION", " ".join(["{:.3f}".format(x) for x in precision])))
    print("{:10s}: {}\t".format("RECALL", " ".join(["{:.3f}".format(x) for x in recall])))
    print("{:10s}: {}\n".format("F1", " ".join(["{:.3f}".format(x) for x in f1])))
    return loss_, mAP, mean_acc, mean_precision, mean_recall, mean_f1, corrects, precision, recall, f1


def average_precision(label, pred):
    epsilon = 1e-8
    #label, pred = label.numpy(), pred.numpy()
    # sort examples
    indices = pred.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(pred), 1)))

    label_ = label[indices]
    ind = label_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def AP(label, logit):
    if np.size(logit) == 0:
        return 0
    ap = np.zeros((logit.shape[1]))
    # compute average precision for each class
    for k in range(logit.shape[1]):
        # sort scores
        logits = logit[:, k]
        labels = label[:, k]
        
        ap[k] = average_precision(labels, logits)
    return ap