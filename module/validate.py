import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from torch.nn import functional as F
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from util import pyutils
from data.dataset import get_categories
import wandb
from tqdm import tqdm
from module.helper import *
import pdb 

def validate(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, bg_last=False, get_dict=True)
    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]
    model.eval()
    with torch.no_grad():
        preds = []
        cams = []
        uncrts = []
        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            B = len(img_id)
            img = img.cuda()
            label = label.cuda(non_blocking=True)[:,:,None,None]    # 20, 1, 1
            logit = model.module.forward_cam(img)
            # Available only batch size 1
            # if args.network_type == 'seam':
            #     logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)
            #     logit = logit.cpu().numpy()

            #     sum_cam = np.sum(logit, axis=0)
            #     sum_cam[sum_cam < 0] = 0
            #     cam_max = np.max(sum_cam, (1,2), keepdims=True)
            #     cam_min = np.min(sum_cam, (1,2), keepdims=True)
            #     sum_cam[sum_cam < cam_min+1e-5] = 0 
            #     norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
            #     logit = torch.tensor(norm_cam).unsqueeze(0).cuda()
            #     # pdb.set_trace()
                
            # else:
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)
            logit = logit.cpu().numpy()
            sum_cam = np.sum(logit, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
            logit = torch.tensor(norm_cam).unsqueeze(0).cuda()
            
            cam = logit.clone()
            cam[:, :-1] *= label
            max_probs, pred = torch.max(logit, dim=1)
            _, cam = torch.max(cam, dim=1)
            
            # background(20 -> 0)
            pred += 1
            pred[pred==args.num_sample] = 0
            cam += 1
            cam[cam==args.num_sample] = 0

            preds.append(pred[0].cpu().numpy().copy())
            cams.append(cam[0].cpu().numpy().copy())
            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())
            

        # Calculate Metrics
        confusion = calc_semantic_segmentation_confusion(cams, labels)
        gtj = confusion.sum(axis=1)
        predj = confusion.sum(axis=0)
        gtjpredj = np.diag(confusion)
        
        # 1. BG
        tp_bg = confusion[0, 0]        
        fp_bg = gtj[0] - tp_bg
        fn_bg = predj[0] - tp_bg
        tn_bg = confusion.sum() - (tp_bg+fp_bg+fn_bg)
        
        fpr_bg = fp_bg / (fp_bg + tn_bg + 1e-10)
        fnr_bg = fn_bg / (tp_bg + fn_bg + 1e-10)

        # 2. FG
        confusion_fg = confusion[1:, 1:]
        gtj_fg = confusion_fg.sum(axis=1)
        predj_fg = confusion_fg.sum(axis=0)
        tp_fg = np.diag(confusion_fg)

        prec_fg = tp_fg / (gtj_fg + 1e-10)
        recall_fg = tp_fg / (predj_fg + 1e-10)

        # 3. ALL
        denominator = gtj + predj - gtjpredj
        precision = gtjpredj / (gtj + 1e-10)
        recall = gtjpredj / (predj + 1e-10)
        iou = gtjpredj / denominator
        acc = gtjpredj.sum() / confusion.sum()

        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {
            'miou': np.nanmean(iou),
            'mprecision': np.nanmean(precision),
            'mrecall': np.nanmean(recall),
            'accuracy': acc,
            'mFPR_fg': 1 - np.nanmean(prec_fg),
            'mFPR_bg': np.nanmean(fpr_bg),
            'mFNR_fg': 1 - np.nanmean(recall_fg),
            'mFNR_bg': np.nanmean(fnr_bg)
            }

        ### Logging Images
        N_val = 30
        for i, (cam, pred, uncrt, (img, _)) in enumerate(zip(cams[:N_val], preds[:N_val], uncrts[:N_val], gt_dataset)):
            timg[gt_dataset.ids[i]] = wandb.Image(np.transpose(img, axes=(1,2,0)),
                                                  masks={'CAM': {
                                                            'mask_data': cam,
                                                            'class_labels': idx2class},
                                                         'Prediction': {
                                                            'mask_data': pred,
                                                            'class_labels': idx2class},
                                                         'Uncertainty': {
                                                            'mask_data': uncrt,
                                                            'class_labels': {0: 'certain', 1: 'uncertain'}},
                                                         'Ground_truth': {
                                                            'mask_data': np.where(labels[i]==-1, 255, labels[i]),
                                                            'class_labels': idx2class}})
        # Logging
        print(f"mIoU     : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrec    : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall  : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy : {log_scalar['accuracy'] * 100:.2f}%")
        print('')
        print(f"mFPR_fg  : {log_scalar['mFPR_fg'] * 100:.2f}%")
        print(f"mFPR_bg  : {log_scalar['mFPR_bg'] * 100:.2f}%")
        print(f"mFNR_fg  : {log_scalar['mFNR_fg'] * 100:.2f}%")
        print(f"mFNR_bg  : {log_scalar['mFNR_bg'] * 100:.2f}%")
        print('')
        print('IoU (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print('Prec (%)  :', ' '.join([f'{v*100:0>4.1f}' for v in precision]))
        print('Recall (%):', ' '.join([f'{v*100:0>4.1f}' for v in recall]))                                              
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)

    model.train()

def validate_acc_by_class(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, get_dict=True)

    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]

    model.eval()

    with torch.no_grad():
        preds = []
        cams = []
        uncrts = []
        class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', \
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', \
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        crt_by_cr = torch.zeros(size=(len(class_list), 6))
        num_by_cr = torch.zeros(size=(len(class_list), 6))

        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            label = label.cuda(non_blocking=True)[0,:,None,None]
            gt = torch.tensor(gt).long()         # H, W
            gt[gt==-1] = 0
            gt_ohe = F.one_hot(gt, num_classes = 21).permute(2, 0, 1).cuda()  # 21, H, W
            logit = model.module.forward_cam(img)

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, gt.shape[-2:], mode='bilinear', align_corners=False)            
            cam = logit.clone()
            cam[:, :-1] *= label

            max_probs, pred = torch.max(logit, dim=1)
            _, cam = torch.max(cam, dim=1)

            # background(20 -> 0)
            pred += 1
            pred[pred==args.num_sample] = 0
            preds.append(pred[0].cpu().numpy().copy())
            cam += 1
            cam[cam==args.num_sample] = 0
            cam = cam[0].clone().detach().long()
            cam[cam==-1] = 0
            cam_ohe = F.one_hot(cam, num_classes=21).permute(2, 0, 1).cuda()  # 21, H, W

            # masks by confidence 
            mask = torch.empty(size=(6, max_probs.shape[-2], max_probs.shape[-1]))
            mask[0, :, :] = max_probs.lt(0.4)
            mask[1, :, :] = torch.logical_and(max_probs.ge(0.40), max_probs.lt(0.60))
            mask[2, :, :] = torch.logical_and(max_probs.ge(0.60), max_probs.lt(0.80))
            mask[3, :, :] = torch.logical_and(max_probs.ge(0.80), max_probs.lt(0.95))
            mask[4, :, :] = torch.logical_and(max_probs.ge(0.95), max_probs.lt(0.99))
            mask[5, :, :] = max_probs.ge(0.99)
            mask = mask.long().cuda() #  # 6, H, W
            
            crt = ((cam_ohe==gt_ohe) * cam_ohe).unsqueeze(1)   # 21, 1, H, W
            crt = crt * mask
            num = cam_ohe.unsqueeze(1) * mask

            '''
            correct[0, 0].sum()
            '''
            for class_idx in range(len(class_list)):
                for range_idx in range(mask.shape[0]):
                    crt_by_cr[class_idx, range_idx] += crt[class_idx, range_idx].sum().item()
                    num_by_cr[class_idx, range_idx] += num[class_idx, range_idx].sum().item()


            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())
            cams.append(cam.cpu().numpy())
            
        '''
        cams : 1464 * (21, H, W),  0~20
        cam_ohe : (21, H, W), binary
        masks : 1464 * (6, H, W),  binary
        corrects : 1464 * (21, 6, H, W), binary
        crt_by_c[1] / num_by_c[1]
        crt_by_c[0] / num_by_c[0]
        '''
        num_by_c = num_by_cr.sum(axis=1)
        num_by_r = num_by_cr.sum(axis=0)
        crt_by_c = crt_by_cr.sum(axis=1)
        crt_by_r = crt_by_cr.sum(axis=0)

        acc_by_cr = crt_by_cr / (num_by_cr + 1e-9)
        acc_by_r = crt_by_r / (num_by_r + 1e-9)
        acc_by_c = crt_by_c / (num_by_c + 1e-9)

        # confusion
        confusion = calc_semantic_segmentation_confusion(cams, labels)
        gtj = confusion.sum(axis=1)
        predj = confusion.sum(axis=0)
        gtjpredj = np.diag(confusion)
        denominator = gtj + predj - gtjpredj

        precision = gtjpredj / (gtj + 1e-10)
        recall = gtjpredj / (predj + 1e-10)
        iou = gtjpredj / denominator
        acc_total = gtjpredj.sum() / confusion.sum()

        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {'miou': np.nanmean(iou),
                      'mprecision': np.nanmean(precision),
                      'mrecall': np.nanmean(recall),
                      'accuracy': acc_total
                      }

        for i in range(mask.shape[0]):
            log_scalar['acc_'+ str(i)] = acc_by_r[i]

        for class_idx, c in enumerate(class_list):
            log_scalar['acc_' + c] = acc_by_c[class_idx]
            log_scalar[str('acc_' + c + '_1')] = acc_by_cr[class_idx, 0]
            log_scalar[str('acc_' + c + '_2')] = acc_by_cr[class_idx, 1]
            log_scalar[str('acc_' + c + '_3')] = acc_by_cr[class_idx, 2]
            log_scalar[str('acc_' + c + '_4')] = acc_by_cr[class_idx, 3]
            log_scalar[str('acc_' + c + '_5')] = acc_by_cr[class_idx, 4]
            log_scalar[str('acc_' + c + '_6')] = acc_by_cr[class_idx, 5]
        


        ### Logging Images
        N_val = 30
        for i, (cam, pred, uncrt, (img, _)) in enumerate(zip(cams[:N_val], preds[:N_val], uncrts[:N_val], gt_dataset)):
            timg[gt_dataset.ids[i]] = wandb.Image(np.transpose(img, axes=(1,2,0)),
                                                  masks={
                                                    'CAM': {
                                                            'mask_data': cam,
                                                            'class_labels': idx2class},
                                                    'Prediction': {
                                                            'mask_data': pred,
                                                            'class_labels': idx2class},
                                                    'Uncertainty': {
                                                            'mask_data': uncrt,
                                                            'class_labels': {0: 'certain', 1: 'uncertain'}},
                                                    'Ground_truth': {
                                                            'mask_data': np.where(gt==-1, 255, gt),
                                                            'class_labels': idx2class
                                                            }})

        # Logging
        num_by_c = [num_by_cr.sum(axis=1)[i].item() for i in range(len(class_list))]
        num_by_r = [num_by_cr.sum(axis=0)[i].item() for i in range(6)]
        # print(f"num by class : {num_by_c}")
        # print(f"num by range : {num_by_r}")
        print(f"mIoU         : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrecision   : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall      : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy     : {log_scalar['accuracy'] * 100:.2f}%")
        print( 'Accuracy (%) :', ' '.join([f'{v*100:0>4.1f}' for v in acc_by_c]))
        print( 'IoU (%)      :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Precision (%):', ' '.join([f'{v*100:0>4.1f}' for v in precision]))
        print( 'Recall (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in recall]))                                              
        
        # Wandb logging
        if args.use_wandb:
            print('wandb Logging...')
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)

    model.train()
    return np.nanmean(iou)

def validate_acc_by_class_and_conf(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, get_dict=True)

    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]

    model.eval()

    with torch.no_grad():
        preds = []
        cams = []
        uncrts = []
        class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', \
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', \
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        crt_by_cr = torch.zeros(size=(len(class_list), 6))
        num_by_cr = torch.zeros(size=(len(class_list), 6))

        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            label = label.cuda(non_blocking=True)[0,:,None,None]
            gt = torch.tensor(gt).long()         # H, W
            gt[gt==-1] = 0
            gt_ohe = F.one_hot(gt, num_classes = 21).permute(2, 0, 1).cuda()  # 21, H, W
            logit = model.module.forward_cam(img)

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, gt.shape[-2:], mode='bilinear', align_corners=False)            
            cam = logit.clone()
            cam[:, :-1] *= label

            max_probs, pred = torch.max(logit, dim=1)
            _, cam = torch.max(cam, dim=1)

            # background(20 -> 0)
            pred += 1
            pred[pred==args.num_sample] = 0
            preds.append(pred[0].cpu().numpy().copy())
            cam += 1
            cam[cam==args.num_sample] = 0
            cam = cam[0].clone().detach().long()
            cam[cam==-1] = 0
            cam_ohe = F.one_hot(cam, num_classes=21).permute(2, 0, 1).cuda()  # 21, H, W

            # masks by confidence 
            mask = torch.empty(size=(6, max_probs.shape[-2], max_probs.shape[-1]))
            mask[0, :, :] = max_probs.lt(0.4)
            mask[1, :, :] = torch.logical_and(max_probs.ge(0.40), max_probs.lt(0.60))
            mask[2, :, :] = torch.logical_and(max_probs.ge(0.60), max_probs.lt(0.80))
            mask[3, :, :] = torch.logical_and(max_probs.ge(0.80), max_probs.lt(0.95))
            mask[4, :, :] = torch.logical_and(max_probs.ge(0.95), max_probs.lt(0.99))
            mask[5, :, :] = max_probs.ge(0.99)
            mask = mask.long().cuda() #  # 6, H, W
            
            crt = ((cam_ohe==gt_ohe) * cam_ohe).unsqueeze(1)   # 21, 1, H, W
            crt = crt * mask
            num = cam_ohe.unsqueeze(1) * mask

            '''
            correct[0, 0].sum()
            '''
            for class_idx in range(len(class_list)):
                for range_idx in range(mask.shape[0]):
                    crt_by_cr[class_idx, range_idx] += crt[class_idx, range_idx].sum().item()
                    num_by_cr[class_idx, range_idx] += num[class_idx, range_idx].sum().item()


            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())
            cams.append(cam.cpu().numpy())
            

        '''
        cams : 1464 * (21, H, W),  0~20
        cam_ohe : (21, H, W), binary
        masks : 1464 * (6, H, W),  binary
        corrects : 1464 * (21, 6, H, W), binary
        crt_by_c[1] / num_by_c[1]
        crt_by_c[0] / num_by_c[0]
        '''
        num_by_c = num_by_cr.sum(axis=1)
        num_by_r = num_by_cr.sum(axis=0)
        crt_by_c = crt_by_cr.sum(axis=1)
        crt_by_r = crt_by_cr.sum(axis=0)

        acc_by_cr = crt_by_cr / (num_by_cr + 1e-9)
        acc_by_r = crt_by_r / (num_by_r + 1e-9)
        acc_by_c = crt_by_c / (num_by_c + 1e-9)

        # confusion
        confusion = calc_semantic_segmentation_confusion(cams, labels)
        gtj = confusion.sum(axis=1)
        predj = confusion.sum(axis=0)
        gtjpredj = np.diag(confusion)
        denominator = gtj + predj - gtjpredj

        precision = gtjpredj / (gtj + 1e-10)
        recall = gtjpredj / (predj + 1e-10)
        iou = gtjpredj / denominator
        acc_total = gtjpredj.sum() / confusion.sum()

        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {'miou': np.nanmean(iou),
                      'mprecision': np.nanmean(precision),
                      'mrecall': np.nanmean(recall),
                      'accuracy': acc_total
                      }

        for i in range(mask.shape[0]):
            log_scalar['acc_'+ str(i)] = acc_by_r[i]

        for class_idx, c in enumerate(class_list):
            log_scalar['acc_' + c] = acc_by_c[class_idx]
            log_scalar[str('acc_' + c + '_1')] = acc_by_cr[class_idx, 0]
            log_scalar[str('acc_' + c + '_2')] = acc_by_cr[class_idx, 1]
            log_scalar[str('acc_' + c + '_3')] = acc_by_cr[class_idx, 2]
            log_scalar[str('acc_' + c + '_4')] = acc_by_cr[class_idx, 3]
            log_scalar[str('acc_' + c + '_5')] = acc_by_cr[class_idx, 4]
            log_scalar[str('acc_' + c + '_6')] = acc_by_cr[class_idx, 5]
        


        ### Logging Images
        N_val = 30
        for i, (cam, pred, uncrt, (img, _)) in enumerate(zip(cams[:N_val], preds[:N_val], uncrts[:N_val], gt_dataset)):
            timg[gt_dataset.ids[i]] = wandb.Image(np.transpose(img, axes=(1,2,0)),
                                                  masks={
                                                    'CAM': {
                                                            'mask_data': cam,
                                                            'class_labels': idx2class},
                                                    'Prediction': {
                                                            'mask_data': pred,
                                                            'class_labels': idx2class},
                                                    'Uncertainty': {
                                                            'mask_data': uncrt,
                                                            'class_labels': {0: 'certain', 1: 'uncertain'}},
                                                    'Ground_truth': {
                                                            'mask_data': np.where(gt==-1, 255, gt),
                                                            'class_labels': idx2class
                                                            }})

        # Logging
        num_by_c = [num_by_cr.sum(axis=1)[i].item() for i in range(len(class_list))]
        num_by_r = [num_by_cr.sum(axis=0)[i].item() for i in range(6)]
        # print(f"num by class : {num_by_c}")
        # print(f"num by range : {num_by_r}")
        print(f"mIoU         : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrecision   : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall      : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy     : {log_scalar['accuracy'] * 100:.2f}%")
        print( 'Accuracy (%) :', ' '.join([f'{v*100:0>4.1f}' for v in acc_by_c]))
        print( 'IoU (%)      :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Precision (%):', ' '.join([f'{v*100:0>4.1f}' for v in precision]))
        print( 'Recall (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in recall]))                                              
        
        # Wandb logging
        if args.use_wandb:
            print('wandb Logging...')
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)

    model.train()
    return np.nanmean(iou)

def validate_legacy(model, data_loader, epoch, args):
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
        mean_precision = precision.mean().item()
        mean_recall = recall.mean().item()
        mean_f1 = f1.mean().item()

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