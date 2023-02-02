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

def validate(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, get_dict=True)

    # # Network type
    # if 'cls' in args.network:
    #     idx_cam = 0
    # elif 'seam' in args.network:
    #     idx_cam = 3
    # elif 'eps' in args.network or 'contrast' in args.network:
    #     idx_cam = 1
    # else:
    #     raise Exception('No appropriate model type')

    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]

    model.eval()

    with torch.no_grad():
        preds = []
        cams = []
        uncrts = []
        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            label = label.cuda(non_blocking=True)[0,:,None,None]

            logit = model.module.forward_cam(img)
            # logit = pack[idx_cam]

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)
            # logit_norm = (F.adaptive_max_pool2d(logit, (1, 1)) + 1e-5)
            # logit /= logit_norm
            
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
            cams.append(cam[0].cpu().numpy().copy())
            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())

        # Calcaulate Metrics
        confusion = calc_semantic_segmentation_confusion(cams, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj

        precision = gtjresj / (gtj + 1e-10)
        recall = gtjresj / (resj + 1e-10)
        iou = gtjresj / denominator
        acc = gtjresj.sum() / confusion.sum()

        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {'miou': np.nanmean(iou),
                      'mprecision': np.nanmean(precision),
                      'mrecall': np.nanmean(recall),
                      'accuracy': acc}
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
        print(f"mIoU         : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrecision   : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall      : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy     : {log_scalar['accuracy'] * 100:.2f}%")
        print( 'IoU (%)      :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Precision (%):', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Recall (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))                                              
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)

    model.train()

    return np.nanmean(iou)

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
            gt = torch.tensor(labels[i]).long()         # H, W
            gt[gt==-1] = 0
            gt_ohe = F.one_hot(gt, num_classes = 21).permute(2, 0, 1).cuda()  # 21, H, W
            logit = model.module.forward_cam(img)

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)            
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
            # import pdb; pdb.set_trace()
            # masks.append(mask.cpu().numpy())
            # corrects.append(correct.cpu().numpy())
            

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
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj

        precision = gtjresj / (gtj + 1e-10)
        recall = gtjresj / (resj + 1e-10)
        iou = gtjresj / denominator
        acc_total = gtjresj.sum() / confusion.sum()

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

def validate3(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, get_dict=True)

    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]
    model.eval()

    with torch.no_grad():
        preds = []
        cams = []
        uncrts = []
        masks = []
        cams_1, cams_2, cams_3, cams_4, cams_5, cams_6 = [], [], [], [], [], []
        gts_1, gts_2, gts_3, gts_4, gts_5, gts_6 = [], [], [], [], [], []
        n1, n2, n3, n4, n5, n6 = 0, 0, 0, 0, 0, 0
        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            label = label.cuda(non_blocking=True)[0,:,None,None]
            gt = torch.tensor(labels[i]).long()
            gt[gt==-1] = 0
            gt = F.one_hot(gt, num_classes=21).permute(2, 0, 1).cuda()  # Nclass, H, W
            logit = model.module.forward_cam(img)

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)            
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
            cam = F.one_hot(cam, num_classes=21).permute(2, 0, 1).cuda()

            # assert gt.min() == 0 and cam.min() == 0
            # masks by confidence
            mask = torch.empty(size=(6, max_probs.shape[-2], max_probs.shape[-1]))
            mask[0, :, :] = max_probs.lt(0.4)
            mask[1, :, :] = torch.logical_and(max_probs.ge(0.40), max_probs.lt(0.60))
            mask[2, :, :] = torch.logical_and(max_probs.ge(0.60), max_probs.lt(0.80))
            mask[3, :, :] = torch.logical_and(max_probs.ge(0.80), max_probs.lt(0.95))
            mask[4, :, :] = torch.logical_and(max_probs.ge(0.95), max_probs.lt(0.99))
            mask[5, :, :] = max_probs.ge(0.99)
            mask = mask.long().cuda()
            
            import pdb; pdb.set_trace()

            cams_1.append((cam * mask[0]).cpu().numpy())
            cams_2.append((cam * mask[1]).cpu().numpy())
            cams_3.append((cam * mask[2]).cpu().numpy())
            cams_4.append((cam * mask[3]).cpu().numpy())
            cams_5.append((cam * mask[4]).cpu().numpy())
            cams_6.append((cam * mask[5]).cpu().numpy())

            gts_1.append((gt * mask[0]).cpu().numpy())
            gts_2.append((gt * mask[1]).cpu().numpy())
            gts_3.append((gt * mask[2]).cpu().numpy())
            gts_4.append((gt * mask[3]).cpu().numpy())
            gts_5.append((gt * mask[4]).cpu().numpy())
            gts_6.append((gt * mask[5]).cpu().numpy())

            cam = cam.cpu().numpy()
            cams.append(cam)
            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())
            masks.append(mask)
            n1 += mask[0].sum()
            n2 += mask[1].sum()
            n3 += mask[2].sum()
            n4 += mask[3].sum()
            n5 += mask[4].sum()
            n6 += mask[5].sum()

        '''
        Calculate Metrics
        cams : 1464 * (21, H, W), value: prediction class
        gts: 1464 * (21, H, W)
        confusion: (21, 21)
        masks : 1464 * (6, H, W)
        cams_by_conf: 1464 * (6, H, W)
        conf 1, 2, 3, 4, 5, 6에 대해서 mask를 만들어
        그리고 cams list와 labels list를 순회하면서 6개의 마스크를 만들어서 결과로 6개의 confusion을 만들어
        Accuracy : TP + TN / TP + TN + FP + FN

        '''
        print("n1  n2  n3  n4  n5  n6")
        print(n1.item(), n2.item(), n3.item(), n4.item(), n5.item(), n6.item())

        

        acc_by_classes_1, acc_total_1 = calc_acc_byclass(cams_1, gts_1)
        acc_by_classes_2, acc_total_2 = calc_acc_byclass(cams_2, gts_2)
        acc_by_classes_3, acc_total_3 = calc_acc_byclass(cams_3, gts_3)
        acc_by_classes_4, acc_total_4 = calc_acc_byclass(cams_4, gts_4)
        acc_by_classes_5, acc_total_5 = calc_acc_byclass(cams_5, gts_5)
        acc_by_classes_6, acc_total_6 = calc_acc_byclass(cams_6, gts_6)
        
        precision, recall, iou, acc_total, confusion = calc_metrics_val(cams, labels)
        # Logging Values

        class_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', \
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        # import pdb; pdb.set_trace()
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {'miou': np.nanmean(iou),
                      'mprecision': np.nanmean(precision),
                      'mrecall': np.nanmean(recall),
                      'accuracy': acc_total,
                      'acc_1': acc_total_1,
                      'acc_2': acc_total_2,
                      'acc_3': acc_total_3,
                      'acc_4': acc_total_4,
                      'acc_5': acc_total_5,
                      'acc_6': acc_total_6,
                      }

        

        '''
        
        import pdb; pdb.set_trace()
        import code; code.interact(local=vars())
        import torch; import torch.nn.functional as F
        x = torch.randint(20, (10, 10))
        
        '''

        
        # idx : 0-20
        for idx, c in enumerate(class_list):
            log_scalar[str('acc_' + c + '_1')] = acc_by_classes_1[idx]
            log_scalar[str('acc_' + c + '_2')] = acc_by_classes_2[idx]
            log_scalar[str('acc_' + c + '_3')] = acc_by_classes_3[idx]
            log_scalar[str('acc_' + c + '_4')] = acc_by_classes_4[idx]
            log_scalar[str('acc_' + c + '_5')] = acc_by_classes_5[idx]
            log_scalar[str('acc_' + c + '_6')] = acc_by_classes_6[idx]


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
        print(f"mIoU         : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrecision   : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall      : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy     : {log_scalar['accuracy'] * 100:.2f}%")
        print( 'IoU (%)      :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Precision (%):', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Recall (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))                                              
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            # wandb.log({'val/pix_counts': wandb.Histogram(np_histogram=np_hist)}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)

    model.train()
    return np.nanmean(iou)

def validate2_old(args, model, data_loader, iter, tag='val'):
    
    timg = {}
    idx2class = get_categories(args.num_sample, get_dict=True)

    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../') # Temporary 
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]

    model.eval()

    with torch.no_grad():
        preds = []
        cams = []
        # cams_by_conf = []
        uncrts = []
        masks = []
        # hists = []
        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            label = label.cuda(non_blocking=True)[0,:,None,None]

            logit = model.module.forward_cam(img)

            # Available only batch size 1
            logit = F.softmax(logit, dim=1)
            logit = F.interpolate(logit, labels[i].shape[-2:], mode='bilinear', align_corners=False)            
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
            cam = cam[0]

            # masks by confidence 
            mask = torch.empty(size=(6, max_probs.shape[-2], max_probs.shape[-1]))
            mask[0, :, :] = max_probs.lt(0.4)
            mask[1, :, :] = (max_probs.ge(0.4), max_probs.lt(0.6))
            mask[2, :, :] = (max_probs.ge(0.6), max_probs.lt(0.8))
            mask[3, :, :] = (max_probs.ge(0.8), max_probs.lt(0.95))
            mask[4, :, :] = (max_probs.ge(0.95), max_probs.lt(0.99))
            mask[5, :, :]= max_probs.ge(0.99)

            cams.append(cam.cpu().numpy().copy())
            # cams_by_conf.append(cam_by_conf.cpu().numpy())
            uncrts.append(max_probs[0].lt(args.p_cutoff).cpu().numpy())
            masks.append(mask)
    
            # hist = np.histogram(cam.detach().cpu().numpy(), bins=[0, 0.4, 0.6, 0.8, 0.95, 0.99, 1])
            # hists.append(hist)
    
        '''
        Calculate Metrics
        cams : 1464 * (H, W), value: prediction class
        labels: 1464 * (H, W)
        confusion: (21, 21)
        masks : 1464 * (6, H, W)
        cams_by_conf: 1464 * (6, H, W)
        '''
        # tmp = [np.zeros_like(hist[0]), hist[1]]
        # for i in range(len(hists)):
        #     tmp[0] += hists[i][0]
        # np_hist = tuple(tmp)
        # print("Histogram: ")
        # print(np_hist)

        # confidence range별 ACC

        # import pdb; pdb.set_trace()

        crt_1, crt_2, crt_3, crt_4, crt_5, crt_6 = 0, 0, 0, 0, 0, 0
        n1, n2, n3, n4, n5, n6 = 0, 0, 0, 0, 0, 0
        for i in range(len(labels)):    # for all data
            mask = masks[i].numpy()
            correct = (cams[i] == labels[i])
            correct_with_conf = correct * mask
            crt_1 += correct_with_conf[0].sum()
            crt_2 += correct_with_conf[1].sum()
            crt_3 += correct_with_conf[2].sum()
            crt_4 += correct_with_conf[3].sum()
            crt_5 += correct_with_conf[4].sum()
            crt_6 += correct_with_conf[5].sum()
            n1 += mask[0].sum()
            n2 += mask[1].sum()
            n3 += mask[2].sum()
            n4 += mask[3].sum()
            n5 += mask[4].sum()
            n6 += mask[5].sum()
            correct
        
        # import pdb; pdb.set_trace()

        acc_1 = crt_1.sum() / (n1.sum() + 1e-10)
        acc_2 = crt_2 / (n2 + 1e-10)
        acc_3 = crt_3 / (n3 + 1e-10)
        acc_4 = crt_4 / (n4 + 1e-10)
        acc_5 = crt_5 / (n5 + 1e-10)
        acc_6 = crt_6 / (n6 + 1e-10)
        

            # i = 0; j = 2; ((cams[i][j] == labels[i]) * masks[i][j].numpy()).sum() / masks[i][j].numpy().sum()

        

        confusion = calc_semantic_segmentation_confusion(cams, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj

        precision = gtjresj / (gtj + 1e-10)
        recall = gtjresj / (resj + 1e-10)
        iou = gtjresj / denominator
        acc = gtjresj.sum() / confusion.sum()




        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {'miou': np.nanmean(iou),
                      'mprecision': np.nanmean(precision),
                      'mrecall': np.nanmean(recall),
                      'accuracy': acc,
                      'acc_1': acc_1,
                      'acc_2': acc_2,
                      'acc_3': acc_3,
                      'acc_4': acc_4,
                      'acc_5': acc_5,
                      'acc_6': acc_6
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
        print(f"mIoU         : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrecision   : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall      : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy     : {log_scalar['accuracy'] * 100:.2f}%")
        print( 'IoU (%)      :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Precision (%):', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print( 'Recall (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))                                              
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            # wandb.log({'val/pix_counts': wandb.Histogram(np_histogram=np_hist)}, step=iter)
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