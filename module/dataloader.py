from re import I
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ImageDataset, ClassificationDataset, ClassificationDatasetOnMemory, ClassificationDatasetWithSaliency, ClassificationDatasetWithSaliencyOnMemory
from util import imutils
from util.imutils import Normalize


def get_dataloader(args):
    if not args.data_on_mem:
        CLS_DATASET = ClassificationDataset
        CLS_SAL_DATASET = ClassificationDatasetWithSaliency
    else:
        CLS_DATASET = ClassificationDatasetOnMemory
        CLS_SAL_DATASET = ClassificationDatasetWithSaliencyOnMemory

    if args.ssl:
        ssl_params = {'aug_type': args.ulb_aug_type, 'n_strong_aug': args.n_strong_aug}
    else:
        ssl_params = {}

    if args.network_type == 'cls':
        train_dataset = CLS_DATASET(
            dataset             = args.dataset,
            img_id_list_file    = args.train_list,
            img_root            = args.data_root,
            tv_transform        = transforms.Compose([
                                imutils.RandomResizeLong(args.resize_size[0], args.resize_size[1]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                np.asarray,
                                Normalize(),
                                imutils.RandomCrop(args.crop_size),
                                imutils.HWC_to_CHW,
                                torch.from_numpy
            ]))
    elif args.network_type in ['seam']:
        train_dataset = CLS_DATASET(
            dataset             = args.dataset,
            img_id_list_file    = args.train_list,
            img_root            = args.data_root,
            crop_size           = args.crop_size,
            resize_size         = args.resize_size,
            **ssl_params
        )
    elif args.network_type in ['eps', 'contrast']:
        train_dataset = CLS_SAL_DATASET(
            dataset             = args.dataset,
            img_id_list_file    = args.train_list,
            img_root            = args.data_root,
            saliency_root       = args.saliency_root,
            crop_size           = args.crop_size,
            resize_size         = args.resize_size,
            **ssl_params
        )
    # elif args.network_type == 'eps_seam' or args.network_type == 'eps_seam_with_PCM':
    #     train_dataset = CLS_SAL_DATASET(
    #         dataset             = args.dataset,
    #         img_id_list_file    = args.train_list,
    #         img_root            = args.data_root,
    #         saliency_root       = args.saliency_root,
    #         crop_size           = args.crop_size,
    #         resize_size         = args.resize_size,
    #         **ssl_params
    #     )
    else:
        raise Exception("No appropriate train type")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    try:
        val_dataset = CLS_DATASET(
            dataset             = args.dataset,
            img_id_list_file    = args.val_list,
            img_root            = args.data_root,
            tv_transform        = transforms.Compose([
                                    transforms.Resize((args.crop_size,args.crop_size)),
                                    np.asarray,
                                    Normalize(),
                                    # imutils.CenterCrop(args.crop_size),
                                    imutils.HWC_to_CHW,
                                    torch.from_numpy
            ]))
        
        # Currently avaliable batch size 1
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    except: # coco (no val labels in cls_labels.npy)
        print('No validation label list found. Train without validation dataloader.')
        val_loader = None

    ### Unlabeled dataset ###
    if args.train_ulb_list:
        train_ulb_dataset = ImageDataset(
            dataset             = args.ulb_dataset,
            img_id_list_file    = args.train_ulb_list,
            img_root            = args.ulb_data_root,
            crop_size           = args.crop_size,
            resize_size         = args.resize_size,
            aug_type            = args.ulb_aug_type,
            n_strong_aug        = args.n_strong_aug
        )
        train_ulb_loader = DataLoader(train_ulb_dataset, batch_size=int(args.batch_size*args.mu), shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else: 
        train_ulb_loader = None
    
    return train_loader, train_ulb_loader, val_loader
