from re import I
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ImageDataset, ClassificationDataset, ClassificationDatasetWithSaliency, \
COCOClassificationDataset, COCOClassificationDatasetWithSaliency, TorchvisionNormalize
import os.path as osp

from util import imutils

def get_dataloader(args):

    if args.mode != 'base':
        ssl_params = {
            'aug_type'      : args.aug_type, 
            'n_strong_augs' : args.n_strong_augs,
            'patch_k'       : args.patch_k, 
            'use_geom_augs' : args.use_geom_augs
            }
    else:
        ssl_params = {}
    
    # VOC12 dataset
    if args.dataset == 'voc12':
        if args.network_type in ['cls', 'seam']:
            train_dataset = ClassificationDataset(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = args.data_root,
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )
            
        elif args.network_type in ['eps', 'contrast']:
            train_dataset = ClassificationDatasetWithSaliency(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = args.data_root,
                saliency_root       = args.saliency_root,
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )
        else:
            raise Exception("No appropriate train type")
        
        val_dataset = ClassificationDataset(
            dataset             = args.dataset,
            img_id_list_file    = args.val_list,
            img_root            = args.data_root,
            tv_transform        = transforms.Compose([
                                    np.asarray,
                                    TorchvisionNormalize(),
                                    imutils.HWC_to_CHW,
                                    torch.from_numpy
                                    ])
                )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True, 
            drop_last=True
        )

        # Currently avaliable batch size 1
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=True
            )

        # ######## Unlabeled dataset ########
        # if args.train_ulb_list:
        #     train_ulb_dataset = ImageDataset(
        #         dataset             = args.ulb_dataset,
        #         img_id_list_file    = args.train_ulb_list,
        #         img_root            = args.ulb_data_root,
        #         crop_size           = args.crop_size,
        #         resize_size         = args.resize_size,
        #         aug_type            = args.ulb_aug_type,
        #         n_strong_augs       = args.n_strong_augs
        #         )
        #     train_ulb_loader = DataLoader(
        #         train_ulb_dataset, 
        #         batch_size=int(args.batch_size*args.mu), 
        #         shuffle=True,
        #         num_workers=args.num_workers, 
        #         pin_memory=True, 
        #         drop_last=True
        #         )
        # else: 
        #     train_ulb_loader = None
    
    # COCO dataset
    elif args.dataset == 'coco':
        if args.network_type in ['cls', 'seam']:
            # train_dataset = COCOClassificationDataset(
            #     image_dir   = osp.join(args.data_root, 'images/train2014/'),
            #     anno_path   = osp.join(args.data_root, 'annotations/instances_train2014.json'),
            #     labels_path = 'data/coco/train_labels.npy',
            #     hor_flip    = True,
            #     crop_size   = args.crop_size,
            #     resize_size = args.resize_size,
            #     **ssl_params
            # )
            train_dataset = ClassificationDataset(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = osp.join(args.data_root, 'images/train2014/'),
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )

        elif args.network_type in ['eps', 'contrast']:
            # train_dataset = COCOClassificationDatasetWithSaliency(
            #     image_dir   = osp.join(args.data_root, 'images/train2014/'),
            #     anno_path   = osp.join(args.data_root, 'annotations/instances_train2014.json'),
            #     sal_path    = osp.join(args.data_root, 'SALImages'),
            #     labels_path = 'data/coco/train_labels.npy',
            #     hor_flip    = True,
            #     crop_size   = args.crop_size,
            #     resize_size = args.resize_size,
            #     **ssl_params
            # )
            train_dataset = ClassificationDatasetWithSaliency(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = osp.join(args.data_root, 'images/train2014/'),
                saliency_root       = args.saliency_root,
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )

            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True, 
                drop_last=True
            )

            val_dataset = COCOClassificationDataset(
                image_dir   = osp.join(args.data_root, 'images/val2014/'),
                anno_path   = osp.join(args.data_root, 'annotations/instances_val2014.json'),
                labels_path = 'data/coco/val_labels.npy',
                hor_flip    = False,
                tv_transform        = transforms.Compose([
                                        np.asarray,
                                        TorchvisionNormalize(),
                                        imutils.HWC_to_CHW,
                                        torch.from_numpy
                                        ])
            )
            # val_dataset = VOCClassificationDataset(
            #     dataset             = args.dataset,
            #     img_id_list_file    = args.val_list,
            #     img_root            = args.data_root,
            #     tv_transform        = transforms.Compose([
            #                             np.asarray,
            #                             TorchvisionNormalize(),
            #                             imutils.HWC_to_CHW,
            #                             torch.from_numpy
            #                             ])
            #         )

            # Currently avaliable batch size 1
            val_loader = DataLoader(
                val_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=args.num_workers, 
                pin_memory=True, 
                drop_last=True
                )

    return train_loader, val_loader
