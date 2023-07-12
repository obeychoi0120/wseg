import os
import shutil
import torch
import argparse
from torch.backends import cudnn
import wandb

import numpy as np
import random

from util import pyutils 

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import train_ppc_ssl, train_ppc

dataset_list = ['voc12', 'coco']

def get_arguments():
    parser = argparse.ArgumentParser()
    # Session
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val_only', action='store_true')
    parser.add_argument('--session', default='wsss', type=str)
    parser.add_argument('--use_wandb', action='store_true') ### Use wandb Logging
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--val_times', default=20, type=int)
    parser.add_argument('--seed', default=None, type=int)

    # Data
    parser.add_argument("--dataset", default='voc12', choices=dataset_list, type=str)
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--saliency_root', type=str)
    parser.add_argument('--train_list', default='data/voc12/train_aug_id.txt', type=str)
    parser.add_argument('--val_list', default='data/voc12/train_id.txt', type=str)
    parser.add_argument('--data_on_mem', action='store_true') ### Load dataset on RAM(need 20GB additional RAM)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--resize_size', default=(448, 768), type=int, nargs='*')
    parser.add_argument('--crop_size', default=448, type=int)        

    # Iteration & Optimizer
    parser.add_argument('--iter_size', default=2, type=int)
    parser.add_argument('--max_iters', default=10000, type=int)
    parser.add_argument('--max_epoches', default=None, type=int) # default=15
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--wt_dec', default=5e-4, type=float)

    # Network
    parser.add_argument('--network', default='network.resnet38_cls', type=str)
    parser.add_argument('--weights', required=True, type=str, default='pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
    
    # Hyperparameters for EPS
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)

    ### Semi-supervised Learning ###
    parser.add_argument('--mode', required=True, choices=['base', 'ssl'])
    parser.add_argument('--ssl_type', nargs='+', default=[3], type=int) # 1: MT, 2: pixel-wise MT, 3: fixmatch
    parser.add_argument("--ulb_dataset", default=None, choices=dataset_list, type=str)
    parser.add_argument('--ulb_data_root', default=None, type=str)
    parser.add_argument('--ulb_saliency_root', default=None, type=str)
    parser.add_argument('--train_ulb_list', default='', type=str)
    parser.add_argument('--mu', default=1.0, type=float) # ratio of ulb / lb data
        
    parser.add_argument('--use_ema',action='store_true') 
    parser.add_argument('--ema_m', default=0.999, type=float) 
    parser.add_argument('--mt_warmup', type=float, default=0.4) # mean teacher warmup
    parser.add_argument('--mt_lambda', default=50.0, type=float) # ratio of ssl loss
    parser.add_argument('--mt_p', default=0., type=float) # ratio of ssl loss
    parser.add_argument('--ssl_lambda', default=1.0, type=float) # ratio of ssl loss
    parser.add_argument('--p_cutoff', default=0.95, type=float)
    parser.add_argument('--th_scheduler', action='store_true') # Threshold Scheduler
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--soft_label', action='store_true') # hard label(Default) or soft label
    parser.add_argument('--cdc_lambda', default=1.0, type=float) # ratio of cdc loss
    parser.add_argument('--cdc_T', default=0.5, type=float) # Temperature of cdc loss
    parser.add_argument('--cdc_norm', action='store_true') # Normalize feature to calculate cdc loss
    parser.add_argument('--cdc_inter', action='store_true') # Calculate Inter-image pixel    
    
    ### Augmentations ###
    parser.add_argument('--ulb_aug_type', default='strong', type=str)   # None / weak / strong : 'aug_type'
    parser.add_argument('--n_strong_augs', required=True, type=int)     # number of RandAug
    parser.add_argument('--use_cutmix', action='store_true')            # Use CutMix
    parser.add_argument('--patch_k', default=None, type=int)
    parser.add_argument('--use_geom_augs', action='store_true')

    parser.add_argument('--bdry_size', default=0, type=int)
    parser.add_argument('--bdry_lambda', default=0.0, type=float)
    parser.add_argument('--bdry', action='store_true')

    parser.add_argument('--recon_lambda', default=0.0, type=float)
    parser.add_argument('--screg_lambda', default=0.0, type=float)
    args = parser.parse_args()

    # Dataset(Class Number)
    if args.dataset == 'voc12':
        args.num_sample = 21
    elif args.dataset == 'coco':
        args.num_sample = 81
    
    # Unlabeled Dataset
    if args.mode == 'ssl':
        if args.ulb_dataset is None:
            args.ulb_dataset = args.dataset
        if args.ulb_data_root is None:
            args.ulb_data_root = args.data_root
        if args.ulb_saliency_root is None:
            args.ulb_saliency_root = args.saliency_root

    # Network type
    if 'cls' in args.network:
        args.network_type = 'cls'
    elif 'seam' in args.network:
        args.network_type = 'seam'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    elif 'contrast' in args.network:
        args.network_type = 'contrast'
    else:
        raise Exception('No appropriate model type')
    
    return args


if __name__ == '__main__':
    # Get arguments
    args = get_arguments()

    # Set wandb Logger
    if args.use_wandb:
        wandb.init(name=args.session, project='WSSS')

    # Set Python Logger
    args.log_folder = os.path.join('train_log', args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    pyutils.Logger(os.path.join(args.log_folder, 'log_cls.log'))
    shutil.copyfile('./contrast_train.py', os.path.join(args.log_folder, 'contrast_train.py'))
    shutil.copyfile('./contrast_infer.py', os.path.join(args.log_folder, 'contrast_infer.py'))
    shutil.copyfile('./module/train.py', os.path.join(args.log_folder, 'train.py'))
    shutil.copyfile('./module/ssl.py', os.path.join(args.log_folder, 'ssl.py'))
    shutil.copyfile('./module/helper.py', os.path.join(args.log_folder, 'helper.py'))
    if 'seam' in args.network:
        shutil.copyfile('./network/resnet38_seam.py', os.path.join(args.log_folder, 'resnet38_seam.py'))
    elif 'contrast' in args.network:
        shutil.copyfile('./network/resnet38_contrast.py', os.path.join(args.log_folder, 'resnet38_contrast.py'))
    elif 'eps' in args.network:
        shutil.copyfile('./network/resnet38_eps.py', os.path.join(args.log_folder, 'resnet38_eps.py'))

    # Control Randomness
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load dataset (train_ulb_loader=None where args.ssl==False)
    train_loader, train_ulb_loader, val_loader = get_dataloader(args) ###

    # Max step
    num_data = len(open(args.train_list).read().splitlines())
    if args.max_epoches is None:
        args.max_epoches = int(args.max_iters * args.iter_size // (num_data // args.batch_size))
    # max_step = (num_data // args.batch_size) * args.max_epoches
    max_step = len(train_loader) * args.max_epoches
    
    # Load (ImageNet) Pretrained Model
    model = get_model(args)
    
    # Set optimizer
    optimizer = get_optimizer(args, model, max_step)
    
    # DP
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.train()
    
    if args.use_wandb:
        wandb.config.update(args)
    if args.mode == 'ssl':
        train_ppc_ssl(train_loader, train_ulb_loader, val_loader, model, optimizer, max_step, args)
    else:
        train_ppc(train_loader, val_loader, model, optimizer, max_step, args)
    print('Train Done.')