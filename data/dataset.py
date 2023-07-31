import random
import os.path
import PIL
from PIL import Image
import numpy as np
import pdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.datasets as dset

from torchvision import transforms
import torchvision.transforms.functional as vision_tf

from util import imutils
from util.imutils import RandomResizeLong, random_crop_with_saliency, HWC_to_CHW
from data.augmentation.randaugment import RandAugment
from module.helper import merge_patches_np, patch_with_tr

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_utils import get_coco
from chainercv import utils

import json
import imageio


def get_categories(num_classes=None, bg_last=False, get_dict=False):
    # VOC
    if num_classes == 21:
        categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # COCO
    elif num_classes == 81:
        categories =  ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                       'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                       'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                       'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                       'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']   
    if bg_last:
        categories.pop(0)
        categories.append('background')
    if get_dict:
        return {i:c for i, c in enumerate(categories)}
    else:
        return categories


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'data/{dataset}/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_saliency_path(img_name, saliency_root='SALImages'):
    return os.path.join(saliency_root, img_name + '.png')


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    Performs Weak or Strong Augmentations.
    """
    def __init__(self, dataset, img_id_list_file, img_root, tv_transform=None,
                 crop_size=448, resize_size=(448, 768), 
                 aug_type=None, use_geom_augs=False, n_strong_augs=5, patch_k=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root

        # Dataset use (1).self.transform(Torchvision Transformations)
        self.tv_transform = tv_transform
        # or (2).Weak & Strong Augmentations
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.aug_type = aug_type
        self.patch_k = patch_k
        self.use_geom_augs = use_geom_augs

        self.resizelong = RandomResizeLong(resize_size[0], resize_size[1])
        self.colorjitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.img_normal = TorchvisionNormalize()

        if self.aug_type == 'strong': ###
            blur_kernel_size = int(random.random() * 4.95)
            blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
            self.blur = transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2.0)) # non-geometric transformations
            self.randaug = RandAugment(self.use_geom_augs, n_strong_augs, 5)

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        # Use torchvision Transforms
        if self.tv_transform:
            return img_id, self.tv_transform(img)

        # Use weak | strong augmentations
        else:
            ### Image 1. Weak Augmentation ###
            img_w, (weak_target_long, weak_hflip, weak_box) = self.__apply_transform(img,
                                                                                    get_transform=True, 
                                                                                    strong=False, 
                                                                                    target_long=None,
                                                                                    crop_size=self.crop_size, 
                                                                                    hflip=None, 
                                                                                    box=None)
            img_w = self.__totensor(img_w)    # (448, 448)
            if not self.aug_type:
                return img_id, img_w

            ### Image 2. Strong augmetation (for consistency regularization) ###
            elif self.aug_type == 'strong':
                img_s, tr_ops, ra_ops = self.__apply_transform(img,
                                                                get_transform=True,
                                                                strong=True,
                                                                target_long=weak_target_long,
                                                                crop_size=self.crop_size, 
                                                                hflip=weak_hflip,
                                                                box=weak_box,
                                                                patch_k = self.patch_k
                                                                ) 

                img_s = self.__totensor(img_s)
                return img_id, img_w, img_s, ra_ops

            else:
                raise Exception('No appropriate Augmentation type')
    
    def __apply_transform(self, img, get_transform=False, strong=False, target_long=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # randomly resize
        if target_long is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
        
        # Colorjitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)
        img, _, tr_box = random_crop_with_saliency(imgarr=img, 
                                                   sal=None,
                                                   crop_size=crop_size,
                                                   get_box=True, 
                                                   box=box)

        # Strong Augmentation
        if strong:
            if patch_k:
                patches, ra_ops = patch_with_tr(img, patch_k, self.randaug)
                img = merge_patches_np(patches, patch_k)
            
            else:
                img = Image.fromarray(img)
                img = self.blur(img)
                img, ra_ops = self.randaug(img)
                img = np.asarray(img)

        # normalize
        img = self.img_normal(img)

        if get_transform: ###
            if strong:
                return img, (target_long, hflip, tr_box), ra_ops
            else:
                return img, (target_long, hflip, tr_box)
        else:
            return img
    
    def __totensor(self, img):
        # Image
        img = HWC_to_CHW(img)
        img = torch.from_numpy(img)

        return img

class ClassificationDataset(ImageDataset):
    """
    for SEAM, SIPE
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, self.dataset)

    def __getitem__(self, idx):
        label = torch.from_numpy(self.label_list[idx])
        return super().__getitem__(idx) + (label,)


class ClassificationDatasetWithSaliency(ImageDataset):
    """
    for EPS, PPC
    """
    def __init__(self, saliency_root=None, **kwargs):
        super().__init__(**kwargs)
        # self.tv_transform is useless in ClassificationDatasetWithSaliency
        self.saliency_root = saliency_root
        self.label_list = load_img_label_list_from_npy(self.img_id_list, self.dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        saliency = PIL.Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")

        label = torch.from_numpy(self.label_list[idx])

        ### Image 1 ###
        img_w, saliency1, (weak_target_long, weak_hflip, weak_box) = self.__apply_transform_with_sal(img, 
                                                                                                    saliency, 
                                                                                                    get_transform=True,
                                                                                                    strong=False,
                                                                                                    target_long=None,
                                                                                                    crop_size=self.crop_size,
                                                                                                    hflip=None,
                                                                                                    box=None)
        
        img_w, saliency1 = self.__totensor(img_w, saliency1)
        if not self.aug_type:
            return img_id, img_w, saliency1, label

        ### Image 2: Strong augmetation (for MT, FixMatch)
        elif self.aug_type == 'strong':
            ### TODO: mask transform, return aug information
            img_s, saliency2, tr_ops, ra_ops = self.__apply_transform_with_sal(img, 
                                                                                  saliency, 
                                                                                  get_transform=True, 
                                                                                  strong=True, 
                                                                                  target_long=weak_target_long,
                                                                                  crop_size=self.crop_size,
                                                                                  hflip=weak_hflip,
                                                                                  box=weak_box,
                                                                                  patch_k=self.patch_k
                                                                                  )
            # pdb.set_trace()
            img_s = self.__totensor(img_s)
            return img_id, img_w, saliency1, img_s, ra_ops, label
        
        else:
            raise Exception('No appropriate Augmentation type')

    def __apply_transform_with_sal(self, img, sal, get_transform=False, strong=False, target_long=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # Randomly resize
        if target_long is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)
        sal = self.resizelong(sal, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
            sal = vision_tf.hflip(sal)
        
        # Color jitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)   # H, W, 3
        sal = np.asarray(sal)   # H, W, 3
        img, sal, tr_box = random_crop_with_saliency(imgarr=img,
                                                    sal=sal,
                                                    crop_size=crop_size,
                                                    get_box=True, 
                                                    box=box)

        # Strong Augmentation
        if strong:
            if patch_k:
                patches, ra_ops = patch_with_tr(img, patch_k, self.randaug)
                img = merge_patches_np(patches, patch_k)
                img = np.asarray(img)

            else:
                img = Image.fromarray(img)
                img = self.blur(img)
                img, ra_ops = self.randaug(img)
                img = np.asarray(img)
            
        # Normalize
        img = self.img_normal(img)
        sal = sal / 255.

        if get_transform:
            if strong:
                return img, sal, (target_long, hflip, tr_box), ra_ops
            else:
                return img, sal, (target_long, hflip, tr_box)
        else:
            return img, sal
    
    def __totensor(self, img, mask=None):
        # Permute to C, H, W
        img = HWC_to_CHW(img)

        # Make torch tensor
        img = torch.from_numpy(img)

        if mask is not None:
            mask = HWC_to_CHW(mask)
            mask = torch.from_numpy(mask)
            mask = torch.mean(mask, dim=0, keepdim=True)
            return img, mask
        else:
            return img
    
# COCO
category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class COCOClassificationDataset(Dataset):
    def __init__(self, image_dir, anno_path, labels_path=None, tv_transform=None,
                 resize_size=(256, 448), rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method='random', to_torch=True,
                 aug_type=None, use_geom_augs=False, n_strong_augs=5, patch_k=None
                 ):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.labels_path = labels_path
        self.category_map = category_map

        self.resizelong = RandomResizeLong(resize_size[0], resize_size[1])
        self.resize_size = resize_size
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.labels = []
        
        # added for W-S transforms
        self.aug_type = aug_type
        self.patch_k = patch_k
        self.use_geom_augs = use_geom_augs
        self.colorjitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.tv_transform = tv_transform

        if self.aug_type == 'strong': ###
            blur_kernel_size = int(random.random() * 4.95)
            blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
            self.blur = transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2.0)) # non-geometric transformations
            self.randaug = RandAugment(self.use_geom_augs, n_strong_augs, 5)

        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path, allow_pickle=True).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in range(l):
                item = self.coco[i]
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)
        
    def __len__(self):
        return len(self.coco)
    
    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num
        return label

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)
    
    def __apply_transform(self, img, get_transform=False, strong=False, resize_size=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # randomly resize
        if resize_size is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
        
        # Colorjitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)
        img, _, tr_box = random_crop_with_saliency(imgarr=img, 
                                                   sal=None,
                                                   crop_size=crop_size,
                                                   get_box=True, 
                                                   box=box)
        # Strong Augmentation
        if strong:
            if patch_k:
                img = Image.fromarray(img)
                img = self.blur(img)
                img = np.asarray(img)
                patches, ra_ops = patch_with_tr(img, patch_k, self.randaug)
                img = merge_patches_np(patches, patch_k)
            
            else:
                img = Image.fromarray(img)
                img = self.blur(img)
                img, ra_ops = self.randaug(img)
                img = np.asarray(img)

        # normalize
        img = self.img_normal(img)

        if get_transform:
            if strong:
                return img, (resize_size, hflip, tr_box), ra_ops
            else:
                return img, (resize_size, hflip, tr_box)
        else:
            return img
    
    def __totensor(self, img):
        # Image
        img = HWC_to_CHW(img)
        img = torch.from_numpy(img)

        return img

    def __getitem__(self, index):
        name = self.coco.ids[index]
        name = self.coco.coco.loadImgs(name)[0]["file_name"].split('.')[0]
        # img = np.asarray(self.coco[index][0])
        img = self.coco[index][0]
        label = self.labels[index]
        
        if self.tv_transform:   # if validation
            return name, self.tv_transform(img), label 
        else:
            # Use weak | strong augmentations
            ### Image 1. Weak Augmentation ###
            img_w, (weak_resize_size, weak_hflip, weak_box) = self.__apply_transform(img,
                                                                                    get_transform=True, 
                                                                                    strong=False, 
                                                                                    resize_size=None,
                                                                                    crop_size=self.crop_size, 
                                                                                    hflip=None, 
                                                                                    box=None
                                                                                    )
            img_w = self.__totensor(img_w)    # (448, 448)

            ### Image 2. Strong augmetation (for consistency regularization) ###
            img_s, _, ra_ops = self.__apply_transform(img,
                                                    get_transform=True,
                                                    strong=True,
                                                    resize_size=weak_resize_size,
                                                    crop_size=self.crop_size, 
                                                    hflip=weak_hflip,
                                                    box=weak_box,
                                                    patch_k = self.patch_k
                                                    ) 

            img_s = self.__totensor(img_s)

            return name, img_w, img_s, ra_ops, label

class COCOClassificationDatasetWithSaliency(Dataset):
    def __init__(self, image_dir, anno_path, labels_path=None, sal_path=None,
                 resize_size=(256, 448), rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method='random', to_torch=True,
                 aug_type=None, use_geom_augs=False, n_strong_augs=5, patch_k=None
                 ):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.labels_path = labels_path
        self.category_map = category_map
        self.sal_path = sal_path

        self.resizelong = RandomResizeLong(resize_size[0], resize_size[1])
        self.resize_size = resize_size
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.labels = []
        
        # added for W-S transforms
        self.aug_type = aug_type
        self.patch_k = patch_k
        self.use_geom_augs = use_geom_augs
        self.colorjitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        if self.aug_type == 'strong': ###
            blur_kernel_size = int(random.random() * 4.95)
            blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
            self.blur = transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2.0)) # non-geometric transformations
            self.randaug = RandAugment(self.use_geom_augs, n_strong_augs, 5)

        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path, allow_pickle=True).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in range(l):
                item = self.coco[i]
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)

    def __len__(self):
        return len(self.coco)
    
    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num
        return label

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)
    
    def __apply_transform_with_sal(self, img, sal, get_transform=False, strong=False, resize_size=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # Randomly resize
        if resize_size is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)
        sal = self.resizelong(sal, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
            sal = vision_tf.hflip(sal)
        
        # Color jitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)   # H, W, 3
        sal = np.asarray(sal)   # H, W, 3
        img, sal, tr_box = random_crop_with_saliency(imgarr=img,
                                                    sal=sal,
                                                    crop_size=crop_size,
                                                    get_box=True, 
                                                    box=box)

        # Strong Augmentation
        if strong:
            if patch_k:
                patches, ra_ops = patch_with_tr(img, patch_k, self.randaug)
                img = merge_patches_np(patches, patch_k)
                img = np.asarray(img)

            else:
                img = Image.fromarray(img)
                img = self.blur(img)
                img, ra_ops = self.randaug(img)
                img = np.asarray(img)
            
        # Normalize
        img = self.img_normal(img)
        sal = sal / 255.

        if get_transform:
            if strong:
                return img, sal, (resize_size, hflip, tr_box), ra_ops
            else:
                return img, sal, (resize_size, hflip, tr_box)
        else:
            return img, sal
    
    def __totensor(self, img, mask=None):
        # Permute to C, H, W
        img = HWC_to_CHW(img)
        mask = HWC_to_CHW(mask)
        # Make torch tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask = torch.mean(mask, dim=0, keepdim=True)

        return img, mask

    def __getitem__(self, index):
        name = self.coco.ids[index]
        name = self.coco.coco.loadImgs(name)[0]["file_name"].split('.')[0]
        # img = np.asarray(self.coco[index][0])
        img = self.coco[index][0]
        sal = PIL.Image.open(get_saliency_path(name, self.sal_path)).convert("RGB")
        label = self.labels[index]
 
        # Use weak | strong augmentations
        ### Image 1. Weak Augmentation ###
        img_w, sal, (weak_resize_size, weak_hflip, weak_box) = self.__apply_transform_with_sal(img,
                                                                                                sal,
                                                                                                get_transform=True, 
                                                                                                strong=False, 
                                                                                                resize_size=None,
                                                                                                crop_size=self.crop_size, 
                                                                                                hflip=None, 
                                                                                                box=None
                                                                                                )
        img_w, sal = self.__totensor(img_w, sal)    # (448, 448)

        if not self.aug_type:
            return name, img_w, sal, label
        
        ### Image 2: Strong augmetation (for MT, FixMatch)
        elif self.aug_type == 'strong':

            ### Image 2. Strong augmetation (for consistency regularization) ###
            img_s, _, ra_ops = self.__apply_transform_with_sal(img,
                                                                sal,
                                                                get_transform=True,
                                                                strong=True,
                                                                resize_size=weak_resize_size,
                                                                crop_size=self.crop_size, 
                                                                hflip=weak_hflip,
                                                                box=weak_box,
                                                                patch_k = self.patch_k
                                                                ) 

            img_s = self.__totensor(img_s, None)

            return name, img_w, sal, img_s, ra_ops, label
        
        else:
            raise Exception('No appropriate Augmentation type')


def _rgb2id(color):
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


class COCOSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for `MS COCO`_.

    Semantic segmentations are generated from panoptic segmentations
    as done in the `official toolkit`_.

    .. _`MS COCO`: http://cocodataset.org/#home

    .. _`official toolkit`: https://github.com/cocodataset/panopticapi/
        blob/master/converters/panoptic2semantic_segmentation.py

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"

    """

    def __init__(self, data_dir='auto', split='train'):
        super(COCOSemanticSegmentationDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = get_coco(split, split, '2014', 'instances')

        self.img_root = os.path.join(
            data_dir, 'images', '{}{}'.format(split, 2014))

        self.label_root = os.path.join(
            data_dir, 'annotations', 'instances_{}{}'.format(split, 2014))
        anno_path = os.path.join(
            data_dir, 'annotations',
            'instances_{}{}.json'.format(split, 2014))

        self.data_dir = data_dir
        annos = json.load(open(anno_path, 'r'))
        self.annos = annos
        pdb.set_trace()
        self.cat_ids = [cat['id'] for cat in annos['categories']]
        self.img_paths = [ann['file_name'][:-4] + '.jpg' for ann in annos['annotations']]

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

        self.keys = ('img', 'label')

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        img_path = os.path.join(
            self.img_root, self.img_paths[i])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        return img

    def _get_label(self, i):
        # https://github.com/cocodataset/panopticapi/blob/master/converters/
        # panoptic2semantic_segmentation.py#L58
        anno = self.annos['annotations'][i]
        label_path = os.path.join(self.label_root, anno['file_name'])
        rgb_id_map = utils.read_image(
            label_path,
            dtype=np.uint32, color=True)
        id_map = _rgb2id(rgb_id_map)
        label = -1 * np.ones_like(id_map, dtype=np.int32)
        for inst in anno['segments_info']:
            mask = id_map == inst['id']
            label[mask] = self.cat_ids.index(inst['category_id'])
        return label
    
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, anno_path, masks_path, crop_size, rescale=None, img_normal=TorchvisionNormalize(), 
                hor_flip=False, crop_method='random', read_ir_label=False):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.masks_path = masks_path
        self.category_map = category_map

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.read_ir_label = read_ir_label

        self.ids2name = {}
        for ids in self.coco.ids:
            self.ids2name[ids] = self.coco.coco.loadImgs(ids)[0]["file_name"].split('.')[0]
    
    def __getitem__(self, index):
        ids = self.coco.ids[index]
        name = self.ids2name[ids]

        img = np.asarray(self.coco[index][0])
        if self.read_ir_label:
          label = imageio.imread(os.path.join(self.masks_path, name+'.png'))
        else:
            label = imageio.imread(os.path.join(self.masks_path, str(ids) + '.png'))

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label':label}
    
    def get_label_by_id(self,ids):
        label = imageio.imread(os.path.join(self.masks_path, str(ids) + '.png'))
        return label
    
    def get_label_by_name(self,name):
        # COCO_val2014_000000159977.jpg
        label = imageio.imread(os.path.join(self.masks_path, str(int(name.split('.')[0].split('_')[-1])) + '.png'))
        return label

    def __len__(self):
        return len(self.coco)
