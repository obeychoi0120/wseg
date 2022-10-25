import random
import os.path
import PIL.Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as vision_tf

from util.imutils import RandomResizeLong,\
    random_crop_with_saliency, random_crop_with_saliency_pil, HWC_to_CHW, Normalize
from data.augmentation.randaugment import RandAugment

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
                 crop_size=224, resize_size=(256, 512), aug_type=None, n_strong_aug=5):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        # Dataset use (1).self.transform(Torchvision Transformations)
        self.tv_transform = tv_transform
        
        # or (2).Weak & Strong Augmentations
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.aug_type = aug_type

        self.resize = RandomResizeLong(resize_size[0], resize_size[1])
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = Normalize()

        if self.aug_type == 'strong': ###
            blur_kernel_size = int(random.random() * 4.95)
            blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
            self.strong_transforms = [transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2.0))] # non-geometric transformations
            self.randaug = RandAugment(n_strong_aug, 5) ###

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
            img1, weak_tr = self.__apply_transform(img, get_transform=True)
            img1 = self.__totensor(img1)

            if not self.aug_type:
                return img_id, img1

            ### Image 2. Weak augmentation ###
            elif self.aug_type == 'weak':
                img2, _  = self.__apply_transform(img, True, False, *weak_tr)
                img2 = self.__totensor(img2)

                return img_id, img1, img2, []

            ### Image 2. Strong augmetation (for consistency regularization) ###
            elif self.aug_type == 'strong':
                img2, _, strong_tr = self.__apply_transform(img, True, True, *weak_tr)
                img2 = self.__totensor(img2)
                
                return img_id, img1, img2, strong_tr

            else:
                raise Exception('No appropriate Augmentation type')
    
    def __apply_transform(self, img, get_transform=False, strong=False, target_size=None, hflip=None, tr_random_crop=None):
        # randomly resize
        if target_size is None:
            target_size = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resize(img, target_size)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        img = vision_tf.hflip(img)

        # Add color jitter
        img = self.color(img)

        # Random Crop
        img, _, tr_random_crop = random_crop_with_saliency_pil(img, crop_size=self.crop_size, get_transform=True, transforms=tr_random_crop)

        # Strong Augmentation
        if strong:
            for tr in self.strong_transforms:
                img = tr(img)
            img, strong_tr = self.randaug(img) # saliency2

        # Make numpy and normalize
        img = np.asarray(img, dtype=np.float32)
        img = self.normalize(img)

        if get_transform: ###
            if strong:
                return img, (target_size, hflip, tr_random_crop), strong_tr
            else:
                return img, (target_size, hflip, tr_random_crop)
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
        img1, saliency1, weak_tr = self.__apply_transform_with_mask(img, saliency, get_transform=True)
        img1, saliency1 = self.__totensor(img1, saliency1)

        if not self.aug_type:
            return img_id, img1, saliency1, label

        ### Image 2: Weak augmentation (for Mean Teacher)
        elif self.aug_type == 'weak':
            img2, saliency2, _  = self.__apply_transform_with_mask(img, saliency, True, False, *weak_tr)
            img2, saliency2 = self.__totensor(img2, saliency2)

            return img_id, img1, saliency1, img2, saliency2, [], label

        ### Image 2: Strong augmetation (for MT, FixMatch)
        elif self.aug_type == 'strong':
            ### TODO: mask transform, return aug information
            img2, saliency2, _, strong_tr = self.__apply_transform_with_mask(img, saliency, True, True, *weak_tr)
            img2, saliency2 = self.__totensor(img2, saliency2)
            
            return img_id, img1, saliency1, img2, saliency2, strong_tr, label
        else:
            raise Exception('No appropriate Augmentation type')

    def __apply_transform_with_mask(self, img, mask, get_transform=False, strong=False, target_size=None, hflip=None, tr_random_crop=None):
        # Randomly resize
        if target_size is None:
            target_size = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resize(img, target_size)
        mask = self.resize(mask, target_size)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        img = vision_tf.hflip(img)
        mask = vision_tf.hflip(mask)

        # Add color jitter
        img = self.color(img)

        # Random Crop
        img, mask, tr_random_crop = random_crop_with_saliency_pil(img, mask, self.crop_size, get_transform=True, transforms=tr_random_crop)

        # Strong Augmentation
        if strong:
            for tr in self.strong_transforms:
                img = tr(img)
                mask = tr(mask) if mask is not None else None ###
            img, strong_tr = self.randaug(img) # saliency2
            mask, _ = self.randaug(mask, trs=strong_tr, only_geometric=True)

        # Make numpy array
        img = np.asarray(img, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        # Normalize
        img = self.normalize(img)
        mask = mask / 255.
        
        if get_transform: ###
            if strong:
                return img, mask, (target_size, hflip, tr_random_crop), strong_tr
            else:
                return img, mask, (target_size, hflip, tr_random_crop)
        else:
            return img, mask
    
    def __totensor(self, img, mask=None):
        # Permute Channels
        img = HWC_to_CHW(img)
        mask = HWC_to_CHW(mask)
        # Make numpy array
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask = torch.mean(mask, dim=0, keepdim=True)
        
        return img, mask


class ClassificationDatasetOnMemory(ClassificationDataset):
    """
    Classification Dataset on Memory (base)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_list = []
        for img_id in tqdm(self.img_id_list, desc=f'Loading {len(self.img_id_list)} Images...'):
            img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
            self.img_list.append(img)

    def __getitem__(self, idx):    
        img_id = self.img_id_list[idx]
        img = self.img_list[idx]

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.label_list[idx])
        return img_id, img, label


class ClassificationDatasetWithSaliencyOnMemory(ClassificationDatasetWithSaliency):
    """
    Classification Dataset with saliency (load on Memory)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_list = []
        self.saliency_list = []
        for img_id in tqdm(self.img_id_list, desc=f'Loading {len(self.img_id_list)} Images...'):
            img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
            saliency = PIL.Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")
            self.img_list.append(img)
            self.saliency_list.append(saliency)
            
        self.saliency_list

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]

        img = self.img_list[idx]
        saliency = self.saliency_list[idx]

        return self._getitem_with_aug(idx, img_id, img, saliency)