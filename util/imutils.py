import PIL.Image
import random
import numpy as np

import torchvision.transforms.functional as vision_tf


class RandomResizeLong:
    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, target_long=None, mode='image'):
        if target_long is None:
            target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if mode == 'image':
            img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        elif mode == 'mask':
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)

        return img

class RandomCrop():
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, sal=None):
        h, w, c = imgarr.shape
        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)
        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0
        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]
        if sal is not None:
            container_sal = np.zeros((self.cropsize, self.cropsize,1), np.float32)
            container_sal[cont_top:cont_top+ch, cont_left:cont_left+cw,0] = \
                sal[img_top:img_top+ch, img_left:img_left+cw]
            return container, container_sal

        return container

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont

def random_crop_with_saliency(imgarr, sal, old_crop_size, crop_size, get_box=False, box=None):
    h, w, c = imgarr.shape
    
    if box:
        assert old_crop_size is not None
        # original cropped img, sal
        img_cont = np.ones((old_crop_size, old_crop_size, imgarr.shape[-1]), imgarr.dtype)
        img_cont[box[0]:box[1], box[2]:box[3]] = imgarr[box[4]:box[5], box[6]:box[7]]
        if sal is not None:
            sal_cont = np.zeros((old_crop_size, old_crop_size, 3), np.float32)
            sal_cont[box[0]:box[1], box[2]:box[3]] = sal[box[4]:box[5], box[6]:box[7]]
            sal = sal_cont
        old_box = box
        imgarr = img_cont

        # new cropped img
        h, w, c = imgarr.shape
        box = get_random_crop_box((h, w), crop_size)
        img_cont = np.ones((crop_size, crop_size, imgarr.shape[-1]), imgarr.dtype)
        img_cont[box[0]:box[1], box[2]:box[3]] = imgarr[box[4]:box[5], box[6]:box[7]]
        if sal is not None:
            sal_cont = np.zeros((crop_size, crop_size, 3), np.float32)
            sal_cont[box[0]:box[1], box[2]:box[3]] = sal[box[4]:box[5], box[6]:box[7]]
        else:
            sal_cont = None
    
    else:
        box = get_random_crop_box((h, w), crop_size)
        img_cont = np.ones((crop_size, crop_size, imgarr.shape[-1]), imgarr.dtype)
        img_cont[box[0]:box[1], box[2]:box[3]] = imgarr[box[4]:box[5], box[6]:box[7]]
        
        if sal is not None:
            sal_cont = np.zeros((crop_size, crop_size, 3), np.float32)
            sal_cont[box[0]:box[1], box[2]:box[3]] = sal[box[4]:box[5], box[6]:box[7]]
        else:
            sal_cont = None

    if get_box: 
        return img_cont, sal_cont, box
    else:
        return img_cont, sal_cont

def random_crop_with_saliency_pil(img, mask=None, crop_size=448, get_transform=False, transforms=None):
    w, h = img.size
    w_space = crop_size - w
    h_space = crop_size - h
    w_padding = w_space // 2 if w_space > 0 else 0
    h_padding = h_space // 2 if h_space > 0 else 0

    if transforms is None:
        left = random.randrange(abs(w_padding)+1)
        top = random.randrange(abs(h_padding)+1)
        transforms = [left, top]
    else:
        left, top = transforms

    img = vision_tf.pad(img, [w_padding, h_padding, w_padding + w_space%2, h_padding + h_space%2])
    img = vision_tf.crop(img, top, left, crop_size, crop_size)

    if mask is not None:
        mask = vision_tf.pad(mask, [w_padding, h_padding, w_padding+w_space%2, h_padding+h_space%2])
        mask = vision_tf.crop(mask, top, left, crop_size, crop_size)
   
    if get_transform:
        return img, mask, transforms
    else:
        return img, mask


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():
    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img
