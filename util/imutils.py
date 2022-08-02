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


class RandomCrop:

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

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

        return container


def random_crop_with_saliency(imgarr, mask, crop_size, get_transform=False, transforms=None):

    h, w, c = imgarr.shape

    if transforms is None:
        ch = min(crop_size, h)
        cw = min(crop_size, w)

        w_space = w - crop_size
        h_space = h - crop_size

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
    else:
        ch, cw, cont_left, img_left, cont_top, img_top = transforms

    container = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container_mask = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    container_mask[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        mask[img_top:img_top+ch, img_left:img_left+cw]

    if get_transform: ###
        return container, container_mask, [ch, cw, cont_left, img_left, cont_top, img_top]
    else:
        return container, container_mask


def random_crop_with_saliency_pil(img, mask, crop_size, get_transform=False, transforms=None):
    w, h = img.size
    w_space = crop_size - w
    h_space = crop_size - h
    w_padding = w_space // 2 if w_space > 0 else 0
    h_padding = h_space // 2 if h_space > 0 else 0

    if transforms is None:
        left = random.randrange(abs(w_padding)+1)
        top = random.randrange(abs(h_padding)+1)
        transforms = [left, top]
        #print(f'w:{w},h:{h},w_padding:{w_padding}, h_padding:{h_padding}, left:{left}, top:{top}\n')
    else:
        left, top = transforms
    #print('size', img.size, vision_tf.pad(img, [w_padding, h_padding, w_padding + w_space%2, h_padding + h_space%2]).size)

    img = vision_tf.pad(img, [w_padding, h_padding, w_padding + w_space%2, h_padding + h_space%2])
    img = vision_tf.crop(img, top, left, crop_size, crop_size)
    mask = vision_tf.pad(mask, [w_padding, h_padding, w_padding + w_space%2, h_padding + h_space%2])
    mask = vision_tf.crop(mask, top, left, crop_size, crop_size)
   
    if get_transform: ###
        return img, mask, transforms
    else:
        return img, mask


# # Use batch
# def reverse_random_crop_with_saliency(img, mask, org_h, org_w, org_top, org_left):
    
#     B, C, H, W = img.shape
#     print(B, org_h, org_w, img.shape[-1])
#     container = np.zeros((B, org_h, org_w, img.shape[-1]), np.float32)
#     container_mask = np.zeros((B, org_h, org_w, img.shape[-1]), np.float32)

#     # Non-broadcasting
#     for b in range(B):
#         container[b, org_top:org_top+H, org_left:org_left+W] = \
#             img[b, :, :]
#         container_mask[b, org_top:org_top+H, org_left:org_left+W] = \
#             mask[b, :, :]

#     return container, container_mask


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
