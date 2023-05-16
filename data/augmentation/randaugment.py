# copyright: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random
import math

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Identity(img, v):
    return img

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)

def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def tensorIdentity(img, v):
    return {'angle': 0., 'translate':[0, 0], 'shear':[0., 0.], 'scale': 1.}

def tensorRotate(img, v):
    v *= -1
    return {'angle': v, 'translate':[0, 0], 'shear':[0., 0.], 'scale': 1.}

def tensorShearX(img, v): # radian -> degree(v * 180 / math.pi)
    dx = int(-1 * math.tan(v) * img.size(-2) / 2) # additional translation(matching to PIL affine shear)
    return {'angle': 0., 'translate':[dx, 0], 'shear':[math.degrees(v), 0.], 'scale': 1.}

def tensorShearY(img, v): # radian -> degree
    dy = int(-1 * math.tan(v) * img.size(-2) / 2) # additional translation(matching to PIL affine shear)
    return {'angle': 0., 'translate':[0, dy], 'shear':[0., math.degrees(v)], 'scale': 1.}

def tensorTranslateX(img, v):
    v *= img.size(-1) * -1
    return {'angle': 0., 'translate':[v, 0], 'shear':[0., 0.], 'scale': 1.}

def tensorTranslateY(img, v):
    v *= img.size(-2) * -1
    return {'angle': 0., 'translate':[0, v], 'shear':[0., 0.], 'scale': 1.}
    
def augment_list(use_geom_augs):
    if use_geom_augs:
        l = [
            (AutoContrast, 0, 1),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.05, 0.95),
            (Solarize, 0, 256),
            (ShearX, -0.3, 0.3),        # geometric
            (ShearY, -0.3, 0.3),        # geometric
            (TranslateX, -0.3, 0.3),    # geometric
            (TranslateY, -0.3, 0.3),    # geometric
            (Rotate, -30, 30),          # geometric
        ]
    else:
        l = [
            (AutoContrast, 0, 1),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.05, 0.95),
            (Solarize, 0, 256),
        ]
    return l

def tensor_augment_list(use_geom_augs): # ignore non-geometric transformations 
    if use_geom_augs:
        l = [
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorShearX,       # geometric
            tensorShearY,       # geometric
            tensorTranslateX,   # geometric
            tensorTranslateY,   # geometric
            tensorRotate,       # geometric
        ]
    else:
        l = [
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
            tensorIdentity,
        ]
    return l

    
class RandAugment:
    def __init__(self, use_geom_augs, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list(use_geom_augs)

    def __call__(self, img, trs=None, only_geometric=False):
        if trs is None:
            transform_idxs = random.choices(range(len(self.augment_list)), k=self.n)
            trs = []
            for i, idx in enumerate(transform_idxs): #ops
                op, min_val, max_val = self.augment_list[idx]
                val = min_val + float(max_val - min_val)*random.random()
                img = op(img, val) 
                # save transforms
                trs.append([idx, val])
        
        # Transform value exists (must use only_geometric=True where trs not None)
        else:
            for idx, val in trs:
                if only_geometric and idx not in [9, 10, 11, 12, 13]:   # ignore non_geometric transforms
                    continue
                op, min_val, max_val = self.augment_list[idx]
                img = op(img, val)

        #cutout_val = random.random() * 0.5 
        #img = Cutout(img, cutout_val) #for fixmatch
        return img, trs

    
if __name__ == '__main__':
    # randaug = RandAugment(3,5)
    # print(randaug)
    # for item in randaug.augment_list:
    #     print(item)
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    img = PIL.Image.open('./u.jpg')
    randaug = RandAugment(3,6)
    img = randaug(img)
    import matplotlib
    from matplotlib import pyplot as plt 
    plt.imshow(img)
    plt.show()
