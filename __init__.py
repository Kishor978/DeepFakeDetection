import os 
import torch
import numpy as np
from PIL import Image
from torchvision import transforms,datasets
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    HueSaturationValue,
    GaussNoise,
    Sharpen,Emboss,
    RandomBrightnessContrast,
    OneOf,Compose
    )

def strong_aug(p=0.5):
    """returns a composition of various image augmentation transformations"""
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf([GaussNoise()],p=0.2),
            ShiftScaleRotate(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast()
                ],p=0.2
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p
    )
    
def augument(aug,image):
    return aug(image=image)["image"]

class Aug(object):
    def __call__(self,img):
        aug=strong_aug(p=0.9)
        return Image.fromarray(augument(aug,np.array(img)))
    
def normalize_data():
    mean= [0.485, 0.456, 0.406]
    std=[0.229,0.224,0.225]
    
    return {
        "train":transforms.Compose(
            [Aug(),transforms.ToTensor(),transforms.Normalize(mean,std)]
        ),
        "valid":transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize(mean,std)]
        ),
        "test":transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize(mean,std)]
        ),
        "vid":transforms.Compose([transforms.Normalize(mean,std)])
    }
    