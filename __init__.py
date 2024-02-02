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

