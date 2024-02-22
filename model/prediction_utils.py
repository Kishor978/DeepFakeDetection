import os
import numpy as np
import cv2
import torch
import face_recognition
import dlib
from torchvision import transforms
import tqdm
from decord import VideoReader,cpu
from .config import load_config
from .conSwinT import ConSwinT


device="cuda" if torch.cuda.is_available() else "cpu"

def load_conswint(net,fp16):
    """This function is used to prepare a pre-trained conswint model for inference tasks,
    such as image generation or classification.

        Input:
        net: Specifies the type of network to load, which can be either 'ed' for the 
            Encoder-Decoder (ED) variant or 'vae' for the Variational Autoencoder (VAE) variant.
        fp16: Boolean flag indicating whether to use FP16 (half-precision) precision. It enables 
            faster computation with reduced memory usage if supported by the hardware."""
    config=load_config()
    model=ConSwinT(
        config,ed="conswint_ed_inference",
        vae= "conswint_vae_inference",
        net=net,fp16=fp16
    )
    model.to(device)
    model.eval()
    if fp16:
        model.half()
    return model

def face_recog(frames,p=None,klass=None):
    temp_face=np.zeros((len(frames),224,224,3),dtype=np.uint8)
    count=0
    mod='cnn'if dlib.DLID_USE_CUDA else "hog"
     