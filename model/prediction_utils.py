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

def face_recog(frames):
    """This function is designed for face recognition tasks.
        Return:
        If faces are detected (count > 0), return a tuple containing the list 
        of cropped face images (temp_face[:count]) and the number of faces detected
        (count). If no faces are detected (count == 0), return an empty list and 0.    
    """
    temp_face=np.zeros((len(frames),224,224,3),dtype=np.uint8)
    count=0
    mod='cnn'if dlib.DLID_USE_CUDA else "hog"        #Histogram of Oriented Gradients               
    for _,frame in tqdm(enumerate(frames),total=len(frames)):
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        face_locations=face_recognition.face_locations(
            frame,number_of_times_to_upsample=0,model=mod
        )
        
        for face_location in face_location:
            if count<len(frames):
                top,right,bottom,left=face_location
                face_image=frame[top:bottom,left:right] 
                face_image=cv2.resize(
                    face_image,(224,224),interpolation=cv2.INTER_AREA
                )
                face_image=cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB)
                temp_face[count]=face_image
                count+=1
                
            else:
                break
    return ([],0) if count==0 else (temp_face[:count],count)
     