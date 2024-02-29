import os
import numpy as np
import cv2
import torch
import face_recognition
import dlib
from torchvision import transforms
import tqdm
from decord import VideoReader,cpu

from dataloader import normalize_data
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
        config,
        ed="conswint_ed_inference",
        vae= "conswint_vae_inference",
        net=net,
        fp16=fp16
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
        
        for face_location in face_locations:
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


def preprocess_frame(frame):
    """function is designed to preprocess a single frame for input into a neural network model.
        Return:
            a preprocessed tensor suitable for input into a neural network model.
    """
    df_tensor=torch.tensor(frame,device=device).float()
    df_tensor=df_tensor.permute((0,3,1,2))
    
    for i in range(len(df_tensor)):
        df_tensor[i]=normalize_data()["vid"](df_tensor[i]/255.0)
        
    return df_tensor

def pred_video(df,model):
    """ function is used to perform inference on a video represented as a sequence of frames.
        Inputs:
            df:video data in the form of tensor where each frame is stacked along the first dimension.
            model: This is the neural network model used for prediction.
        Output:
            The function returns the maximum prediction value obtained from the model's output.
    """
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))
    
def max_prediction_value(y_pred):
    """Finds the index and value of the maximum prediction value"""
    mean_val=torch.mean(y_pred,dim=0)
    return(
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0]>mean_val[1]
        else abs(1-mean_val[1]).item()
    )
    
def real_or_fake(prediction):
    return {0:"Real",1:"Fake"}[prediction^1]

def extract_frames(video_file,frames_nums=10):
    vr=VideoReader(video_file,ctx=cpu(0))
    step_size=max(1,len(vr)//frames_nums)
    return vr.get_batch(list(range(0,len(vr),step_size))[:frames_nums]).asnumpy()

def df_face(vid,num_frames,net):
    img=extract_frames(vid,num_frames)
    face,count=face_recog(img)
    return preprocess_frame(face)if count >0 else []

def is_video(vid):
    return os.path.isfile(vid)and vid.endswith(tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"]))


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }

def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
