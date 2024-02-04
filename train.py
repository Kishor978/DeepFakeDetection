import sys,os
import numpy as np 
import torch
from torch import nn
from torch import optim
import time
from time import perf_counter
import pickle

from model.config import load_config
from dataloader import load_data,load_checkpoint
from model.autoencoder import ConvAutoEncoder
from model.varient_autoencoder import ConvVAE

config=load_config()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dir_path,mod,num_epochs,pretrained_model_filename,
                test_model, batch_size):
    print("Loading data....")
    dataloaders,dataset_size=load_data(dir_path,batch_size)
    print("Loadibg data is completed...")
    
    if mod=="ed":
        from train.train_autoencoder import train,valid
        model=ConvAutoEncoder(config)
    
    else:
        from train.train_VAE import train,valid
        model=ConvVAE(config)
    
    optimizer=optim.Adam()