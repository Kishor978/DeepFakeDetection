import torch 
from torch import nn 
from torchvision import transforms 
from timm import create_model 
from config import load_config

class Encoder(nn.Module):
    """Encoder in a variational autoencoder (VAE). 
    It is responsible for transforming input images into a lower-dimensional representation. 
    latent space is characterized by mean (mu) and variance (var) parameters, 
    which are used to sample from a Gaussian distribution to generate latent vectors
    """
    def __init__(self,latent_dims=4) -> None:
        super().__init__()
        
        self.features=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            
            nn.Conv2d(3,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),  
        )
        self.latent_dims=latent_dims
        #fully connected layer 128=number of channels or filters in last CNN, 14*14=spatial dimensions of the feature map, 256=hyperparameter.
        self.fc1=nn.Linear(128*14*14,256) 
        self.fc2=nn.Linear(256,128)
        self.mean=nn.Linear(128*14*14,self.latent_dims)
        self.variance=nn.Linear(128*14*14,self.latent_dims)
        
        self.kl=0
        self.kl_weight=0.5
        self.relu=nn.LeakyReLU()