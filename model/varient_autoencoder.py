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
    def __init__(self,latent_dims=4):
        super(Encoder,self).__init__()
        
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
        
    def reparameterize(self,x):
        """This function is generating a latent vector z from the mean (mu) and variance (std)
        parameters produced by the encoder. It implements the reparameterization trick, 
        which allows for backpropagation through the sampling process."""
        std = torch.exp(0.5 * self.mu(x))  # standard deviation
        eps = torch.randn_like(std)           #Generate Random Noise
        z = eps * std + self.mu(x)
        
        return z,std
    
    def forward(self,x):
        x=self.features()
        x=torch.flatten(x,start_dim=1)
        mu=self.mu(x)
        var=self.var(x)
        z,_=self.reparameterize(x)
        #  Kullback-Leibler (KL) Divergence used as the loss function  during training 
        # encourage the learned latent space to be close to a standard Gaussian distribution.
        self.kl = self.kl_weight*torch.mean(-0.5*torch.sum(1+var - mu**2 - var.exp(), dim=1), dim=0) 
        return z
    
    
class Decoder(nn.Module):
    """Decoder in a variational autoencoder (VAE). 
    Is responsible for transforming the latent vector back into an image by using transpose convolutional layers"""
    def __int__(self,latent_dims=4):
        super(Decoder,self).__init__()
        
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.latent_dims = latent_dims
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))

    def forward(self, x): 
        x = self.unflatten(x)
        x = self.features(x)
        return x