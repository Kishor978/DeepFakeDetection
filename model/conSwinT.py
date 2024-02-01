import torch
from torch import nn
from autoencoder import ConvAutoEncoder
from varient_autoencoder import ConvVAE

class ConSwinT(nn.Module):
    