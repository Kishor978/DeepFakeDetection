import torch
from torch import nn
from autoencoder import ConvAutoEncoder
from varient_autoencoder import ConvVAE

class ConSwinT(nn.Module):
    """
    class is designed to handle the initialization and loading of model 
    weights based on the value of the net parameter. Depending on the 
    specified value of net, the forward pass of the model involves either 
    the ConvAutoEncoder model, the ConvVAE model, or both."""
    def __init__(self,config,ed,vae,net,fp16):
        super(ConSwinT,self).__init__()
        self.net=net
        self.fp16=fp16
        if self.net=='ed':
            try: 
                self.model_ed=ConvAutoEncoder(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.model_ed.load_state_dict(self.checkpoint_ed)
                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.helf()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net=='var':
            try: