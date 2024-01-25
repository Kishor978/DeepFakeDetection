import torch
from torch import nn 

class Embedder(nn.Module):
    """
    Extract feature map form CNN, flatten, project to embedding dimensions
    This is typically used in vision tasks where you want to convert image data into a lower-dimensional embedding.
    """
    
    def __init__(self, backbone,img_size=224,patch_size=1, feature_size=None,in_chans=2,embed_dim=768):
        super().__init__()
        assert isinstance(backbone,nn.Module)
        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size),
        self.img_size=img_size
        self.patch_size=patch_size
        self.backbone=backbone
        if feature_size is None:
            with torch.no_grad():
                training=backbone.training
                if training:
                    backbone.eval()
                backbone_output=self.backbone(torch.zeros(1,in_chans,img_size[0],img_size[1]))
                if isinstance(backbone_output,(list,tuple)):
                    backbone_output=backbone_output[-1]
                feature_size=backbone_output.shape[-2]
                feature_dim=backbone_output.shape[-1]
                backbone.train(training)
                
        else:
            feature_size=(feature_size,feature_size)
            if hasattr(self.backbone,'feature_info'):
                feature_dim=self.backbone.feature_info.channels()[-1]
            else:
                feature_dim=self.backbone.num_features
        
        assert feature_size[0]%patch_size[0]== 0 and feature_size[1]%patch_size[1]==0
        self.grid_size=feature_size[0]//patch_size[0],feature_size[1]//patch_size[1]
        self.num_patches=self.grid_size[0]*self.grid_size[1]
        self.proj=nn.Conv2d(feature_dim,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x=self.backbone(x)
        if isinstance(x,(list,tuple)):
            x=x[-1]
        x-self.proj(x).flatten(2).transpose(1,2)
        return x