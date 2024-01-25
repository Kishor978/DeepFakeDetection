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