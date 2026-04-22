import timm
import torch.nn as nn
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class MobileNetV4(nn.Module):
    """MobileNetV4 backbone via timm.
    
    Args:
        model_name (str): timm model name e.g. 'mobilenetv4_conv_small'
        pretrained (bool): load ImageNet pretrained weights
        out_indices (tuple): which stages to output (0-4)
    """
    
    def __init__(self,
                 model_name='mobilenetv4_conv_small',
                 pretrained=True,
                 out_indices=(1, 2, 3, 4)):
        super().__init__()
        self.out_indices = out_indices
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices)
        
    def forward(self, x):
        return self.model(x)
    
    def init_weights(self, pretrained=None):
        pass  # handled by timm
