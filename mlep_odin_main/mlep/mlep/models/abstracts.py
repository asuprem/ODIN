from torch import nn
import torch.nn.functional as F
import torch


class ReidModel(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        super(ReidModel, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.soft_dimensions = soft_dimensions
        self.normalization = normalization if normalization != '' else None
        self.build_base(base, weights, **kwargs)
        
        self.feat_norm = None
        self.build_normalization(self.normalization)
        
        if self.soft_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.soft_dimensions, bias=False)
            self.softmax.apply(self.weights_init_softmax)
        else:
            self.softmax = None

    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('InstanceNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def weights_init_softmax(self, m):
        """ Initialize linear weights to standard normal. Mean 0. Standard Deviation 0.001 """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, std=0.001)
                if m.bias:
                        nn.init.constant_(m.bias, 0.0)
    
    def partial_load(self,weights_path):
        params = torch.load(weights_path)
        for _key in params:
            if _key not in self.state_dict().keys() or params[_key].shape != self.state_dict()[_key].shape: 
                continue
            self.state_dict()[_key].copy_(params[_key])


    def build_base(self,**kwargs):
        """Build the architecture base.        
        """
        raise NotImplementedError()
    def build_normalization(self,**kwargs):
        raise NotImplementedError()
    def base_forward(self,**kwargs):
        raise NotImplementedError()
    def forward(self,**kwargs):
        raise NotImplementedError()

    

