import pdb
import importlib
from torch import nn
from .abstracts import ReidModel
from utils import layers
import torch

class CarzamResnet(ReidModel):
    """Basic CarZam Resnet model.

    A CarZam model is similar to a Re-ID model, except it may incorporate proxies in addition to a softmax layer. It yields a feature map of an input.

    Args:
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.

    Kwargs (MODEL_KWARGS):
        last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
        attention (str, None): The attention module to use. Only supports ['cbam', None]
        input_attention (bool, false): Whether to include the IA module
        secondary_attention (int, None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic block number provided here.

        proxies (bool, False): Whether to incorporate proxies into the CarZam Model
        proxy_classes (int): Required if using proxies. Sets the number of proxies to use. For now, using static proxy assignment.

    Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS):
        zero_init_residual (bool, false): Whether the final layer uses zero initialization
        top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        num_classes (int, 1000): Number of features in final imagenet FC layer
        groups (int, 1): Used during resnet variants construction
        width_per_group (int, 64): Used during resnet variants construction
        replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
        norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D

    Methods: 
        forward: Process a batch

    """
    def __init__(self, base = 'resnet50', weights=None, normalization=None, embedding_dimensions=None, **kwargs):
        super(CarzamResnet, self).__init__(base, weights, normalization, embedding_dimensions, soft_dimensions=None, **kwargs)

        # Set up the proxy
        proxies = kwargs.get("proxies", False)
        if proxies:
            proxy_classes = kwargs.get("proxy_classes")
            self.proxy = nn.Parameter(torch.randn(proxy_classes, embedding_dimensions))
        else:
            self.proxy = None

    def build_base(self,base, weights, **kwargs):
        """Build the model base.

        Builds the architecture base/core.
        """
        _resnet = __import__("backbones.resnet", fromlist=["resnet"])
        _resnet = getattr(_resnet, base)
        self.base = _resnet(last_stride=1, **kwargs)
        if weights is not None:
            self.base.load_param(weights)
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.emb_linear = None
        if self.embedding_dimensions > 512 * self.base.block.expansion:
            raise ValueError("You are trying to scale up embedding dimensions from %i to %i. Try using same or less dimensions."%(512*self.base.block.expansion, self.embedding_dimensions))
        elif self.embedding_dimensions == 512*self.base.block.expansion:
            pass
        else:
            self.emb_linear = nn.Linear(self.base.block.expansion*512, self.embedding_dimensions, bias=False)
    
    def build_normalization(self, normalization):
        if self.normalization == 'bn':
            self.feat_norm = nn.BatchNorm1d(self.embedding_dimensions)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "in":
            self.feat_norm = layers.FixedInstanceNorm1d(self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "gn":
            self.feat_norm = nn.GroupNorm(self.embedding_dimensions // 16, self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "ln":
            self.feat_norm = nn.LayerNorm(self.embedding_dimensions,elementwise_affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == 'l2':
            self.feat_norm = layers.L2Norm(self.embedding_dimensions,scale=1.0)
        elif self.normalization is None or self.normalization == '':
            self.feat_norm = None
        else:
            raise NotImplementedError()


    def base_forward(self,x):
        features = self.gap(self.base(x))
        features = features.view(features.shape[0],-1)
        if self.emb_linear is not None:
            features = self.emb_linear(features)
        return features


    def forward(self,x):
        features = self.base_forward(x)
        
        if self.feat_norm is not None:
            inference = self.feat_norm(features)
        else:
            inference = torch.nn.functional.normalize(features, p = 2, dim = 1)

        if self.training:
            return inference, self.proxy
        else:
            return inference
        """
        if self.training:
            if self.softmax is not None:
                soft_logits = self.softmax(inference)
            else:
                soft_logits = None
            return soft_logits, features
        else:
            return inference
        """