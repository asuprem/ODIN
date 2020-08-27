import pdb
import torch
from torch import nn
from torch.nn import functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_bn = nn.BatchNorm2d(planes)
        self.lrelu = F.leaky_relu
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.lrelu(x, 0.2)
        return x

class DeConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=2, padding=1):
        super(DeConvBlock, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)
        self.dconv1_bn = nn.BatchNorm2d(planes)
        self.relu = F.relu
    def forward(self, x):
        x = self.dconv1(x)
        x = self.dconv1_bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    expansion_base = 64
    def __init__(self, base, latent_dimensions, channels, init="normal", **kwargs):
        super(Encoder, self).__init__()
        self.init = init
        self.kwargs = kwargs

        self.conv1 = nn.Conv2d(channels, self.expansion_base, kernel_size=3, stride=2, padding=1)
        self.extractor, pre_embedding = self._make_extractor(base, self.expansion_base)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lrelu = F.leaky_relu
        self.embedding = nn.Linear(pre_embedding, latent_dimensions)

    def _make_extractor(self, base, expansion_base):
        starting_size = base // 2   #16
        lblocks = []
        inplanes = expansion_base
        while starting_size > 2:
            lblocks.append(ConvBlock(inplanes, inplanes*2,kernel_size=3))
            inplanes *= 2
            starting_size = starting_size // 2
        return nn.Sequential(*lblocks), inplanes

    def forward(self,x):
        x = self.conv1(x)
        x = self.lrelu(x, 0.2)
        x = self.extractor(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.embedding(x)
        return x        

    def weights_init(self,init="normal"):
        if init != "normal":
            raise NotImplementedError()
        for m in self._modules:
            #m.apply(self.weight_init_normal)
            self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

    def weight_init_normal(self, m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)



class Decoder(nn.Module):
    expansion_base = 64
    def __init__(self, base, latent_dimensions, channels, init="normal", **kwargs):
        super(Decoder, self).__init__()
        self.init = init
        self.kwargs = kwargs

        max_layers = int(math.log(base / 4, 2))
        
        self.dconv1 = nn.ConvTranspose2d(latent_dimensions, int(self.expansion_base*(base/4)), 4, 1, 0)
        self.dconv1_bn = nn.BatchNorm2d(int(self.expansion_base*(base/4)))

        self.upsampler = self._make_upsampler(base, self.expansion_base)
        self.dconv4 = nn.ConvTranspose2d(self.expansion_base, channels, 3, 1, 1)

        self.relu = F.relu
        self.tanh = torch.tanh

    def _make_upsampler(self,base,expansion_base):
        startsize = 4
        lblocks = []
        max_layers = int(math.log(base / 4, 2))
        while startsize < base:
            lblocks.append(DeConvBlock(int(expansion_base*(base/startsize)), int(expansion_base*(base/(startsize*2))),kernel_size=3,stride=2,padding=1))
            startsize*= 2
        return nn.Sequential(*lblocks)

    def forward(self,x):
        x = x.unsqueeze(-1)
        x = self.dconv1(x)
        x = self.dconv1_bn(x)
        x = self.upsampler(x)
        x = self.dconv4(x)
        x = self.tanh(x) * 0.5 + 0.5
        return x        

    def weights_init(self,init="normal"):
        if init != "normal":
            raise NotImplementedError()
        for m in self._modules:
            #m.apply(self.weight_init_normal)
            self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

    def weight_init_normal(self, m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)

class LatentDiscriminator(nn.Module):
    expansion_base = 64
    def __init__(self, latent_dimensions, init="normal", **kwargs):
        super(LatentDiscriminator, self).__init__()
        self.init = init
        self.kwargs = kwargs

        self.dense1 = nn.Linear(latent_dimensions, self.expansion_base)
        self.dense2 = nn.Linear(self.expansion_base, self.expansion_base*2)
        self.dense3 = nn.Linear(self.expansion_base*2, 1)

        self.lrelu = F.leaky_relu
        self.sigmoid = torch.sigmoid

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.dense1(x)
        x = self.lrelu(x, 0.2)
        x = self.dense2(x)
        x = self.lrelu(x, 0.2)
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x

    def weights_init(self,init="normal"):
        if init != "normal":
            raise NotImplementedError()
        for m in self._modules:
            #m.apply(self.weight_init_normal)
            self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

    def weight_init_normal(self, m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):
    expansion_base = 64
    def __init__(self, base, channels, init="normal", **kwargs):
        super(Discriminator, self).__init__()
        self.init = init
        self.kwargs = kwargs

        self.conv1 = nn.Conv2d(channels, self.expansion_base, kernel_size=3, stride=2, padding=1)
        self.extractor, pre_embedding = self._make_extractor(base, self.expansion_base)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lrelu = F.leaky_relu
        self.sigmoid = torch.sigmoid
        self.fc1 = nn.Linear(pre_embedding, int(pre_embedding/2))
        self.fc2 = nn.Linear(int(pre_embedding/2), 1)

    def _make_extractor(self, base, expansion_base):
        starting_size = base // 2   #16
        lblocks = []
        inplanes = expansion_base
        while starting_size > 2:
            lblocks.append(ConvBlock(inplanes, inplanes*2,kernel_size=3))
            inplanes *= 2
            starting_size = starting_size // 2
        return nn.Sequential(*lblocks), inplanes

    def forward(self,x):
        x = self.conv1(x)
        x = self.lrelu(x, 0.2)
        x = self.extractor(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x        
    
    def weights_init(self,init="normal"):
        if init != "normal":
            raise NotImplementedError()
        for m in self._modules:
            #m.apply(self.weight_init_normal)
            self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

    def weight_init_normal(self, m, mean=0.0, std=0.02):
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.constant_(m.bias, 0.0)


class VAEGAN(nn.Module):
    def __init__(self, base = 32, latent_dimensions = 128,**kwargs):
        """ VAEGAN Builder
        Images must be square sized.

        Args:
            -- base: Image dimensions. Must be a power of 2
            -- latent_dimensions: Latend dimensionality...

        """
        super(VAEGAN, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.channels = kwargs.get("channels", 3)
        pass
        self.Encoder = Encoder(base, self.latent_dimensions, self.channels)
        self.Decoder = Decoder(base, self.latent_dimensions, self.channels)
        self.LatentDiscriminator = LatentDiscriminator(self.latent_dimensions)
        self.Discriminator = Discriminator(base, self.channels)

        # self.Encoder.cuda()
        # self.Decoder.cuda()
        # self.LatentDiscriminator.cuda()
        # self.Discriminator.cuda()

        self.Encoder.weights_init()
        self.Decoder.weights_init()
        self.LatentDiscriminator.weights_init()
        self.Discriminator.weights_init()

    def forward(self,x):
        return self.Decoder(self.Encoder(x).unsqueeze(-1))

