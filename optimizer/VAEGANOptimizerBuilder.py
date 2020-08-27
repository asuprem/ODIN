import torch

class VAEGANOptimizerBuilder:
  def __init__(self,base_lr, gpus=1, lr_bias=None, weight_decay=None, weight_bias=None):
    self.base_lr = base_lr
    self.gpus = gpus
    if self.gpus > 1:
      raise NotImplementedError()
    #self.weight_decay = weight_decay
    #self.lr_bias = lr_bias
    #self.weight_bias = weight_bias


  def build(self, vaegan_model, _name = 'Adam', **kwargs):
    
    optimizer = __import__('torch.optim', fromlist=['optim'])
    optimizer = getattr(optimizer, _name)
    
    encoder_opt = optimizer(vaegan_model.Encoder.parameters(), lr = self.base_lr, **kwargs)
    decoder_opt = optimizer(vaegan_model.Decoder.parameters(), lr = self.base_lr, **kwargs)
    discriminator_opt = optimizer(vaegan_model.Discriminator.parameters(), lr = self.base_lr, **kwargs)
    autoencoder_opt = optimizer(list(vaegan_model.Encoder.parameters()) + list(vaegan_model.Decoder.parameters()), lr = self.base_lr, **kwargs)
    latent_opt = optimizer(vaegan_model.LatentDiscriminator.parameters(), lr = self.base_lr, **kwargs)

    return {"Encoder":              encoder_opt, \
            "Decoder":              decoder_opt, \
            "Discriminator":        discriminator_opt, \
            "Autoencoder":          autoencoder_opt, \
            "LatentDiscriminator":  latent_opt}
