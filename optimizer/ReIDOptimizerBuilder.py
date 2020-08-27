import torch

class ReIDOptimizerBuilder:
  def __init__(self,base_lr, lr_bias, gpus, weight_decay, weight_bias):
    self.base_lr = base_lr
    self.gpus = gpus
    self.weight_decay = weight_decay
    self.lr_bias = lr_bias
    self.weight_bias = weight_bias


  def build(self, model, _name = 'Adam', **kwargs):
    params = []
    for key, value in model.named_parameters():
      if value.requires_grad:
        if "bias" in key:
            learning_rate = self.base_lr * self.lr_bias
            weight_decay = self.weight_decay * self.weight_bias
        else:
            learning_rate = self.base_lr * self.gpus
            weight_decay = self.weight_decay
        params += [{"params": [value], "lr":learning_rate, "weight_decay": weight_decay}]
    optimizer = __import__('torch.optim', fromlist=['optim'])
    optimizer = getattr(optimizer, _name)
    optimizer = optimizer(params, **kwargs)
    return optimizer  