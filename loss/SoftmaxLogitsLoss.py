import torch
from . import Loss
class SoftmaxLogitsLoss(Loss):
  """Standard softmax loss

    Calculates the softmax loss.

    Methods: 
        __call__: Returns loss given logits and labels.

    """
  def __call__(self, logits, labels):
    """
    Args:
        logits: prediction matrix (before softmax) with shape (batch_size, soft_dim)
        labels: ground truth labels with shape (batch_size)
    """
    return torch.nn.functional.cross_entropy(logits, labels)