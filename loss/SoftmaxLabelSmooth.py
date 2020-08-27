from torch import nn
import torch
from . import Loss

class SoftmaxLabelSmooth(Loss):
    """Softmax with label smoothing

    Performs softmax with label smoothing.

    Args (kwargs only):
        soft_dim (int): Dimension of the softmax layer. Used for the smoothing constant scaling.
        eps (float): Smothing constant. Default 0.1

    Methods: 
        __call__: Returns loss given logits and labels.

    """
    def __init__(self, **kwargs):
        super(SoftmaxLabelSmooth, self).__init__()
        self.soft_dim = kwargs.get('soft_dim')
        self.eps = kwargs.get('eps', 0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def __call__(self, logits, labels):
        """
        Args:
            logits: prediction matrix (before softmax) with shape (batch_size, soft_dim)
            labels: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(logits)
        labels = torch.zeros(log_probs.size()).scatter_(1, labels.unsqueeze(1).data.cpu(), 1)
        labels = labels.cuda()
        labels = (1 - self.eps) * labels + self.eps / self.soft_dim
        loss = (- labels * log_probs).mean(0).sum()
        return loss