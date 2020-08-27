import torch
from torch import nn
from . import Loss

class MarginLoss(Loss):
  """Standard triplet loss

    Calculates the margin loss of a mini-batch.

    Args (kwargs only):
        margin (float): Margin constraint to use in triplet liming. If not provided, loss uses torch.nn.SoftMarginLoss. Else uses nn.SoftMarginLoss.
        mine (str): Mining method. Default 'hard'. Supports ['hard', 'all']. 

    Methods: 
        __call__: Returns loss given features and labels.
    """
  def __init__(self, **kwargs):

    self.margin = kwargs.get('margin', None)
    mine = kwargs.get('mine', 'hard')
    
    if self.margin is None:
      self.loss_fn = nn.SoftMarginLoss()
      print("Using SoftMarginLoss")
    else:
      self.loss_fn = nn.MarginRankingLoss(margin=self.margin)
      print("Using Margin Loss with margin=%f"%self.margin)
    if mine == 'hard':
      self.mine = self.hard_mine
      print("Using batch hard mining")
    elif mine == 'all':
      self.mine = self.average_mine
      print("Using batch all mining")
    else:
      raise NotImplementedError()

  def __call__(self, features, labels):
    """
    Args:
        features: features matrix with shape (batch_size, emb_dim)
        labels: ground truth labels with shape (batch_size)
    """
    distances = self.euclidean_dist(features, features)
    distances_pos, distances_neg = self.mine(distances, labels)
    y = distances_neg.new().resize_as_(distances_neg).fill_(1)
    if self.margin is None:
      loss = self.loss_fn(distances_neg - distances_pos, y)
    else:
      loss = self.loss_fn(distances_neg, distances_pos, y)
    return loss

  def euclidean_dist(self, x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
  
  def hard_mine(self, distances, labels):
    N = distances.size(0)
    # shape [N, N]
    pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, _ = torch.max(distances[pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, _ = torch.min(distances[neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an
  def average_mine(self, distances, labels):
    N = distances.size(0)
    # shape [N, N]
    pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, _ = torch.mean(distances[pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, _ = torch.mean(distances[neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an