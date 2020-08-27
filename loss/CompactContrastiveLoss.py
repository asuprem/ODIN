from itertools import combinations
import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from . import Loss


class CompactContrastiveLoss(Loss):
    """Standard contrastive loss

    Calculates the contrastive loss of a mini-batch.

    Args (kwargs only):
        positive-margin (float, 0.3): Margin constraint to use in loss. Positive classes MUST be closer than this value.
        negative-margin (float, 0.5): Negative classes must be further than d(x,p)+margin
        mine (str): Mining method. Default 'hard'. Supports ['hard', 'all']. 

    Methods: 
        __call__: Returns loss given features and labels.
    """

    def __init__(self, **kwargs):
        self.nmargin = kwargs.get("negative-margin", 0.5)
        self.pmargin = kwargs.get("positive-margin", 0.3)

    def __call__(self, features, labels, epoch):
        positive_pairs, negative_pairs = self.get_pairs(features, labels)
        if features.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        if epoch%2 == 0:
            loss = F.relu(
                (features[positive_pairs[:, 0]] - features[positive_pairs[:, 1]]).pow(2).sum(
                    1).sqrt() - self.pmargin).pow(2)
        #(features[positive_pairs[:, 0]] - features[positive_pairs[:, 1]]).pow(2).sum(1)
        else:   # negative loss
            loss = F.relu(
                self.nmargin - (features[negative_pairs[:, 0]] - features[negative_pairs[:, 1]]).pow(2).sum(
                    1).sqrt()).pow(2)
        #loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()



    
    def get_pairs(self, features, labels):
        distance_matrix = self.pdist(features)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

    def pdist(self, vectors):
        distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
            dim=1).view(-1, 1)
        return distance_matrix