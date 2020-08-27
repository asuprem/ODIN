import torch
from bisect import bisect_right
import pdb

class FineGrainedSteppedLR(torch.optim.lr_scheduler._LRScheduler):
    """FineGrainedSteppedLR allows definition of additively or multiplicative changes to LR (either increasing or decreasing).

    Args:
        -- optimizer: torch.optim object
        -- lr_ops: list of tuples. Each tuple is a 3-tuple of (milestone, op, val).
            -- milestone: int for epoch when an op is triggered
            -- op: is a string, either '+' or '*' for addition or multiplication
            -- val: is a float that will act as an operand with the 'op' operator with the current learning rate.
        -- last_epoch: Previous epoch if resume state or launching new optimizer state during training
        -- relative: `bool`, whether milestones are relative to last_epoch or absolute epoch milestones.

            Example: 
                base_lr = 1. 
                lr_ops = [(1, '+', 0.1), (3, '*', 2), (4, '+', -0.1)]
                milestones = [1,3,4]

                lr_schedule:
                    Epoch 0 -- 1.0      previous_lr = base_lr
                    Epoch 1 -- 1.1      previous_lr = previous_lr '+' 0.1
                    Epoch 2 -- 1.1      no change
                    Epoch 3 -- 2.2      previous_lr = previous_lr '*' 1.0
                    Epoch 4 -- 2.1      previous_lr = previous_lr '+' -0.1
    """

    def __init__(self,optimizer, last_epoch = -1, **kwargs):
        lr_ops = kwargs.get('lr_ops')
        milestones = [item[0] for item in lr_ops]
        if milestones != sorted(milestones):
            raise ValueError("`milestones` in `lr_ops` should be a sorted `list`.")

        relative = kwargs.get('relative', False)
        self.build_lr = [(-1, self.lr_mult, 1)]
        
        for _idx, _item in enumerate(lr_ops):
            _milestone, _op, _val = _item
            if relative:
                _milestone += last_epoch + 1
            if _op == '+':
                _op = self.lr_plus
            elif _op == '*':
                _op = self.lr_mult
            else:
                raise ValueError("Unknown operation token {0}. Expected one of '+', '*'.".format(_item[0]))
            self.build_lr.append((_milestone, _op, _val))

        self.milestones = milestones
        self.lr_ops = lr_ops
        self.lr_idx = last_epoch
        # if relative milestones, then keep initial last_epoch, and use last_epoch-initial_last_epoch as comparison. Else use last_epoch as comparison.
        # use bisect_right to perform binary search to get the correct value using this comparison variable 
        super(FineGrainedSteppedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == -1:
            self.lr_idx = 0
            lr_idx = 0
        else:
            lr_idx = bisect_right(self.milestones, self.last_epoch)
        if lr_idx == self.lr_idx:
            return self.base_lrs
        else:
            self.lr_idx = lr_idx
            self.base_lrs = [self.build_lr[lr_idx][1](base_lr, self.build_lr[lr_idx][2]) for base_lr in self.base_lrs]
            return self.base_lrs

    def lr_mult(self, val, operand):
        return val*operand
    def lr_plus(self, val, operand):
        return val+operand