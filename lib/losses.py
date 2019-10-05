import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.25, weight=None, reduction='mean', eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        output = output.clamp(self.eps, 1 - self.eps)
        loss = -(target * self.alpha * (1 - output)**self.gamma * torch.log(output) \
                + (1 - target) * (1 - self.alpha) * output**self.gamma * torch.log(1 - output))
        if self.weight is not None:
            loss *= torch.unsqueeze(self.weight, 0)
        if self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss


class BCEWithLogitsLoss(nn.Module):
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, reduction='mean', pos_weight=None, smooth=0):
        super(BCEWithLogitsLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, input, target):
        if self.smooth != 0:
            target = torch.clamp(target, self.smooth, 1 - self.smooth)

        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
