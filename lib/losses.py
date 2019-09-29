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
