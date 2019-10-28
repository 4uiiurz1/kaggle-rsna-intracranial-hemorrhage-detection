import torch
import torch.nn as nn
import torch.nn.functional as F
from . import efficientnet_pytorch


def lp_pool2d(input, p):
    return torch.mean(input**p, (2, 3))**(1 / p)


class EfficientNet(efficientnet_pytorch.EfficientNet):
    def __init__(self, blocks_args=None, global_params=None, pooling='avg', lp_p=3):
        super().__init__(blocks_args, global_params)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.pooling = pooling
        self.lp_p = lp_p

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        if self.pooling == 'avg':
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        elif self.pooling == 'lp':
            x = lp_pool2d(x, self.lp_p)
        else:
            raise NotImplementedError

        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x
