import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np


def log_loss(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()

    score = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')

    return score
