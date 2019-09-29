import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics


def log_loss(output, target, eps=1e-15):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = np.clip(output, eps, 1 - eps)
    loss = -(target * np.log(output) + (1 - target) * np.log(1 - output))
    weights = np.array([1., 1., 1., 1., 1., 2.])
    loss = np.average(loss, axis=1, weights=weights)
    score = np.mean(loss)

    return score
