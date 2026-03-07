from utils.common import *
import torch
import torch.nn.functional as F
import numpy as np
import time

def mixup_data(inputs, labels, alpha):
    """
    Apply MixUp augmentation.
    inputs : tensor (B,C,H,W)
    labels : tensor (B)
    alpha  : Beta distribution parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = inputs.size(0)
    # random permutation of batch
    index = torch.randperm(batch_size).to(device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_inputs, labels_a, labels_b, lam

images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

mixed_images, la, lb, lam = mixup_data(images, labels, alpha=0.4)

print(mixed_images.shape)
print(lam)