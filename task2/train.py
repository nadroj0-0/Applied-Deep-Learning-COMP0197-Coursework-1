from utils.common import *
import torch
import torch.nn.functional as F
import numpy as np
import time

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 50,
    "optimiser": "SGD",
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "validation_fraction": 0.2,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5
}

def mixup_data(inputs, labels, alpha):
    """
    Apply MixUp augmentation.
    inputs : tensor (B,C,H,W)
    labels : tensor (B)
    alpha  : Beta distr param
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

def label_smoothing_loss(outputs, targets, smoothing=0.1):
    """
    Custom label-smoothed cross entropy.
    outputs  : model logits (batch_size, num_classes)
    targets  : integer class labels (batch_size)
    smoothing: epsilon value
    """
    num_classes = outputs.size(1)
    # convert logits → log probabilities
    log_probs = F.log_softmax(outputs, dim=1)
    # create smoothed target distribution
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    loss = (-true_dist * log_probs).sum(dim=1).mean()
    return loss

def main():
    try:
        cfg = TRAIN_CONFIG
    except NameError:
        raise RuntimeError(
            "TRAIN_CONFIG must be defined before calling main(). "
            "It defines the experiment hyperparameters."
        )
    generator = init_seed(cfg)
    train_dataset, test_dataset = download_data()

    images, labels, train_loader, val_loader = load_data_pytorch(
        train_dataset,
        batch_size=cfg["batch_size"],
        validation_fraction=cfg["validation_fraction"],
        generator=generator
    )

    # ---- test MixUp ----
    images = images.to(device)
    labels = labels.to(device)

    mixed_images, la, lb, lam = mixup_data(images, labels, alpha=cfg["mixup_alpha"])

    print("MixUp output shape:", mixed_images.shape)
    print("Lambda:", lam)

    # ---- test label smoothing ----
    outputs = torch.randn(4,10).to(device)
    labels = torch.tensor([1,3,5,2]).to(device)

    loss = label_smoothing_loss(outputs, labels)

    print("Label smoothing loss:", loss)


if __name__ == "__main__":
    main()