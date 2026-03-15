import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def baseline_step(model, inputs, labels, criterion, **kwargs):
    """
    Standard training step.
    """
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return loss, outputs

def mixup_data(inputs, labels, alpha):
    """
    Apply MixUp augmentation.
    inputs : tensor (B,C,H,W)
    labels : tensor (B)
    alpha  : Beta distr param
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = inputs.size(0)
    # random permutation of batch
    index = torch.randperm(batch_size).to(device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_inputs, labels_a, labels_b, lam

def mixup_step(model, inputs, labels, criterion, **kwargs):
    """
    MixUp training step.
    """
    mixup_alpha = kwargs["mixup_alpha"]
    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
    outputs = model(mixed_inputs)
    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    return loss, outputs


def label_smoothing_loss(outputs, targets, smoothing):
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


def smoothing_step(model, inputs, labels, criterion, **kwargs):
    """
    Label smoothing training step.
    """
    label_smoothing = kwargs["label_smoothing"]
    outputs = model(inputs)
    loss = label_smoothing_loss(outputs, labels, label_smoothing)
    return loss, outputs


def mixup_smoothing_step(model, inputs, labels, criterion, **kwargs):
    """
    MixUp + label smoothing.
    """
    mixup_alpha = kwargs["mixup_alpha"]
    label_smoothing = kwargs["label_smoothing"]
    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
    outputs = model(mixed_inputs)
    loss_a = label_smoothing_loss(outputs, y_a, label_smoothing)
    loss_b = label_smoothing_loss(outputs, y_b, label_smoothing)
    loss = lam * loss_a + (1 - lam) * loss_b
    return loss, outputs

baseline_step.valid_train_accuracy = True
smoothing_step.valid_train_accuracy = True
mixup_step.valid_train_accuracy = False
mixup_smoothing_step.valid_train_accuracy = False