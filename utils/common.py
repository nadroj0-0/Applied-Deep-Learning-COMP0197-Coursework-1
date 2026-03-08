import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import time
import json
import random
import numpy as np
from .network import CNN




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def set_seed(seed=None):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Also configures deterministic CUDA behaviour.
    """
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)
    print(f"Random seed set to {seed}")
    return generator, seed

def init_seed(cfg):
    """
    Resolve seed from config, initialise RNGs, and record the final seed.
    Returns the dataloader generator.
    """
    seed = cfg.get("seed")
    generator, seed = set_seed(seed)
    cfg["seed"] = seed
    return generator

def download_data():
    # Download the data
    print('Downloading CIFAR-10 dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    print('Dataset downloaded successfully.')
    return (train_dataset, test_dataset)


def load_data_pytorch(train_dataset, batch_size, validation_fraction, generator):
    # Load the data into PyTorch
    print('Loading dataset into PyTorch...')
    total_size = len(train_dataset)
    val_size = int(validation_fraction * total_size)
    train_size = total_size - val_size
    train_subset, val_subset = random_split(train_dataset,[train_size, val_size],generator=generator)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    images, labels = next(iter(train_loader))
    return images, labels, train_loader, val_loader


def inspect_data(images, labels, train_dataset):
    # Inspect a few samples of the data
    print('Batch images shape:', images.shape)
    print('Batch labels shape:', labels.shape)
    image = images[0]  # Sample 1 image
    label = labels[0]  # Sample 1 label
    print('First image tensor shape:', image.shape)
    print('First label:', label)
    print('Min pixel value:', image.min().item())
    print('Max pixel value:', image.max().item())
    # Inspect the different labels
    classes = train_dataset.classes
    print('Classes:', classes)
    return classes


def init_model(images, dropout_prob=0.0):
    # Create the model
    print('\nCreating model...')
    model = CNN(dropout_prob).to(device)
    print(model)
    # Test a forward pass
    print('\nTesting forward pass...')
    images = images.to(device)
    outputs = model(images)
    print('Model output shape:', outputs.shape)
    return model, outputs


def init_loss(outputs, labels):
    # Loss function
    print('\nCreating loss function...')
    criterion = nn.CrossEntropyLoss()
    labels = labels.to(device)
    # Test loss calculation on the current batch
    loss = criterion(outputs, labels)
    print('Initial loss:', loss.item())
    return criterion, loss


def init_optimiser(model, method, **kwargs):
    import inspect
    # Optimiser
    print('\nCreating optimiser...')
    # optimMethod = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if not hasattr(optim, method):
        raise ValueError(f'Optimizer {method} not found in torch.optim')
    optimiser_class = getattr(optim, method)
    try:
        optim_method = optimiser_class(model.parameters(), **kwargs)
    except TypeError as e:
        expected_signature = inspect.signature(optimiser_class)
        print(f"\nInvalid arguments for optimizer '{method}'")
        print('Expected constructor signature:')
        print(f'{method}{expected_signature}')
        raise TypeError(
            f"Invalid arguments for optimizer '{method}'."
            f'Expected signature: {method}{expected_signature}'
        )
    #optim_method = optimiser_class(model.parameters(), lr=learn_rate, momentum=momentum_par)
    print('Optimiser created:', optim_method)
    return optim_method

# ================================
# Training step strategies
# ================================

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

def mixup_step(model, inputs, labels, criterion, **kwargs):
    """
    MixUp training step.
    """
    alpha = kwargs["alpha"]
    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha)
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
    smoothing = kwargs["smoothing"]
    outputs = model(inputs)
    loss = label_smoothing_loss(outputs, labels, smoothing)
    return loss, outputs


def mixup_smoothing_step(model, inputs, labels, criterion, **kwargs):
    """
    MixUp + label smoothing.
    """
    alpha = kwargs["alpha"]
    smoothing = kwargs["smoothing"]
    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha)
    outputs = model(mixed_inputs)
    loss_a = label_smoothing_loss(outputs, y_a, smoothing)
    loss_b = label_smoothing_loss(outputs, y_b, smoothing)
    loss = lam * loss_a + (1 - lam) * loss_b
    return loss, outputs

def evaluate_model(data_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy

def train_model(epochs, train_loader, val_loader, model, criterion, optim_method, training_step=baseline_step, **kwargs):
    # Training
    print('\nStarting training...')
    # num_epochs = 50
    history = {'epoch_metrics': [], 'batch_losses': []}
    #batch_losses = []
    #epoch_losses = []
    for epoch in range(epochs):
        model.train()
        #epoch_loss = 0
        #num_batches = 0
        epoch_train_loss_sum = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optim_method.zero_grad()
            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
            loss, outputs = training_step(model,inputs,labels,criterion,**kwargs)
            loss.backward()
            optim_method.step()
            batch_size = labels.size(0)
            loss_value = loss.item()
            history['batch_losses'].append({
                'epoch': epoch+1,
                'batch': i+1,
                'loss': loss_value
            })
            #epoch_loss += loss_value
            #num_batches += 1
            epoch_train_loss_sum += loss_value * batch_size
            predictions = outputs.argmax(dim=1)
            epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += batch_size
        #avg_epoch_loss = epoch_loss / num_batches
        #epoch_losses.append(avg_epoch_loss)
        train_loss = epoch_train_loss_sum / epoch_train_samples
        train_accuracy = epoch_train_correct / epoch_train_samples
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion)
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        }
        history['epoch_metrics'].append(epoch_record)
        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )
        #print(f'Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}')
    print('Training finished.')
    #return batch_losses, epoch_losses
    return history

def save_model(model, name, model_dir):
    """
    Saves trained PyTorch model inside 'models' dir. If dir doesn't exist dir created
    Parameters
    model : torch.nn.Module - Trained model to save
    name : str - Name for file
    Returns
    path : Path - Full path to saved file
    """
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{name}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')
    return model_path

def save_history(history, name, stage, model, model_dir, config=None):
    model_dir.mkdir(exist_ok=True)
    history_path = model_dir / f'{name}_{stage}_history.json'
    payload = {
        "model": name,
        "architecture": str(model),
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": history
    }
    with open(history_path, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f'History saved to: {history_path}')
    return history_path

def full_train(name, images, labels, train_loader, val_loader, method, epochs, model_dir,
               config=None, dropout_prob=0.0, training_step=baseline_step, **kwargs):
    start_time = time.time()
    model, outputs = init_model(images, dropout_prob)
    criterion, loss = init_loss(outputs, labels)
    #optim_method = init_optimiser(model, 'SGD', lr=0.001, momentum=0.9)
    optim_method = init_optimiser(model, method, **kwargs)
    #batch_losses, epoch_losses = train_model(epochs, train_loader, model, criterion, optim_method)
    history = train_model(epochs, train_loader, val_loader, model, criterion,
                          optim_method, training_step=training_step, **(kwargs or {}))
    model_path = save_model(model, name, model_dir)
    history_path = save_history(history, name, 'train', model, model_dir, config=config)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    #return model, batch_losses, epoch_losses
    return model, history, model_path, history_path