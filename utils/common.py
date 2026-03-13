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
from .early_stopping import EarlyStopping





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

def train_model(epochs, train_loader, val_loader, model, criterion, optim_method,
                training_step=baseline_step,  early_stopping_patience=None,early_stopping_min_delta=0.0, **kwargs):
    # Training
    print('\nStarting training...')
    # num_epochs = 50
    history: dict = {'epoch_metrics': [],'early_stopping': None,'batch_losses': []}
    #batch_losses = []
    #epoch_losses = []
    accuracy_valid = getattr(training_step, "valid_train_accuracy", True)
    early_stopping_enabled = (
            early_stopping_patience is not None and early_stopping_patience > 0
    )
    early_stopper = EarlyStopping(early_stopping_patience,min_delta=early_stopping_min_delta) if early_stopping_enabled else None
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
            if accuracy_valid:
                predictions = outputs.argmax(dim=1)
                epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += batch_size
        #avg_epoch_loss = epoch_loss / num_batches
        #epoch_losses.append(avg_epoch_loss)
        train_loss = epoch_train_loss_sum / epoch_train_samples
        if accuracy_valid:
            train_accuracy = epoch_train_correct / epoch_train_samples
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion)
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        }
        if accuracy_valid:
            epoch_record['train_accuracy'] = train_accuracy
        history['epoch_metrics'].append(epoch_record)
        if early_stopper:
            stop = early_stopper.update(val_loss, model, epoch + 1)
            if stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        if accuracy_valid:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )
        #print(f'Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}')
    print('Training finished.')
    if early_stopper and early_stopper.stopped_epoch is None:
        early_stopper.stopped_epoch = epochs
    best_val_accuracy = None
    if early_stopper and early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)
        print("Restored best model from early stopping.")
        print(f"Best validation loss {early_stopper.best_val_loss:.4f} at epoch {early_stopper.best_epoch}")
        for m in history["epoch_metrics"]:
            if m["epoch"] == early_stopper.best_epoch:
                best_val_accuracy = m["validation_accuracy"]
                break
        early_stopper.triggered = True
    if early_stopping_enabled:
        history["early_stopping"] = {
            "enabled": True,
            "triggered": early_stopper.triggered,
            "patience": early_stopper.patience,
            "min_delta": early_stopper.min_delta,
            "best_epoch": early_stopper.best_epoch,
            "best_validation_loss": early_stopper.best_val_loss,
            "best_validation_accuracy": best_val_accuracy,
            "stopped_epoch": early_stopper.stopped_epoch
        }
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
        "config": config or {},
        "metrics": history
    }
    with open(history_path, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f'History saved to: {history_path}')
    return history_path

def full_train_old(name, images, labels, train_loader, val_loader, method, epochs, model_dir,
               config=None, dropout_prob=0.0, training_step=baseline_step, save_outputs=True,  **kwargs):
    start_time = time.time()
    model, outputs = init_model(images, dropout_prob)
    criterion, loss = init_loss(outputs, labels)
    #optim_method = init_optimiser(model, 'SGD', lr=0.001, momentum=0.9)
    # separate optimiser kwargs from training-step kwargs
    optimiser_keys = {"lr", "momentum", "weight_decay", "dampening", "nesterov"}
    optimiser_kwargs = {k: v for k, v in kwargs.items() if k in optimiser_keys}
    training_kwargs = {k: v for k, v in kwargs.items() if k not in optimiser_keys}
    optim_method = init_optimiser(model, method, **optimiser_kwargs)
    #batch_losses, epoch_losses = train_model(epochs, train_loader, model, criterion, optim_method)
    history = train_model(epochs, train_loader, val_loader, model, criterion,
                          optim_method, training_step=training_step,
                          early_stopping_patience=config.get("early_stopping_patience") if config else None,
                          early_stopping_min_delta=config.get("early_stopping_min_delta") if config else 0.0,
                          **training_kwargs)
    if save_outputs:
        model_path = save_model(model, name, model_dir)
        history_path = save_history(history, name, 'train', model, model_dir, config=config)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    #return model, batch_losses, epoch_losses
    return model, history, model_path, history_path

def full_train(name, images, labels, train_loader, val_loader, method, epochs, model_dir,
               config=None, dropout_prob=0.0, training_step=baseline_step, save_outputs=True,
               session=None,**kwargs):
    from utils.training_session import create_training_session
    start_time = time.time()
    if session is None:
        session = create_training_session(images, labels, method, dropout_prob, config, training_step, **kwargs)
    session.train(epochs,train_loader,val_loader)
    model_path , history_path = None, None
    if save_outputs:
        model_path = save_model(session.model, name, model_dir)
        history_path = save_history(session.history, name, "train", session.model, model_dir, config=config)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    return session.model, session.history, model_path, history_path

def load_history(path: Path) -> dict:
    """
    Load a JSON history file saved by save_history() in utils/common.py.

    Args:
        path (Path): Path to the JSON history file.

    Returns:
        dict: Full JSON payload including metrics.
    """
    with open(path) as f:
        return json.load(f)


def extract_epoch_metrics(history: dict):
    """
    Pull per-epoch train/val accuracy from a loaded history dict.

    Args:
        history (dict): Loaded JSON history dict from load_history().

    Returns:
        tuple:
            epochs     (list[int])   — epoch numbers
            train_acc  (list[float]) — training accuracy per epoch
            val_acc    (list[float]) — validation accuracy per epoch
    """
    metrics   = history["metrics"]["epoch_metrics"]
    epochs    = [m["epoch"]               for m in metrics]
    train_acc = [m.get("train_accuracy")      for m in metrics]
    val_acc   = [m["validation_accuracy"] for m in metrics]
    return epochs, train_acc, val_acc


def load_model(dropout_prob: float, weights_path: Path) -> torch.nn.Module:
    """
    Instantiate a CNN and load saved weights from a .pt file.

    Args:
        dropout_prob  (float): Dropout probability used when the model was trained.
        weights_path  (Path):  Path to the saved state dict (.pt file).

    Returns:
        torch.nn.Module: Model with loaded weights in eval mode, on device.
    """
    model = CNN(dropout_prob=dropout_prob).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def evaluate_test_set(model, test_loader):
    """
    Evaluate a trained model on the test dataset.
    Args:
        model (torch.nn.Module)
        test_loader (DataLoader)
    Returns:
        dict containing test_loss and test_accuracy
    """
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(test_loader, model, criterion)
    print("\nTest performance")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    return {"test_loss": test_loss, "test_accuracy": test_acc}


def run_test_evaluation(model, test_dataset, batch_size, name, model_dir,config=None):
    """
    Complete test evaluation pipeline.
    Builds test loader → evaluates model → attaches metrics → saves history.
    Args:
        model (torch.nn.Module)
        test_dataset
        batch_size (int)
        history (dict)
        experiment_name (str)
        model_dir (Path)
        config (dict)
    Returns:
        dict: test metrics
    """
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_metrics = evaluate_test_set(model, test_loader)
    history_path = save_history(test_metrics, name, "test", model, model_dir, config=config)
    return test_metrics, history_path

def evaluate_confidence(model, data_loader):
    """
    Compute mean max softmax confidence across a dataset.
    A well-calibrated model produces lower confidence than an overfit one.

    Args:
        model       (torch.nn.Module): Trained model in eval mode.
        data_loader (DataLoader):      Dataset to evaluate over.

    Returns:
        float: Mean of the maximum softmax probability across all samples.
    """
    model.eval()
    total_confidence = 0.0
    total_samples    = 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs   = inputs.to(device)
            logits   = model(inputs)
            probs    = torch.softmax(logits, dim=1)
            max_prob = probs.max(dim=1).values
            total_confidence += max_prob.sum().item()
            total_samples    += inputs.size(0)
    return total_confidence / total_samples

def save_json(data, path):
    """
    Save dictionary as formatted JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)