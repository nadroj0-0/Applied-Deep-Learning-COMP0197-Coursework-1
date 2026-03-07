import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_DIR = Path('models')
TRAIN_CONFIG = {
    'epochs': 10,
    'optimiser': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'reg_dropout': 0.5,
    'batch_size': 64,
    'validation_fraction': 0.2
}


class CNN(nn.Module):
    def __init__(self, dropout_prob = 0.0):
        super().__init__()
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # First convolution layer, Conv2d performs a 2D convolution over the image.
            # number of input channels (CIFAR images are RGB) = 3
            # number of filters (output feature maps) = 32
            # kernel_size=3 = each filter is 3x3
            # padding=1 ensures spatial size stays the same
            # Input shape:  (batch_size, 3, 32, 32), Output shape: (batch_size, 32, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # ReLU activation introduces non-linearity.
            nn.ReLU(),
            # 2nd convolution layer
            # Input:  (batch, 32, 32, 32), Output: (batch, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool to downsample the spatial dimensions.
            # kernel_size=2 halves height and width, (batch, 64, 32, 32) → (batch, 64, 16, 16)
            nn.MaxPool2d(2),
            # 3rd and 4th convolution layers
            # Input channels = 64 (from previous layer), Output channels = 128 feature maps.
            # Input:  (batch, 64, 16, 16), Output: (batch, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Second spatial downsampling
            # (batch, 128, 16, 16) → (batch, 128, 8, 8)
            nn.MaxPool2d(2),
            # 5th and 6th convolution layer
            # Input:  (batch, 128, 8, 8), Output: (batch, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 3rd spatial downsampling
            # (batch, 256, 8, 8) → (batch, 256, 4, 4)
            nn.MaxPool2d(2)
        )

        # Fully connected classifier
        # After convolutions do classical neural network layers.
        self.fc_layers = nn.Sequential(
            # Flatten converts the 3D feature map into a vector.
            # (batch, 256, 4, 4) → (batch, 256 * 4 * 4), 256 * 4 * 4 = 4096 features
            nn.Flatten(),
            # First dense (fully connected) layer
            # Input size = 4096, Output size = 512 hidden units
            # This layer learns combinations of the spatial features discovered by the convolutional layers.
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            # Second classification layer
            # 512 → 256 outputs
            # This layer learns combinations of the 512 features from the previous layer.
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            # Final classification layer
            # 256 → 10 outputs
            # CIFAR-10 has 10 classes, so the network outputs, 10 class scores.
            nn.Linear(256, 10)
        )

    # Defines how data flows through the network.
    def forward(self, x):
        # Pass input images through convolutional feature extractor, x shape: (batch, 3, 32, 32) → (batch, 64, 8, 8)
        x = self.conv_layers(x)
        # Pass the extracted features into the fully connected classifier.
        # Each row corresponds to the class scores for one image, x shape: (batch, 64, 8, 8) → (batch, 10)
        x = self.fc_layers(x)
        return x


def download_data():
    # Download the data
    print('Downloading CIFAR-10 dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print('Dataset downloaded successfully.')
    return (train_dataset, test_dataset)


def load_data_pytorch(train_dataset, batch_size, validation_fraction):
    # Load the data into PyTorch
    print('Loading dataset into PyTorch...')
    total_size = len(train_dataset)
    val_size = int(validation_fraction * total_size)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train_dataset,[train_size, val_size],generator=generator)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
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

def train_model(epochs, train_loader, val_loader, model, criterion, optim_method):
    # Training
    print('\nStarting training...')
    # num_epochs = 50
    history = {'batch_losses': [], 'epoch_metrics': []}
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

def save_model(model, name):
    """
    Saves trained PyTorch model inside 'models' dir. If dir doesn't exist dir created
    Parameters
    model : torch.nn.Module - Trained model to save
    name : str - Name for file
    Returns
    path : Path - Full path to saved file
    """
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f'{name}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')
    return model_path

def save_history(history, name, stage, model, config=None):
    MODEL_DIR.mkdir(exist_ok=True)
    history_path = MODEL_DIR / f'{name}_{stage}_history.json'
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

def full_train(name, images, labels, train_loader, val_loader, method, epochs, dropout_prob=0.0, **kwargs):
    start_time = time.time()
    model, outputs = init_model(images, dropout_prob)
    criterion, loss = init_loss(outputs, labels)
    #optim_method = init_optimiser(model, 'SGD', lr=0.001, momentum=0.9)
    optim_method = init_optimiser(model, method, **kwargs)
    #batch_losses, epoch_losses = train_model(epochs, train_loader, model, criterion, optim_method)
    history = train_model(epochs, train_loader, val_loader, model, criterion, optim_method)
    model_path = save_model(model, name)
    history_path = save_history(history, name, 'train', model, config=TRAIN_CONFIG)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    #return model, batch_losses, epoch_losses
    return model, history, model_path, history_path

def main():
    train_dataset, test_dataset = download_data()
    cfg = TRAIN_CONFIG
    images, labels, train_loader, val_loader = load_data_pytorch(
        train_dataset, batch_size=cfg['batch_size'],
        validation_fraction=cfg['validation_fraction']
    )
    inspect_data(images, labels, train_dataset)

    base_model, base_history, base_model_path, base_history_path = full_train(
        'baseline', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum']
    )
    print('\nBase model:')
    print(base_model)
    print('\nBase final epoch metrics:')
    print(base_history['epoch_metrics'][-1])
    reg_model, reg_history, reg_model_path, reg_history_path = full_train(
        'regularised', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], dropout_prob=cfg['reg_dropout']
    )
    print('\nRegularised model:')
    print(reg_model)
    print('\nRegular final epoch metrics:')
    print(reg_history['epoch_metrics'][-1])

if __name__ == '__main__':
    main()