import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

MODEL_DIR = Path('models')
TRAIN_CONFIG = {
    'epochs': 10,
    'optimiser': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'reg_dropout': 0.5
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


def load_data_pytorch(train_dataset):
    # Load the data into PyTorch
    print('Loading dataset into PyTorch...')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(train_loader))
    return images, labels, train_loader


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
    model = CNN(dropout_prob)
    print(model)
    # Test a forward pass
    print('\nTesting forward pass...')
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


def train_model(epochs, train_loader, model, criterion, optim_method):
    # Training
    print('\nStarting training...')
    model.train()
    # num_epochs = 50
    batch_losses = []
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optim_method.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim_method.step()
            loss_value = loss.item()
            batch_losses.append({
                'epoch': epoch +1,
                'batch': i+1,
                'loss': loss_value
            })
            epoch_loss += loss_value
            num_batches += 1
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}')
    print('Training finished.')
    return batch_losses, epoch_losses

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

def full_train(name, images, labels, train_loader, method, epochs, dropout_prob=0.0, **kwargs):
    model, outputs = init_model(images, dropout_prob)
    criterion, loss = init_loss(outputs, labels)
    #optim_method = init_optimiser(model, 'SGD', lr=0.001, momentum=0.9)
    optim_method = init_optimiser(model, method, **kwargs)
    batch_losses, epoch_losses = train_model(epochs, train_loader, model, criterion, optim_method)
    save_model(model, name)
    return model, batch_losses, epoch_losses

def main():
    train_dataset, test_dataset = download_data()
    images, labels, train_loader = load_data_pytorch(train_dataset)
    inspect_data(images, labels, train_dataset)
    cfg = TRAIN_CONFIG
    base_model, base_batch_losses, base_epoch_losses = full_train(
        'baseline', images, labels, train_loader, cfg['optimiser'],
        epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum']
    )
    print('\nBase model:')
    print(base_model)
    print('\nBase epoch losses:')
    print(base_epoch_losses)
    reg_model, reg_batch_losses, reg_epoch_losses = full_train(
        'regularised', images, labels, train_loader, cfg['optimiser'],
        epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], dropout_prob=cfg['reg_dropout']
    )
    print('\nRegularised model:')
    print(reg_model)
    print('\nRegular epoch losses:')
    print(reg_epoch_losses)

if __name__ == '__main__':
    main()