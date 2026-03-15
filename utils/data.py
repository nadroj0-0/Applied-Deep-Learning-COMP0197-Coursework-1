import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

class Cutout:
    """
    Randomly masks a square patch of an image tensor during training.
    Forces the network to learn distributed representations rather than
    relying on a single discriminative region.
    Args:
        size (int): Side length of the square patch to zero out.
    """
    def __init__(self, size=16):
        self.size = size
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        cx = torch.randint(0, w, (1,)).item()
        cy = torch.randint(0, h, (1,)).item()
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

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


def download_data(augment=False):
    # Download the data
    print('Downloading CIFAR-10 dataset...')
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Cutout(size=16)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
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