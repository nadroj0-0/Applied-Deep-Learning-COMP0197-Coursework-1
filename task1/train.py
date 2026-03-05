import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    print("Downloading CIFAR-10 dataset...")

    datasets.CIFAR10(root="./data", train=True, download=True)
    datasets.CIFAR10(root="./data", train=False, download=True)

    print("Dataset downloaded successfully.")
    print("Loading dataset into PyTorch...")

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    images, labels = next(iter(train_loader))

    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    image = images[0]
    label = labels[0]

    print("First image tensor shape:", image.shape)
    print("First label:", label)

    print("Min pixel value:", image.min().item())
    print("Max pixel value:", image.max().item())

    classes = train_dataset.classes

if __name__ == "__main__":
    main()