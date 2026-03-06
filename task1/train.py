import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
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
            # MaxPool to downsample the spatial dimensions.
            # kernel_size=2 halves height and width, (batch, 32, 32, 32) → (batch, 32, 16, 16)
            nn.MaxPool2d(2),
            # Second convolution layer
            # Input channels = 32 (from previous layer), Output channels = 64 feature maps.
            # Input:  (batch, 32, 16, 16), Output: (batch, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Second spatial downsampling
            # (batch, 64, 16, 16) → (batch, 64, 8, 8)
            nn.MaxPool2d(2)
        )

        # Fully connected classifier
        # After convolutions do classical neural network layers.
        self.fc_layers = nn.Sequential(
            # Flatten converts the 3D feature map into a vector.
            # (batch, 64, 8, 8) → (batch, 64 * 8 * 8), 64 * 8 * 8 = 4096 features
            nn.Flatten(),
            # First dense (fully connected) layer
            # Input size = 4096, Output size = 256 hidden units
            # This layer learns combinations of the spatial features discovered by the convolutional layers.
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            # Second classification layer
            # 256 → 128 outputs
            # This layer learns combinations of the 256 features from the previous layer.
            nn.Linear(256, 128),
            nn.ReLU(),
            # Final classification layer
            # 128 → 10 outputs
            # CIFAR-10 has 10 classes, so the network outputs, 10 class scores.
            nn.Linear(128, 10)
        )

    # Defines how data flows through the network.
    def forward(self, x):
        # Pass input images through convolutional feature extractor, x shape: (batch, 3, 32, 32) → (batch, 64, 8, 8)
        x = self.conv_layers(x)
        # Pass the extracted features into the fully connected classifier.
        # Each row corresponds to the class scores for one image, x shape: (batch, 64, 8, 8) → (batch, 10)
        x = self.fc_layers(x)
        return x

def main():
    #Download the data
    print("Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
    test_dataset = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
    print("Dataset downloaded successfully.")

    #Load the data into PyTorch
    print("Loading dataset into PyTorch...")
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

    #Inspect a few samples of the data
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    image = images[0] #Sample 1 image
    label = labels[0] #Sample 1 label
    print("First image tensor shape:", image.shape)
    print("First label:", label)
    print("Min pixel value:", image.min().item())
    print("Max pixel value:", image.max().item())
    #Inspect the different labels
    classes = train_dataset.classes
    print("Classes:", classes)


    #Create the model
    print("\nCreating model...")
    model = SimpleCNN()
    print(model)
    #Test a forward pass
    print("\nTesting forward pass...")
    outputs = model(images)
    print("Model output shape:", outputs.shape)


    #Loss function
    print("\nCreating loss function...")
    criterion = nn.CrossEntropyLoss()
    # Test loss calculation on the current batch
    loss = criterion(outputs, labels)
    print("Initial loss:", loss.item())


    # Optimiser
    print("\nCreating optimiser...")
    optimMethod = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Optimiser created:", optimMethod)

    #Training
    print("\nStarting training...")
    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimMethod.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimMethod.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0
    print("Training finished.")

if __name__ == "__main__":
    main()