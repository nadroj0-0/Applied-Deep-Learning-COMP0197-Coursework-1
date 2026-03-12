import torch.nn as nn

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
            nn.BatchNorm2d(32),
            # ReLU activation introduces non-linearity.
            nn.ReLU(),
            # 2nd convolution layer
            # Input:  (batch, 32, 32, 32), Output: (batch, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # MaxPool to downsample the spatial dimensions.
            # kernel_size=2 halves height and width, (batch, 64, 32, 32) → (batch, 64, 16, 16)
            nn.MaxPool2d(2),
            # 3rd and 4th convolution layers
            # Input channels = 64 (from previous layer), Output channels = 128 feature maps.
            # Input:  (batch, 64, 16, 16), Output: (batch, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Second spatial downsampling
            # (batch, 128, 16, 16) → (batch, 128, 8, 8)
            nn.MaxPool2d(2),
            # 5th and 6th convolution layer
            # Input:  (batch, 128, 8, 8), Output: (batch, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
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