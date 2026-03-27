"""
GenAI was used to assist with structuring and refining the model architecture.
All architectural choices, implementation details, and design decisions were verified and adapted independently.
"""
import torch.nn as nn

class SEBlock(nn.Module):
    """
        Squeeze-and-Excitation block for channel-wise attention.

        Applies global average pooling followed by a small fully connected
        network to generate channel-wise weights, which are used to rescale
        the input feature maps.
    """
    def __init__(self, channels, reduction):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
        Residual block with two convolutional layers and SE attention.

        Implements a skip connection to preserve input information while
        allowing the network to learn residual features, improving training
        stability and representation quality.
    """
    def __init__(self, channels, reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out

class CNN(nn.Module):
    """
        Convolutional neural network with residual connections and SE attention.

        The architecture consists of:
        - Initial convolutional layers for feature extraction
        - Residual blocks with channel attention
        - Progressive downsampling via max pooling
        - Global average pooling and a linear classifier

        Designed for image classification on CIFAR-10.
    """
    def __init__(self, dropout_prob = 0.0):
        super().__init__()
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Initial feature extractor
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # Residual block at 64 channels
            ResidualBlock(64, reduction=8),
            nn.MaxPool2d(2),  # 32x32 → 16x16
            # Channel expansion
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # Residual block at 128 channels
            ResidualBlock(128,  reduction=16),
            nn.MaxPool2d(2),  # 16x16 → 8x8
            # Channel expansion
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            # Residual block at 256 channels
            ResidualBlock(256,  reduction=16),
            nn.MaxPool2d(2)  # 8x8 → 4x4
        )

        # Fully connected classifier
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 10)
        )

    # Defines how data flows through the network.
    def forward(self, x):
        """
                Defines the forward pass of the network.

                Args:
                    x (Tensor): Input images of shape (batch, 3, 32, 32)

                Returns:
                    Tensor: Class logits of shape (batch, 10)
        """
        # Pass input images through convolutional feature extractor, x shape: (batch, 3, 32, 32) → (batch, 64, 8, 8)
        x = self.conv_layers(x)
        # Pass the extracted features into the fully connected classifier.
        # Each row corresponds to the class scores for one image, x shape: (batch, 64, 8, 8) → (batch, 10)
        x = self.fc_layers(x)
        return x