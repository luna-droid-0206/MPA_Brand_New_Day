"""
Colorization Encoder
====================
ResNet-18 backbone modified for single-channel (L) input.

The encoder outputs feature maps at 5 scales:
  - feat0: after initial conv+bn+relu (64 channels, 1/2 resolution)
  - feat1: after layer1 (64 channels, 1/4 resolution)
  - feat2: after layer2 (128 channels, 1/8 resolution)
  - feat3: after layer3 (256 channels, 1/16 resolution)
  - feat4: after layer4 (512 channels, 1/32 resolution)

These multi-scale features are used for U-Net skip connections.
After pretraining, only feat4 with GAP is used for embeddings (512-d).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ColorizationEncoder(nn.Module):
    """
    Modified ResNet-18 encoder for single-channel grayscale input.
    Returns multi-scale features for the U-Net decoder.
    """

    def __init__(self, pretrained=False):
        """
        Args:
            pretrained: If True, loads ImageNet weights (first conv modified).
                        If False, trains from scratch (our SSL setting).
        """
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Modify first conv layer: 3 channels → 1 channel (L only)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # If using pretrained weights, initialize by averaging the 3-channel weights
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1   # 64 channels
        self.layer2 = resnet.layer2   # 128 channels
        self.layer3 = resnet.layer3   # 256 channels
        self.layer4 = resnet.layer4   # 512 channels

        # Global Average Pooling for embedding extraction
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass returning multi-scale features.

        Args:
            x: (B, 1, H, W) — normalized L channel

        Returns:
            dict with keys 'feat0' through 'feat4', and 'embedding' (512-d)
        """
        feat0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x = self.maxpool(feat0)                       # (B, 64, H/4, W/4)

        feat1 = self.layer1(x)                        # (B, 64, H/4, W/4)
        feat2 = self.layer2(feat1)                    # (B, 128, H/8, W/8)
        feat3 = self.layer3(feat2)                    # (B, 256, H/16, W/16)
        feat4 = self.layer4(feat3)                    # (B, 512, H/32, W/32)

        # Global average pooling for embedding
        embedding = self.gap(feat4).flatten(1)         # (B, 512)

        return {
            "feat0": feat0,
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
            "feat4": feat4,
            "embedding": embedding,
        }

    def extract_embedding(self, x):
        """
        Extract only the 512-d embedding (no skip connections needed).
        More efficient than full forward when you only need embeddings.

        Args:
            x: (B, 1, H, W) — normalized L channel

        Returns:
            (B, 512) embedding tensor
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return x
