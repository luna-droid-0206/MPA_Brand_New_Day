"""
Colorization Decoder
====================
U-Net decoder with skip connections from the encoder.

4 upsampling blocks, each:
  1. Upsample spatial resolution by 2×
  2. Concatenate with corresponding encoder feature map (skip connection)
  3. Two 3×3 conv layers with BatchNorm + ReLU

Final layer outputs 2 channels (a, b) with Tanh activation → [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + skip concat + double conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels:   Channels from previous decoder block (after upsample)
            skip_channels: Channels from the skip connection (encoder feature)
            out_channels:  Output channels of this block
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x:    (B, in_channels, H, W) — upsampled feature from previous block
            skip: (B, skip_channels, H', W') — encoder skip connection

        Returns:
            (B, out_channels, H', W')
        """
        # Upsample x to match skip's spatial dimensions
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate along channel axis
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class ColorizationDecoder(nn.Module):
    """
    U-Net decoder for colorization.
    Takes multi-scale encoder features and reconstructs ab channels.
    """

    def __init__(self):
        super().__init__()

        # Decoder blocks: process from deepest to shallowest
        # feat4 (512) + feat3 (256) → 256
        self.dec4 = DecoderBlock(512, 256, 256)
        # dec4 (256) + feat2 (128) → 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # dec3 (128) + feat1 (64) → 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # dec2 (64) + feat0 (64) → 64
        self.dec1 = DecoderBlock(64, 64, 64)

        # Final output layer: 64 → 2 (a and b channels)
        self.final = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh(),  # Output in [-1, 1] matching normalized ab range
        )

    def forward(self, encoder_features):
        """
        Decode from encoder features to ab prediction.

        Args:
            encoder_features: dict with keys 'feat0' through 'feat4'

        Returns:
            (B, 2, H, W) — predicted a and b channels, normalized to [-1, 1]
        """
        feat0 = encoder_features["feat0"]  # (B, 64, H/2, W/2)
        feat1 = encoder_features["feat1"]  # (B, 64, H/4, W/4)
        feat2 = encoder_features["feat2"]  # (B, 128, H/8, W/8)
        feat3 = encoder_features["feat3"]  # (B, 256, H/16, W/16)
        feat4 = encoder_features["feat4"]  # (B, 512, H/32, W/32)

        x = self.dec4(feat4, feat3)   # (B, 256, H/16, W/16)
        x = self.dec3(x, feat2)       # (B, 128, H/8, W/8)
        x = self.dec2(x, feat1)       # (B, 64, H/4, W/4)
        x = self.dec1(x, feat0)       # (B, 64, H/2, W/2)

        # Upsample to original resolution
        # The encoder halves resolution at conv1 (stride=2), so we need to
        # upsample one more time to get back to input resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.final(x)          # (B, 2, H, W)
