"""
Colorization U-Net
==================
Full U-Net combining encoder and decoder.

Architecture:
  L channel (1, H, W) → Encoder (ResNet-18) → Multi-scale features
                       → Decoder (skip connections) → ab prediction (2, H, W)

After pretraining:
  - Encoder is kept for embedding extraction
  - Decoder is discarded
"""

import torch
import torch.nn as nn

from .encoder import ColorizationEncoder
from .decoder import ColorizationDecoder


class ColorizationUNet(nn.Module):
    """Complete U-Net for the colorization pretext task."""

    def __init__(self, encoder_pretrained=False):
        """
        Args:
            encoder_pretrained: Whether to use ImageNet-pretrained ResNet-18
                                weights. Set to False for full self-supervised
                                learning (our main setting).
        """
        super().__init__()

        self.encoder = ColorizationEncoder(pretrained=encoder_pretrained)
        self.decoder = ColorizationDecoder()

    def forward(self, L):
        """
        Predict ab channels from L channel.

        Args:
            L: (B, 1, H, W) — normalized grayscale luminance ∈ [-1, 1]

        Returns:
            ab_pred: (B, 2, H, W) — predicted chrominance ∈ [-1, 1]
        """
        encoder_features = self.encoder(L)
        ab_pred = self.decoder(encoder_features)
        return ab_pred

    def get_encoder(self):
        """Return the encoder sub-module (for downstream use)."""
        return self.encoder

    @torch.no_grad()
    def colorize(self, L):
        """
        Convenience method for inference: predict ab from L.

        Args:
            L: (B, 1, H, W) normalized L channel

        Returns:
            ab_pred: (B, 2, H, W) predicted ab channels
        """
        self.eval()
        return self.forward(L)
