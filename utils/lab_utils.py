"""
Lab Color Space Utilities
=========================
Handles RGB ↔ Lab conversions and normalization for the colorization pipeline.

Lab color space separates luminance (L) from chrominance (a, b):
  - L ∈ [0, 100]  → grayscale luminance
  - a ∈ [-128, 127] → green-red axis
  - b ∈ [-128, 127] → blue-yellow axis

We normalize L to [-1, 1] and ab to [-1, 1] for stable training.
"""

import numpy as np
import cv2
import torch
from PIL import Image


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (uint8, H×W×3) to Lab color space (float32, H×W×3).
    Returns L ∈ [0, 100], a ∈ [-128, 127], b ∈ [-128, 127].
    """
    # OpenCV expects BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    # OpenCV Lab ranges: L [0, 255], a [0, 255], b [0, 255] (offset by 128)
    # Convert to standard Lab ranges
    image_lab[:, :, 0] = image_lab[:, :, 0] * (100.0 / 255.0)        # L → [0, 100]
    image_lab[:, :, 1] = image_lab[:, :, 1] - 128.0                   # a → [-128, 127]
    image_lab[:, :, 2] = image_lab[:, :, 2] - 128.0                   # b → [-128, 127]
    return image_lab


def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """
    Convert a Lab image (float32, H×W×3) back to RGB (uint8, H×W×3).
    Expects L ∈ [0, 100], a ∈ [-128, 127], b ∈ [-128, 127].
    """
    lab = image_lab.copy()
    lab[:, :, 0] = lab[:, :, 0] * (255.0 / 100.0)                     # L → [0, 255]
    lab[:, :, 1] = lab[:, :, 1] + 128.0                                # a → [0, 255]
    lab[:, :, 2] = lab[:, :, 2] + 128.0                                # b → [0, 255]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def normalize_l(l_channel: np.ndarray) -> np.ndarray:
    """Normalize L channel from [0, 100] to [-1, 1]."""
    return (l_channel / 50.0) - 1.0


def denormalize_l(l_normalized: np.ndarray) -> np.ndarray:
    """Denormalize L channel from [-1, 1] back to [0, 100]."""
    return (l_normalized + 1.0) * 50.0


def normalize_ab(ab_channels: np.ndarray) -> np.ndarray:
    """Normalize ab channels from [-128, 127] to approximately [-1, 1]."""
    return ab_channels / 128.0


def denormalize_ab(ab_normalized: np.ndarray) -> np.ndarray:
    """Denormalize ab channels from [-1, 1] back to [-128, 127]."""
    return ab_normalized * 128.0


def tensor_lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """
    Convert L and ab tensors back to an RGB numpy image.

    Args:
        L:  Tensor of shape (1, H, W), normalized to [-1, 1]
        ab: Tensor of shape (2, H, W), normalized to [-1, 1]

    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3)
    """
    L_np = L.detach().cpu().numpy().squeeze(0)     # (H, W)
    ab_np = ab.detach().cpu().numpy()               # (2, H, W)

    # Denormalize
    L_denorm = denormalize_l(L_np)                  # [0, 100]
    ab_denorm = denormalize_ab(ab_np)               # [-128, 127]

    # Assemble Lab image (H, W, 3)
    lab_image = np.stack([
        L_denorm,
        ab_denorm[0],   # a channel
        ab_denorm[1],   # b channel
    ], axis=-1).astype(np.float32)

    return lab_to_rgb(lab_image)


def pil_to_rgb_array(pil_image: Image.Image, size: int = 96) -> np.ndarray:
    """Resize a PIL Image and convert to RGB numpy array."""
    pil_image = pil_image.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(pil_image, dtype=np.uint8)
