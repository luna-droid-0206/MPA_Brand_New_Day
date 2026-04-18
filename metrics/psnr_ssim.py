"""
PSNR and SSIM Metrics
=====================
Pixel-level fidelity metrics for evaluating colorization quality.

PSNR (Peak Signal-to-Noise Ratio): Higher is better. Target: 22–26 dB.
SSIM (Structural Similarity Index): Higher is better. Target: 0.75–0.88.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_psnr(img_true: np.ndarray, img_pred: np.ndarray,
                  data_range: float = 2.0) -> float:
    """
    Compute PSNR between two images.

    Args:
        img_true: Ground truth image (any shape)
        img_pred: Predicted image (same shape)
        data_range: Range of the data (2.0 for [-1, 1] normalized)

    Returns:
        PSNR value in dB
    """
    return psnr(img_true, img_pred, data_range=data_range)


def compute_ssim(img_true: np.ndarray, img_pred: np.ndarray,
                  data_range: float = 2.0) -> float:
    """
    Compute SSIM between two images.

    Args:
        img_true: Ground truth image (H, W) or (H, W, C)
        img_pred: Predicted image (same shape)
        data_range: Range of the data (2.0 for [-1, 1] normalized)

    Returns:
        SSIM value in [0, 1]
    """
    # Determine if multi-channel
    channel_axis = -1 if img_true.ndim == 3 else None
    win_size = min(7, min(img_true.shape[0], img_true.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    return ssim(img_true, img_pred, data_range=data_range,
                channel_axis=channel_axis, win_size=win_size)


def compute_batch_psnr(ab_true: torch.Tensor, ab_pred: torch.Tensor) -> float:
    """
    Compute average PSNR over a batch of ab channel predictions.

    Args:
        ab_true: (B, 2, H, W) ground truth ab (normalized [-1, 1])
        ab_pred: (B, 2, H, W) predicted ab (normalized [-1, 1])

    Returns:
        Average PSNR across the batch
    """
    ab_true_np = ab_true.detach().cpu().numpy()
    ab_pred_np = ab_pred.detach().cpu().numpy()

    batch_psnr = 0.0
    B = ab_true_np.shape[0]

    for i in range(B):
        # Transpose to (H, W, 2) for skimage
        true_i = ab_true_np[i].transpose(1, 2, 0)
        pred_i = ab_pred_np[i].transpose(1, 2, 0)
        batch_psnr += compute_psnr(true_i, pred_i)

    return batch_psnr / B


def compute_batch_ssim(ab_true: torch.Tensor, ab_pred: torch.Tensor) -> float:
    """
    Compute average SSIM over a batch of ab channel predictions.

    Args:
        ab_true: (B, 2, H, W) ground truth ab (normalized [-1, 1])
        ab_pred: (B, 2, H, W) predicted ab (normalized [-1, 1])

    Returns:
        Average SSIM across the batch
    """
    ab_true_np = ab_true.detach().cpu().numpy()
    ab_pred_np = ab_pred.detach().cpu().numpy()

    batch_ssim = 0.0
    B = ab_true_np.shape[0]

    for i in range(B):
        true_i = ab_true_np[i].transpose(1, 2, 0)
        pred_i = ab_pred_np[i].transpose(1, 2, 0)
        batch_ssim += compute_ssim(true_i, pred_i)

    return batch_ssim / B
