"""
Self-Supervised Pretraining Script
===================================
Trains the U-Net colorization model on STL-10 unlabeled images.

The goal is NOT to produce perfect colorizations, but to force the encoder
to learn rich visual representations through the pretext task.

Usage:
    python -m train.pretrain [--config configs/config.yaml]
"""

import os
import sys
import argparse
import time
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.unet import ColorizationUNet
from datasets.colorization_dataset import get_stl10_colorization_loaders
from utils.checkpoints import save_checkpoint, save_encoder
from utils.visualization import plot_loss_curve, plot_colorization_samples
from metrics.psnr_ssim import compute_batch_psnr, compute_batch_ssim


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for L, ab_true in pbar:
        L = L.to(device)
        ab_true = ab_true.to(device)

        # Forward
        ab_pred = model(L)
        loss = criterion(ab_pred, ab_true)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device, n_vis=8):
    """Run validation, compute loss and PSNR/SSIM, return sample batches for viz."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    vis_L, vis_ab_true, vis_ab_pred = None, None, None

    for L, ab_true in loader:
        L = L.to(device)
        ab_true = ab_true.to(device)

        ab_pred = model(L)
        loss = criterion(ab_pred, ab_true)

        total_loss += loss.item()
        total_psnr += compute_batch_psnr(ab_true, ab_pred)
        total_ssim += compute_batch_ssim(ab_true, ab_pred)
        num_batches += 1

        # Save first batch for visualization
        if vis_L is None:
            vis_L = L[:n_vis].cpu()
            vis_ab_true = ab_true[:n_vis].cpu()
            vis_ab_pred = ab_pred[:n_vis].cpu()

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    return avg_loss, avg_psnr, avg_ssim, vis_L, vis_ab_true, vis_ab_pred


def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Colorization Pretraining")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Self-Supervised Colorization Pretraining")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Directories
    ckpt_dir = config["pretraining"]["checkpoint_dir"]
    log_dir = config["pretraining"]["log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Data
    train_loader = get_stl10_colorization_loaders(
        data_dir=config["dataset"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["pretraining"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    # Model
    model = ColorizationUNet(
        encoder_pretrained=config["model"]["encoder_pretrained"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Total parameters: {total_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["pretraining"]["learning_rate"],
        weight_decay=config["pretraining"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["pretraining"]["epochs"],
    )

    # Training
    epochs = config["pretraining"]["epochs"]
    ckpt_every = config["pretraining"]["checkpoint_every"]
    vis_every = config["pretraining"]["visualize_every"]
    grad_clip = config["pretraining"]["grad_clip_max_norm"]

    all_losses = []
    best_loss = float("inf")

    print(f"\n[TRAIN] Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, device, grad_clip)
        scheduler.step()

        all_losses.append(train_loss)
        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{epochs}  │  "
              f"Loss: {train_loss:.6f}  │  "
              f"LR: {lr_current:.2e}  │  "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss,
                            os.path.join(ckpt_dir, "best_model.pth"))
            save_encoder(model.get_encoder(),
                         os.path.join(ckpt_dir, "best_encoder.pth"))

        # Periodic checkpoint
        if epoch % ckpt_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss,
                            os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth"))

        # Periodic visualization
        if epoch % vis_every == 0:
            _, psnr, ssim, vis_L, vis_ab_true, vis_ab_pred = validate(
                model, train_loader, criterion, device
            )
            print(f"           └─ PSNR: {psnr:.2f} dB  │  SSIM: {ssim:.4f}")

            plot_colorization_samples(
                vis_L, vis_ab_true, vis_ab_pred,
                save_path=os.path.join(log_dir, f"colorization_epoch_{epoch}.png"),
            )

    # Final save
    save_checkpoint(model, optimizer, scheduler, epochs, all_losses[-1],
                    os.path.join(ckpt_dir, "final_model.pth"))
    save_encoder(model.get_encoder(),
                 os.path.join(ckpt_dir, "final_encoder.pth"))

    # Plot training loss curve
    plot_loss_curve(all_losses, os.path.join(log_dir, "training_loss.png"))

    print(f"\n{'='*60}")
    print(f"  Pretraining complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Logs: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
