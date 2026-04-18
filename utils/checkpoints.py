"""
Checkpoint Utilities
====================
Save and load model checkpoints with training state.
"""

import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """
    Save a training checkpoint.

    Args:
        model:     The model (or nn.Module)
        optimizer: The optimizer
        scheduler: The LR scheduler (can be None)
        epoch:     Current epoch number
        loss:      Current loss value
        path:      File path to save to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"[CKPT] Saved checkpoint -> {path}  (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load a training checkpoint.

    Args:
        path:      File path to load from
        model:     The model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore

    Returns:
        dict with 'epoch' and 'loss' from the checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"[CKPT] Loaded checkpoint <- {path}  (epoch {checkpoint['epoch']})")
    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint.get("loss", None),
    }


def save_encoder(encoder, path):
    """Save only the encoder weights (for downstream use after pretraining)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(encoder.state_dict(), path)
    print(f"[CKPT] Saved encoder weights -> {path}")


def load_encoder(encoder, path):
    """Load encoder weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    print(f"[CKPT] Loaded encoder weights <- {path}")
    return encoder
