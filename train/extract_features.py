"""
Feature Extraction Script
=========================
Extract embeddings from the pretrained encoder for downstream evaluation.

Takes all labeled STL-10 images through the frozen encoder and saves
512-d feature vectors as numpy arrays.

Also extracts embeddings from a randomly initialized encoder for baseline.

Usage:
    python -m train.extract_features [--config configs/config.yaml]
"""

import os
import sys
import argparse
import yaml

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder import ColorizationEncoder
from datasets.colorization_dataset import get_stl10_labeled_loaders
from utils.checkpoints import load_encoder


def extract_embeddings(encoder, loader, device):
    """
    Extract embeddings from all samples in a DataLoader.

    Args:
        encoder: ColorizationEncoder in eval mode
        loader:  DataLoader yielding (L, ab, label)
        device:  torch device

    Returns:
        embeddings: (N, 512) numpy array
        labels:     (N,) numpy array of integer labels
    """
    encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for L, ab, label in tqdm(loader, desc="Extracting embeddings"):
            L = L.to(device)
            emb = encoder.extract_embedding(L)  # (B, 512)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(label.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from encoder")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Path to pretrained encoder. Defaults to checkpoints/best_encoder.pth")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Feature Extraction")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Data
    train_loader, test_loader, class_names = get_stl10_labeled_loaders(
        data_dir=config["dataset"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["pretraining"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    # ── Pretrained encoder embeddings ──
    encoder_path = args.encoder_path or os.path.join(
        config["pretraining"]["checkpoint_dir"], "best_encoder.pth"
    )

    encoder = ColorizationEncoder(pretrained=False).to(device)
    load_encoder(encoder, encoder_path)

    print("\n[EXTRACT] Extracting pretrained encoder embeddings...")
    train_emb, train_labels = extract_embeddings(encoder, train_loader, device)
    test_emb, test_labels = extract_embeddings(encoder, test_loader, device)

    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Test embeddings:  {test_emb.shape}")

    # Save
    emb_dir = os.path.join(config["pretraining"]["checkpoint_dir"], "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    np.save(os.path.join(emb_dir, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(emb_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(emb_dir, "test_embeddings.npy"), test_emb)
    np.save(os.path.join(emb_dir, "test_labels.npy"), test_labels)
    print(f"  Saved to {emb_dir}/")

    # ── Random encoder embeddings (baseline) ──
    print("\n[EXTRACT] Extracting RANDOM encoder embeddings (baseline)...")
    random_encoder = ColorizationEncoder(pretrained=False).to(device)
    # No loading — uses random initialization

    random_train_emb, _ = extract_embeddings(random_encoder, train_loader, device)
    random_test_emb, _ = extract_embeddings(random_encoder, test_loader, device)

    print(f"  Random train embeddings: {random_train_emb.shape}")
    print(f"  Random test embeddings:  {random_test_emb.shape}")

    np.save(os.path.join(emb_dir, "random_train_embeddings.npy"), random_train_emb)
    np.save(os.path.join(emb_dir, "random_test_embeddings.npy"), random_test_emb)
    print(f"  Saved random embeddings to {emb_dir}/")

    print(f"\n{'='*60}")
    print(f"  Feature extraction complete!")
    print(f"  Files saved to: {emb_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
