"""
Image Retrieval Evaluation
===========================
Evaluate learned representations via nearest-neighbor image retrieval.

Protocol:
  1. L2-normalize all embeddings
  2. For each query image, compute cosine similarity with all others
  3. Return top-K most similar images (K=9 for 3×3 grid)
  4. Compute Precision@K for quantitative evaluation
  5. Generate visual retrieval grids

Usage:
    python -m eval.retrieval [--config configs/config.yaml]
"""

import os
import sys
import argparse
import json
import yaml

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.representation_metrics import compute_retrieval_precision, compute_mean_precision_at_k
from utils.visualization import plot_retrieval_grid
from utils.lab_utils import rgb_to_lab, normalize_l

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


def l2_normalize(embeddings):
    """L2-normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embeddings / norms


def cosine_similarity_matrix(embeddings):
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, D) L2-normalized

    Returns:
        (N, N) similarity matrix
    """
    return embeddings @ embeddings.T


def retrieve_top_k(query_idx, similarity_matrix, k=9):
    """
    Get top-K most similar images for a query (excluding itself).

    Returns:
        List of indices of the K most similar images
    """
    sims = similarity_matrix[query_idx].copy()
    sims[query_idx] = -1.0  # Exclude self
    top_k_indices = np.argsort(sims)[::-1][:k]
    return top_k_indices


def run_retrieval_eval(embeddings_dir, results_dir, config, data_dir="./data"):
    """Run the full retrieval evaluation pipeline."""

    os.makedirs(results_dir, exist_ok=True)

    retrieval_cfg = config["evaluation"]["retrieval"]
    top_k = retrieval_cfg["top_k"]

    # Load embeddings
    test_emb = np.load(os.path.join(embeddings_dir, "test_embeddings.npy"))
    test_labels = np.load(os.path.join(embeddings_dir, "test_labels.npy"))

    # L2 normalize
    test_emb_norm = l2_normalize(test_emb)

    # Similarity matrix
    print("\n[RETRIEVAL] Computing similarity matrix...")
    sim_matrix = cosine_similarity_matrix(test_emb_norm)

    # ── Precision@K ──
    print(f"[RETRIEVAL] Computing Precision@{top_k} for all test images...")

    all_retrieved_labels = []
    for i in range(len(test_emb)):
        top_k_idx = retrieve_top_k(i, sim_matrix, k=top_k)
        all_retrieved_labels.append(test_labels[top_k_idx])

    mean_p_at_5 = compute_mean_precision_at_k(test_labels, all_retrieved_labels, k=5)
    mean_p_at_k = compute_mean_precision_at_k(test_labels, all_retrieved_labels, k=top_k)

    print(f"  Mean Precision@5: {mean_p_at_5:.4f}")
    print(f"  Mean Precision@{top_k}: {mean_p_at_k:.4f}")

    # ── Visual Retrieval Grids ──
    print("\n[RETRIEVAL] Generating retrieval grids...")

    # Load raw test images for visualization
    stl10_test = torchvision.datasets.STL10(root=data_dir, split="test", download=True)

    # Pick diverse query samples — one per class
    query_indices = []
    for class_idx in range(10):
        class_mask = np.where(test_labels == class_idx)[0]
        if len(class_mask) > 0:
            query_indices.append(class_mask[0])

    for qi in query_indices:
        query_img = np.array(stl10_test[qi][0].convert("RGB").resize((96, 96)))
        query_label = test_labels[qi]

        top_k_idx = retrieve_top_k(qi, sim_matrix, k=top_k)
        retrieved_imgs = [
            np.array(stl10_test[idx][0].convert("RGB").resize((96, 96)))
            for idx in top_k_idx
        ]
        retrieved_labels = test_labels[top_k_idx]

        class_name = CLASS_NAMES[query_label]
        plot_retrieval_grid(
            query_img, retrieved_imgs, retrieved_labels.tolist(), query_label,
            save_path=os.path.join(results_dir, f"retrieval_{class_name}.png"),
            class_names=CLASS_NAMES,
        )

    # ── Per-class Precision ──
    per_class_precision = {}
    for class_idx in range(10):
        class_mask = np.where(test_labels == class_idx)[0]
        class_precisions = []
        for i in class_mask:
            p = compute_retrieval_precision(test_labels[i], all_retrieved_labels[i], k=top_k)
            class_precisions.append(p)
        per_class_precision[CLASS_NAMES[class_idx]] = float(np.mean(class_precisions))

    # ── Summary ──
    print("\n" + "="*60)
    print("  RETRIEVAL RESULTS SUMMARY")
    print("="*60)
    print(f"  Mean Precision@5:  {mean_p_at_5:.4f}")
    print(f"  Mean Precision@{top_k}: {mean_p_at_k:.4f}")
    print(f"\n  Per-class Precision@{top_k}:")
    for name, prec in per_class_precision.items():
        print(f"    {name:12s}: {prec:.4f}")
    print("="*60)

    results = {
        "mean_precision_at_5": float(mean_p_at_5),
        f"mean_precision_at_{top_k}": float(mean_p_at_k),
        "per_class_precision": per_class_precision,
    }

    with open(os.path.join(results_dir, "retrieval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    embeddings_dir = os.path.join(config["pretraining"]["checkpoint_dir"], "embeddings")
    results_dir = os.path.join(config["pretraining"]["log_dir"], "retrieval")

    run_retrieval_eval(embeddings_dir, results_dir, config,
                        data_dir=config["dataset"]["data_dir"])


if __name__ == "__main__":
    main()
