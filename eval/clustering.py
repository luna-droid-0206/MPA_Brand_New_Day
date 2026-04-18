"""
Image Clustering Evaluation
============================
Evaluate learned representations via K-Means clustering and t-SNE.

Protocol:
  1. Standardize embeddings
  2. K-Means with K=10 (matching STL-10 class count), n_init=10
  3. Report Silhouette Score and Adjusted Rand Index
  4. t-SNE visualization (3,000 samples, perplexity=30)
  5. Side-by-side: true labels vs. K-Means assignments
  6. Compare: random encoder vs. pretrained encoder

Usage:
    python -m eval.clustering [--config configs/config.yaml]
"""

import os
import sys
import argparse
import json
import yaml

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.representation_metrics import compute_clustering_metrics
from utils.visualization import (
    plot_tsne,
    plot_tsne_comparison,
)

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


def run_kmeans(embeddings, k=10, n_init=10, random_state=42):
    """
    Run K-Means clustering.

    Returns:
        cluster_labels: Array of cluster assignments
        kmeans: Fitted KMeans model
    """
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans


def run_tsne(embeddings, n_samples=3000, perplexity=30, random_state=42):
    """
    Run t-SNE dimensionality reduction.

    If n_samples < len(embeddings), subsample randomly.

    Returns:
        embeddings_2d: (n_samples, 2) array
        indices: Indices of the selected samples
    """
    n_total = len(embeddings)
    if n_samples < n_total:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total, n_samples, replace=False)
    else:
        indices = np.arange(n_total)

    subset = embeddings[indices]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    embeddings_2d = tsne.fit_transform(subset)

    return embeddings_2d, indices


def run_clustering_eval(embeddings_dir, results_dir, config):
    """Run the full clustering evaluation pipeline."""

    os.makedirs(results_dir, exist_ok=True)

    # Load embeddings
    test_emb = np.load(os.path.join(embeddings_dir, "test_embeddings.npy"))
    test_labels = np.load(os.path.join(embeddings_dir, "test_labels.npy"))
    random_test_emb = np.load(os.path.join(embeddings_dir, "random_test_embeddings.npy"))

    cluster_cfg = config["evaluation"]["clustering"]
    tsne_cfg = config["evaluation"]["tsne"]

    # Standardize
    scaler = StandardScaler()
    test_emb_scaled = scaler.fit_transform(test_emb)

    scaler_random = StandardScaler()
    random_test_emb_scaled = scaler_random.fit_transform(random_test_emb)

    results = {}

    # ── Pretrained: K-Means ──
    print("\n" + "="*50)
    print("  PRETRAINED ENCODER — K-Means Clustering")
    print("="*50)

    cluster_labels, _ = run_kmeans(
        test_emb_scaled,
        k=cluster_cfg["k"],
        n_init=cluster_cfg["n_init"],
    )

    cluster_metrics = compute_clustering_metrics(
        test_labels, cluster_labels, test_emb_scaled
    )

    sil_score = cluster_metrics['silhouette_score']
    sil_str = f"{sil_score:.4f}" if sil_score is not None else "N/A (only 1 cluster found)"
    print(f"  Silhouette Score:    {sil_str}")
    print(f"  Adjusted Rand Index: {cluster_metrics['adjusted_rand_index']:.4f}")

    results["pretrained"] = cluster_metrics

    # ── Random: K-Means ──
    print("\n" + "="*50)
    print("  RANDOM ENCODER — K-Means Clustering (Baseline)")
    print("="*50)

    random_cluster_labels, _ = run_kmeans(
        random_test_emb_scaled,
        k=cluster_cfg["k"],
        n_init=cluster_cfg["n_init"],
    )

    random_cluster_metrics = compute_clustering_metrics(
        test_labels, random_cluster_labels, random_test_emb_scaled
    )

    sil_score_random = random_cluster_metrics['silhouette_score']
    sil_str_random = f"{sil_score_random:.4f}" if sil_score_random is not None else "N/A (only 1 cluster found)"
    print(f"  Silhouette Score:    {sil_str_random}")
    print(f"  Adjusted Rand Index: {random_cluster_metrics['adjusted_rand_index']:.4f}")

    results["random"] = random_cluster_metrics

    # ── t-SNE Visualization ──
    print("\n  [t-SNE] Computing t-SNE for pretrained encoder...")
    pretrained_2d, idx_pretrained = run_tsne(
        test_emb_scaled,
        n_samples=tsne_cfg["n_samples"],
        perplexity=tsne_cfg["perplexity"],
        random_state=tsne_cfg["random_state"],
    )

    # t-SNE plot — true labels
    plot_tsne(
        pretrained_2d, test_labels[idx_pretrained],
        os.path.join(results_dir, "tsne_pretrained_true_labels.png"),
        title="t-SNE — Pretrained Encoder (True Labels)",
        class_names=CLASS_NAMES,
    )

    # t-SNE plot — K-Means assignments
    plot_tsne(
        pretrained_2d, cluster_labels[idx_pretrained],
        os.path.join(results_dir, "tsne_pretrained_kmeans.png"),
        title="t-SNE — Pretrained Encoder (K-Means Clusters)",
    )

    print("  [t-SNE] Computing t-SNE for random encoder...")
    random_2d, idx_random = run_tsne(
        random_test_emb_scaled,
        n_samples=tsne_cfg["n_samples"],
        perplexity=tsne_cfg["perplexity"],
        random_state=tsne_cfg["random_state"],
    )

    # Side-by-side comparison
    # Use same indices for fair comparison if possible
    common_n = min(len(idx_pretrained), len(idx_random))
    plot_tsne_comparison(
        random_2d[:common_n], pretrained_2d[:common_n],
        test_labels[idx_pretrained[:common_n]],
        os.path.join(results_dir, "tsne_comparison.png"),
        class_names=CLASS_NAMES,
    )

    # ── Summary ──
    print("\n" + "="*60)
    print("  CLUSTERING RESULTS SUMMARY")
    print("="*60)
    for method, metrics in results.items():
        sil = metrics.get("silhouette_score", "N/A")
        ari = metrics.get("adjusted_rand_index", "N/A")
        if isinstance(sil, float):
            sil = f"{sil:.4f}"
        if isinstance(ari, float):
            ari = f"{ari:.4f}"
        print(f"  {method:20s}  Silhouette: {sil}  ARI: {ari}")
    print("="*60)

    # Save
    results_serializable = {}
    for k, v in results.items():
        results_serializable[k] = {mk: float(mv) for mk, mv in v.items()}

    with open(os.path.join(results_dir, "clustering_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Clustering Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    embeddings_dir = os.path.join(config["pretraining"]["checkpoint_dir"], "embeddings")
    results_dir = os.path.join(config["pretraining"]["log_dir"], "clustering")

    run_clustering_eval(embeddings_dir, results_dir, config)


if __name__ == "__main__":
    main()
