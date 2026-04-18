"""
Visualization Utilities
=======================
Functions for plotting colorization outputs, t-SNE maps,
loss curves, retrieval grids, and confusion matrices.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.lab_utils import tensor_lab_to_rgb, denormalize_l, denormalize_ab, lab_to_rgb


def plot_loss_curve(losses: list, save_path: str, title: str = "Training Loss"):
    """Plot and save the training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(losses) + 1), losses, color="#C8A96E", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#0D0D0D")
    ax.tick_params(colors="#888580")
    ax.xaxis.label.set_color("#F0EDE6")
    ax.yaxis.label.set_color("#F0EDE6")
    ax.title.set_color("#F0EDE6")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Loss curve saved -> {save_path}")


def plot_colorization_samples(L_batch, ab_true_batch, ab_pred_batch,
                               save_path: str, n_samples: int = 8):
    """
    Plot side-by-side comparison: Grayscale | Ground Truth | Predicted.

    Args:
        L_batch:       (N, 1, H, W) normalized L channel
        ab_true_batch: (N, 2, H, W) normalized true ab
        ab_pred_batch: (N, 2, H, W) normalized predicted ab
        save_path:     Path to save the figure
        n_samples:     Number of samples to display
    """
    n = min(n_samples, L_batch.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    fig.patch.set_facecolor("#0D0D0D")

    if n == 1:
        axes = axes[np.newaxis, :]

    column_titles = ["Grayscale (Input)", "Ground Truth", "Predicted"]
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, color="#F0EDE6", fontsize=11, fontweight="bold")

    for i in range(n):
        L_np = L_batch[i].detach().cpu().numpy().squeeze(0)         # (H, W)
        ab_true_np = ab_true_batch[i].detach().cpu().numpy()        # (2, H, W)
        ab_pred_np = ab_pred_batch[i].detach().cpu().numpy()        # (2, H, W)

        # Grayscale
        L_display = denormalize_l(L_np)  # [0, 100]
        axes[i, 0].imshow(L_display, cmap="gray", vmin=0, vmax=100)

        # Ground truth RGB
        lab_gt = np.stack([L_display, denormalize_ab(ab_true_np[0]),
                           denormalize_ab(ab_true_np[1])], axis=-1).astype(np.float32)
        axes[i, 1].imshow(lab_to_rgb(lab_gt))

        # Predicted RGB
        lab_pred = np.stack([L_display, denormalize_ab(ab_pred_np[0]),
                             denormalize_ab(ab_pred_np[1])], axis=-1).astype(np.float32)
        axes[i, 2].imshow(lab_to_rgb(lab_pred))

        for j in range(3):
            axes[i, j].axis("off")

    fig.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Colorization samples saved -> {save_path}")


def plot_tsne(embeddings_2d: np.ndarray, labels: np.ndarray,
              save_path: str, title: str = "t-SNE Visualization",
              class_names: list = None):
    """
    Plot a 2D t-SNE embedding colored by class labels.

    Args:
        embeddings_2d: (N, 2) array of t-SNE coordinates
        labels:        (N,) array of integer labels
        save_path:     Path to save the figure
        title:         Plot title
        class_names:   Optional list of names for each class
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#161616")

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else f"Class {label}"
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[cmap(idx)], s=8, alpha=0.7, label=name)

    ax.legend(fontsize=8, loc="best", framealpha=0.7,
              facecolor="#1E1E1E", edgecolor="#555250", labelcolor="#F0EDE6")
    ax.set_title(title, color="#F0EDE6", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#888580")
    ax.set_xlabel("t-SNE 1", color="#888580")
    ax.set_ylabel("t-SNE 2", color="#888580")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] t-SNE plot saved -> {save_path}")


def plot_tsne_comparison(emb_random_2d, emb_pretrained_2d, labels,
                          save_path: str, class_names: list = None):
    """
    Side-by-side t-SNE: random encoder vs. pretrained encoder.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0D0D0D")

    for ax, emb_2d, title_text in [
        (ax1, emb_random_2d, "Random Encoder"),
        (ax2, emb_pretrained_2d, "Pretrained Encoder (Ours)"),
    ]:
        ax.set_facecolor("#161616")
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names[label] if class_names else f"Class {label}"
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[cmap(idx)], s=8, alpha=0.6, label=name)

        ax.set_title(title_text, color="#F0EDE6", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#888580")
        ax.legend(fontsize=7, loc="best", framealpha=0.5,
                  facecolor="#1E1E1E", edgecolor="#555250", labelcolor="#F0EDE6")

    fig.suptitle("t-SNE: Random vs. Self-Supervised Representations",
                 color="#C8A96E", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] t-SNE comparison saved -> {save_path}")


def plot_retrieval_grid(query_img: np.ndarray, retrieved_imgs: list,
                         retrieved_labels: list, query_label: int,
                         save_path: str, class_names: list = None):
    """
    Plot a retrieval grid: query on the left, top-K results in a 3×3 grid.

    Args:
        query_img:        RGB numpy array (H, W, 3)
        retrieved_imgs:   List of RGB numpy arrays [(H, W, 3), ...]
        retrieved_labels: List of integer labels for retrieved images
        query_label:      Integer label of the query image
        save_path:        Path to save the figure
        class_names:      Optional list of class names
    """
    k = len(retrieved_imgs)
    cols = 4
    rows = (k + 1 + cols - 1) // cols  # +1 for query

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.patch.set_facecolor("#0D0D0D")

    if rows == 1:
        axes = axes[np.newaxis, :]

    # Flatten for easy indexing
    all_axes = axes.flatten()
    for ax in all_axes:
        ax.axis("off")
        ax.set_facecolor("#0D0D0D")

    # Query image
    all_axes[0].imshow(query_img)
    q_name = class_names[query_label] if class_names else f"Class {query_label}"
    all_axes[0].set_title(f"QUERY\n{q_name}", color="#C8A96E", fontsize=9, fontweight="bold")
    all_axes[0].patch.set_edgecolor("#C8A96E")
    all_axes[0].patch.set_linewidth(2)

    # Retrieved images
    for i, (img, lbl) in enumerate(zip(retrieved_imgs, retrieved_labels)):
        ax = all_axes[i + 1]
        ax.imshow(img)
        is_correct = (lbl == query_label)
        color = "#7EC8A4" if is_correct else "#888580"
        marker = "✓" if is_correct else "✗"
        lbl_name = class_names[lbl] if class_names else f"Class {lbl}"
        ax.set_title(f"{marker} {lbl_name}", color=color, fontsize=8)

    fig.suptitle("Image Retrieval Results", color="#F0EDE6",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Retrieval grid saved -> {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str,
                           title: str = "Confusion Matrix"):
    """Plot a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#161616")

    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrBr")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#888580")

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names)

    ax.set_xlabel("Predicted", color="#F0EDE6", fontsize=11)
    ax.set_ylabel("True", color="#F0EDE6", fontsize=11)
    ax.set_title(title, color="#F0EDE6", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#888580")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Confusion matrix saved -> {save_path}")


def plot_per_class_accuracy(class_accs: dict, save_path: str,
                             title: str = "Per-Class Accuracy"):
    """Bar chart of per-class accuracy."""
    names = list(class_accs.keys())
    accs = list(class_accs.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#161616")

    colors = ["#C8A96E" if a >= 0.5 else "#C87E7E" for a in accs]
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="#555250")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", color="#F0EDE6", fontsize=11)
    ax.set_title(title, color="#F0EDE6", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#888580")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8, color="#F0EDE6")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Per-class accuracy saved -> {save_path}")
