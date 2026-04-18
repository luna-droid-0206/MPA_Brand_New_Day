"""
Representation Quality Metrics
==============================
Metrics for evaluating the quality of learned representations
across classification, clustering, and retrieval tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
)


def compute_classification_metrics(y_true, y_pred, class_names=None):
    """
    Compute classification metrics.

    Returns:
        dict with overall accuracy, per-class accuracy, confusion matrix,
        and classification report.
    """
    overall_acc = accuracy_score(y_true, y_pred)

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(per_class_acc))]

    per_class_dict = {name: acc for name, acc in zip(class_names, per_class_acc)}

    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True)

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_dict,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_clustering_metrics(labels_true, labels_pred, embeddings=None):
    """
    Compute clustering metrics.

    Args:
        labels_true: Ground truth labels
        labels_pred: Cluster assignments from K-Means
        embeddings:  Optional embeddings for silhouette score

    Returns:
        dict with Adjusted Rand Index, and optionally Silhouette Score
    """
    ari = adjusted_rand_score(labels_true, labels_pred)

    result = {
        "adjusted_rand_index": ari,
    }

    if embeddings is not None:
        # Silhouette score requires at least 2 distinct clusters
        n_unique_clusters = len(np.unique(labels_pred))
        if n_unique_clusters >= 2:
            sil = silhouette_score(embeddings, labels_pred)
            result["silhouette_score"] = sil
        else:
            result["silhouette_score"] = None
            result["warning"] = f"Only {n_unique_clusters} cluster(s) found; silhouette_score requires at least 2 clusters"

    return result


def compute_retrieval_precision(query_label, retrieved_labels, k=None):
    """
    Compute Precision@K for a single query.

    Args:
        query_label:      Integer label of the query image
        retrieved_labels: List/array of labels for retrieved images
        k:                Number of top results to consider (default: all)

    Returns:
        Precision@K value
    """
    if k is not None:
        retrieved_labels = retrieved_labels[:k]

    correct = sum(1 for lbl in retrieved_labels if lbl == query_label)
    return correct / len(retrieved_labels)


def compute_mean_precision_at_k(query_labels, all_retrieved_labels, k=5):
    """
    Compute mean Precision@K across multiple queries.

    Args:
        query_labels:        (N,) array of query labels
        all_retrieved_labels: List of arrays, each containing retrieved labels for one query
        k:                   K for Precision@K

    Returns:
        Mean Precision@K
    """
    precisions = []
    for q_label, r_labels in zip(query_labels, all_retrieved_labels):
        p = compute_retrieval_precision(q_label, r_labels, k=k)
        precisions.append(p)
    return np.mean(precisions)
