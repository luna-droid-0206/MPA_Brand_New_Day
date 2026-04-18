"""
Image Classification Evaluation
================================
Evaluate learned representations using linear and MLP probes.

Protocol:
  1. Load frozen embeddings (pretrained or random encoder)
  2. Standardize features
  3. Train logistic regression (linear probe) on frozen embeddings
  4. Train 2-layer MLP probe for stronger comparison
  5. Report overall accuracy + per-class accuracy
  6. Compare: random encoder vs. pretrained vs. fully supervised baseline

Usage:
    python -m eval.classification [--config configs/config.yaml]
"""

import os
import sys
import argparse
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.representation_metrics import compute_classification_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
)

# STL-10 class names
CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


class MLPProbe(nn.Module):
    """2-layer MLP probe for classification on frozen embeddings."""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_linear_probe(X_train, y_train, X_test, y_test):
    """
    Train a logistic regression (linear probe) on frozen embeddings.

    Returns:
        y_pred_train, y_pred_test, model
    """
    print("\n  [LINEAR] Training logistic regression probe...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        verbose=0,
    )
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    print(f"  [LINEAR] Train Accuracy: {train_acc:.4f}")
    print(f"  [LINEAR] Test Accuracy:  {test_acc:.4f}")

    return y_pred_train, y_pred_test, model


def train_mlp_probe(X_train, y_train, X_test, y_test,
                     hidden_dim=256, epochs=50, lr=0.001, batch_size=256):
    """
    Train a 2-layer MLP probe on frozen embeddings.

    Returns:
        y_pred_train, y_pred_test
    """
    print("\n  [MLP] Training 2-layer MLP probe...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled).float(),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_scaled).float(),
        torch.from_numpy(y_test).long(),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = X_train_scaled.shape[1]
    model = MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim,
                      num_classes=len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            # Quick eval
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    correct += (pred.argmax(1) == yb).sum().item()
                    total += yb.size(0)
            print(f"    Epoch {epoch:3d}/{epochs}  Loss: {total_loss/len(train_dl):.4f}  "
                  f"Test Acc: {correct/total:.4f}")

    # Final predictions
    model.eval()
    y_pred_train_list = []
    y_pred_test_list = []

    with torch.no_grad():
        for xb, _ in train_dl:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            y_pred_train_list.append(pred)
        for xb, _ in test_dl:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            y_pred_test_list.append(pred)

    y_pred_train = np.concatenate(y_pred_train_list)
    y_pred_test = np.concatenate(y_pred_test_list)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    print(f"  [MLP] Final Train Accuracy: {train_acc:.4f}")
    print(f"  [MLP] Final Test Accuracy:  {test_acc:.4f}")

    return y_pred_train, y_pred_test


def run_classification_eval(embeddings_dir, results_dir, config):
    """Run the full classification evaluation pipeline."""

    os.makedirs(results_dir, exist_ok=True)

    # Load embeddings
    train_emb = np.load(os.path.join(embeddings_dir, "train_embeddings.npy"))
    test_emb = np.load(os.path.join(embeddings_dir, "test_embeddings.npy"))
    train_labels = np.load(os.path.join(embeddings_dir, "train_labels.npy"))
    test_labels = np.load(os.path.join(embeddings_dir, "test_labels.npy"))

    random_train_emb = np.load(os.path.join(embeddings_dir, "random_train_embeddings.npy"))
    random_test_emb = np.load(os.path.join(embeddings_dir, "random_test_embeddings.npy"))

    eval_cfg = config["evaluation"]["classification"]
    results = {}

    # ── Pretrained: Linear Probe ──
    print("\n" + "="*50)
    print("  PRETRAINED ENCODER — Linear Probe")
    print("="*50)
    _, y_pred_test_linear, _ = train_linear_probe(
        train_emb, train_labels, test_emb, test_labels
    )
    metrics_linear = compute_classification_metrics(
        test_labels, y_pred_test_linear, CLASS_NAMES
    )
    results["pretrained_linear"] = {
        "accuracy": metrics_linear["overall_accuracy"],
        "per_class": metrics_linear["per_class_accuracy"],
    }

    plot_confusion_matrix(
        metrics_linear["confusion_matrix"], CLASS_NAMES,
        os.path.join(results_dir, "pretrained_linear_confusion.png"),
        title="Pretrained Encoder — Linear Probe"
    )
    plot_per_class_accuracy(
        metrics_linear["per_class_accuracy"],
        os.path.join(results_dir, "pretrained_linear_per_class.png"),
        title="Pretrained Encoder — Per-Class Accuracy (Linear)"
    )

    # ── Pretrained: MLP Probe ──
    if eval_cfg.get("mlp_probe", True):
        print("\n" + "="*50)
        print("  PRETRAINED ENCODER — MLP Probe")
        print("="*50)
        _, y_pred_test_mlp = train_mlp_probe(
            train_emb, train_labels, test_emb, test_labels,
            hidden_dim=eval_cfg.get("mlp_hidden_dim", 256),
            epochs=eval_cfg.get("mlp_epochs", 50),
            lr=eval_cfg.get("mlp_lr", 0.001),
        )
        metrics_mlp = compute_classification_metrics(
            test_labels, y_pred_test_mlp, CLASS_NAMES
        )
        results["pretrained_mlp"] = {
            "accuracy": metrics_mlp["overall_accuracy"],
            "per_class": metrics_mlp["per_class_accuracy"],
        }

        plot_confusion_matrix(
            metrics_mlp["confusion_matrix"], CLASS_NAMES,
            os.path.join(results_dir, "pretrained_mlp_confusion.png"),
            title="Pretrained Encoder — MLP Probe"
        )

    # ── Random: Linear Probe (Baseline) ──
    print("\n" + "="*50)
    print("  RANDOM ENCODER — Linear Probe (Baseline)")
    print("="*50)
    _, y_pred_test_random, _ = train_linear_probe(
        random_train_emb, train_labels, random_test_emb, test_labels
    )
    metrics_random = compute_classification_metrics(
        test_labels, y_pred_test_random, CLASS_NAMES
    )
    results["random_linear"] = {
        "accuracy": metrics_random["overall_accuracy"],
        "per_class": metrics_random["per_class_accuracy"],
    }

    # ── Summary ──
    print("\n" + "="*60)
    print("  CLASSIFICATION RESULTS SUMMARY")
    print("="*60)
    for method, data in results.items():
        print(f"  {method:25s}  Accuracy: {data['accuracy']:.4f}")
    print("="*60)

    # Save results
    results_serializable = {}
    for k, v in results.items():
        results_serializable[k] = {
            "accuracy": float(v["accuracy"]),
            "per_class": {cn: float(acc) for cn, acc in v["per_class"].items()},
        }

    with open(os.path.join(results_dir, "classification_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Classification Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    embeddings_dir = os.path.join(config["pretraining"]["checkpoint_dir"], "embeddings")
    results_dir = os.path.join(config["pretraining"]["log_dir"], "classification")

    run_classification_eval(embeddings_dir, results_dir, config)


if __name__ == "__main__":
    main()
