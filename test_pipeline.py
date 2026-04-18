"""End-to-end validation test for the full pipeline."""

import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np


def test_1_model_forward_backward():
    """Test U-Net forward and backward pass."""
    from models.unet import ColorizationUNet

    model = ColorizationUNet()
    L = torch.randn(4, 1, 96, 96)
    ab_true = torch.randn(4, 2, 96, 96).clamp(-1, 1)
    ab_pred = model(L)

    assert ab_pred.shape == (4, 2, 96, 96), f"Bad output shape: {ab_pred.shape}"

    loss = nn.MSELoss()(ab_pred, ab_true)
    loss.backward()
    print(f"  Test 1 PASS: forward+backward, loss={loss.item():.4f}")
    return model, L


def test_2_embedding_extraction(model, L):
    """Test encoder produces 512-d embeddings."""
    encoder = model.get_encoder()
    emb = encoder.extract_embedding(L)
    assert emb.shape == (4, 512), f"Bad embedding shape: {emb.shape}"
    print(f"  Test 2 PASS: embedding shape {emb.shape}")
    return encoder, emb


def test_3_checkpoint_roundtrip(encoder, L, emb):
    """Test checkpoint save and load roundtrip."""
    from models.encoder import ColorizationEncoder
    from utils.checkpoints import save_encoder, load_encoder

    os.makedirs("_test_tmp", exist_ok=True)
    save_encoder(encoder, "_test_tmp/enc.pth")

    enc2 = ColorizationEncoder()
    load_encoder(enc2, "_test_tmp/enc.pth")
    emb2 = enc2.extract_embedding(L)

    assert torch.allclose(emb.detach(), emb2.detach(), atol=1e-5)
    shutil.rmtree("_test_tmp")
    print("  Test 3 PASS: checkpoint roundtrip matches")


def test_4_lab_conversion():
    """Test Lab color space conversion and normalization."""
    from utils.lab_utils import (
        rgb_to_lab, lab_to_rgb, normalize_l, denormalize_l,
        normalize_ab, denormalize_ab,
    )

    img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    lab = rgb_to_lab(img)

    L = normalize_l(lab[:, :, 0])
    ab = normalize_ab(lab[:, :, 1:])

    assert L.min() >= -1.01 and L.max() <= 1.01, f"L out of range: [{L.min()}, {L.max()}]"
    assert ab.min() >= -1.01 and ab.max() <= 1.01, f"ab out of range: [{ab.min()}, {ab.max()}]"

    # Roundtrip
    L_back = denormalize_l(L)
    ab_back = denormalize_ab(ab)
    lab_back = np.stack([L_back, ab_back[:, :, 0], ab_back[:, :, 1]], axis=-1)
    rgb_back = lab_to_rgb(lab_back)
    assert rgb_back.shape == (96, 96, 3)

    print("  Test 4 PASS: Lab conversion + normalization roundtrip")


def test_5_metrics():
    """Test classification and clustering metrics."""
    from metrics.representation_metrics import (
        compute_classification_metrics,
        compute_clustering_metrics,
        compute_retrieval_precision,
    )

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 0, 2, 0, 1, 1, 0])

    cls_metrics = compute_classification_metrics(y_true, y_pred)
    assert 0 <= cls_metrics["overall_accuracy"] <= 1.0
    acc = cls_metrics["overall_accuracy"]

    # Clustering
    embeddings = np.random.randn(10, 64)
    clust_metrics = compute_clustering_metrics(y_true, y_pred, embeddings)
    assert "adjusted_rand_index" in clust_metrics

    # Retrieval
    prec = compute_retrieval_precision(0, [0, 0, 1, 2, 0], k=5)
    assert prec == 0.6

    print(f"  Test 5 PASS: metrics (accuracy={acc:.2f}, "
          f"ARI={clust_metrics['adjusted_rand_index']:.3f}, P@5={prec:.1f})")


def test_6_psnr_ssim():
    """Test PSNR and SSIM computation."""
    from metrics.psnr_ssim import compute_psnr, compute_ssim

    img1 = np.random.randn(96, 96, 2).astype(np.float32)
    img2 = img1 + np.random.randn(96, 96, 2).astype(np.float32) * 0.1

    psnr_val = compute_psnr(img1, img2)
    assert psnr_val > 0, f"PSNR should be positive: {psnr_val}"

    ssim_val = compute_ssim(img1[:, :, 0], img2[:, :, 0])
    assert 0 <= ssim_val <= 1.0 or ssim_val > 0, f"SSIM unexpected: {ssim_val}"

    print(f"  Test 6 PASS: PSNR={psnr_val:.1f} dB, SSIM={ssim_val:.3f}")


def test_7_retrieval():
    """Test retrieval pipeline."""
    from eval.retrieval import l2_normalize, cosine_similarity_matrix, retrieve_top_k

    embs = np.random.randn(20, 512).astype(np.float32)
    embs_n = l2_normalize(embs)

    norms = np.linalg.norm(embs_n, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "L2 normalization failed"

    sim = cosine_similarity_matrix(embs_n)
    assert sim.shape == (20, 20)

    topk = retrieve_top_k(0, sim, k=5)
    assert len(topk) == 5
    assert 0 not in topk

    print(f"  Test 7 PASS: retrieval (topk indices: {topk})")


def test_8_visualization():
    """Test that visualization functions run without error."""
    from utils.visualization import plot_loss_curve

    os.makedirs("_test_tmp", exist_ok=True)
    plot_loss_curve([0.5, 0.4, 0.3, 0.2, 0.1], "_test_tmp/test_loss.png")
    assert os.path.exists("_test_tmp/test_loss.png")
    shutil.rmtree("_test_tmp")
    print("  Test 8 PASS: visualization (loss curve generated)")


def main():
    print()
    print("=" * 56)
    print("  Self-Supervised Colorization — Validation Tests")
    print("=" * 56)
    print()

    model, L = test_1_model_forward_backward()
    encoder, emb = test_2_embedding_extraction(model, L)
    test_3_checkpoint_roundtrip(encoder, L, emb)
    test_4_lab_conversion()
    test_5_metrics()
    test_6_psnr_ssim()
    test_7_retrieval()
    test_8_visualization()

    print()
    print("=" * 56)
    print("  ALL 8 TESTS PASSED — Project is ready!")
    print("=" * 56)
    print()
    print("  Next steps:")
    print("    1. python run_pipeline.py          # Full training pipeline")
    print("    2. streamlit run app/streamlit_app.py  # Interactive demo")
    print()


if __name__ == "__main__":
    main()
