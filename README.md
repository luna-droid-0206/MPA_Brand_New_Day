# Self-Supervised Image Colorization

> **Learning Visual Representations via Self-Supervised Colorization**
> 
> A representation learning framework that uses image colorization as a proxy task to learn rich visual features — without any labeled data.

---

## 🎯 Core Insight

> *"Colorization is not the goal — representation learning is the goal."*

We train a U-Net (ResNet-18 encoder + decoder) to colorize grayscale images from the STL-10 unlabeled split (100K images). The encoder is forced to learn meaningful visual representations to predict colors. After training, we **discard the decoder** and evaluate the encoder's 512-dimensional embeddings on downstream tasks.

---

## 🏗️ Architecture

```
L channel (1×96×96)
    │
    ▼
┌──────────────────┐
│  ResNet-18        │  ← Modified for 1-channel input
│  Encoder          │
│  (kept after      │  → Multi-scale features (64, 64, 128, 256, 512 channels)
│   training)       │  → 512-d embedding via Global Average Pooling
└──────────────────┘
    │ skip connections
    ▼
┌──────────────────┐
│  U-Net Decoder    │  ← 4 upsampling blocks with skip connections
│  (discarded after │
│   training)       │  → 2-channel output (a, b) with Tanh
└──────────────────┘
    │
    ▼
ab prediction (2×96×96)
```

---

## 📂 Repository Structure

```
colorization-ssl/
├── configs/
│   └── config.yaml              ← All hyperparameters
├── models/
│   ├── encoder.py               ← ResNet-18 encoder (1-ch input, multi-scale)
│   ├── decoder.py               ← U-Net decoder with skip connections
│   └── unet.py                  ← Full U-Net combining encoder + decoder
├── datasets/
│   └── colorization_dataset.py  ← STL-10 → Lab conversion pipeline
├── train/
│   ├── pretrain.py              ← Self-supervised pretraining loop
│   └── extract_features.py      ← Embedding extraction (pretrained + random)
├── eval/
│   ├── classification.py        ← Linear probe + MLP probe evaluation
│   ├── clustering.py            ← K-Means + t-SNE evaluation
│   └── retrieval.py             ← Cosine similarity retrieval + Precision@K
├── metrics/
│   ├── psnr_ssim.py             ← PSNR and SSIM for colorization quality
│   └── representation_metrics.py← Accuracy, ARI, Silhouette, Precision@K
├── utils/
│   ├── lab_utils.py             ← RGB ↔ Lab conversion and normalization
│   ├── visualization.py         ← All plotting functions (dark-themed)
│   └── checkpoints.py           ← Model save/load utilities
├── app/
│   └── streamlit_app.py         ← Interactive demo application
├── run_pipeline.py              ← Full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python run_pipeline.py
```

This will automatically:
1. Download STL-10 dataset (~2.6 GB)
2. Train the colorization U-Net for 100 epochs
3. Extract embeddings (pretrained + random baseline)
4. Run classification evaluation (linear + MLP probes)
5. Run clustering evaluation (K-Means + t-SNE)
6. Run retrieval evaluation (cosine similarity + Precision@K)

### 3. Run Individual Stages

```bash
# Only pretraining
python run_pipeline.py --stage pretrain

# Only feature extraction (requires pretrained model)
python run_pipeline.py --stage extract

# Only classification (requires extracted embeddings)
python run_pipeline.py --stage classify

# Only clustering + t-SNE (requires extracted embeddings)
python run_pipeline.py --stage cluster

# Only retrieval (requires extracted embeddings)
python run_pipeline.py --stage retrieve
```

### 4. Launch the Demo App

```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | STL-10 Unlabeled (100K images) |
| Image Size | 96×96 |
| Color Space | Lab (L → input, ab → target) |
| Encoder | ResNet-18 (1-channel input) |
| Loss Function | MSE on ab channels |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing |
| Batch Size | 128 |
| Epochs | 100 |
| Gradient Clipping | max_norm=1.0 |

---

## 📊 Evaluation Protocol

### Task 1: Image Classification
- **Freeze** the encoder — no encoder weights updated
- Train **logistic regression** (linear probe) on 512-d embeddings
- Train **2-layer MLP** probe as stronger comparison
- Compare: random encoder (baseline) vs. pretrained (ours)

### Task 2: Image Clustering
- **K-Means** with K=10 (matching STL-10 class count)
- Metrics: **Silhouette Score** and **Adjusted Rand Index** (ARI)
- **t-SNE** visualization on 3,000 sampled embeddings
- Side-by-side: random encoder vs. pretrained encoder

### Task 3: Image Retrieval
- **L2-normalize** all embeddings
- **Cosine similarity** for nearest-neighbor search
- Top-9 retrieval grids with class-correctness highlighting
- **Precision@K** across all classes

---

## 🎯 Target Metrics

| Metric | Target Range |
|--------|-------------|
| PSNR (colorization) | 22–26 dB |
| SSIM (colorization) | 0.75–0.88 |
| Top-1 Accuracy (linear probe) | 55–70% |
| Silhouette Score | 0.15–0.35 |
| Adjusted Rand Index | 0.10–0.25 |

---

## 💻 Tech Stack

| Library | Use |
|---------|-----|
| PyTorch | Model training, encoder/decoder, data loading |
| torchvision | STL-10 dataset, ResNet-18 backbone |
| scikit-learn | K-Means, t-SNE, logistic regression, metrics |
| scikit-image | PSNR and SSIM computation |
| OpenCV | Lab color space conversion |
| Matplotlib | All plots and visualizations |
| Streamlit | Interactive demo application |
| NumPy | Embedding storage, similarity computation |

---

## 📁 Output Structure

After running the full pipeline:

```
checkpoints/
├── best_model.pth           ← Best U-Net checkpoint
├── best_encoder.pth         ← Best encoder weights only
├── final_model.pth          ← Final epoch checkpoint
├── checkpoint_epoch_*.pth   ← Periodic checkpoints
└── embeddings/
    ├── train_embeddings.npy  ← (5000, 512) pretrained embeddings
    ├── test_embeddings.npy   ← (8000, 512) pretrained embeddings
    ├── train_labels.npy      ← (5000,) integer labels
    ├── test_labels.npy       ← (8000,) integer labels
    ├── random_train_embeddings.npy
    └── random_test_embeddings.npy

logs/
├── training_loss.png
├── colorization_epoch_*.png
├── classification/
│   ├── classification_results.json
│   ├── pretrained_linear_confusion.png
│   ├── pretrained_mlp_confusion.png
│   └── pretrained_linear_per_class.png
├── clustering/
│   ├── clustering_results.json
│   ├── tsne_pretrained_true_labels.png
│   ├── tsne_pretrained_kmeans.png
│   └── tsne_comparison.png
└── retrieval/
    ├── retrieval_results.json
    └── retrieval_*.png
```

---

## 📖 Key Design Decisions

- **Lab color space**: Clean separation of luminance (L) from chrominance (a, b) makes colorization a natural self-supervised pretext task
- **U-Net over plain encoder**: Skip connections let the encoder focus on semantic features rather than spatial reconstruction
- **Linear probe evaluation**: Standard SSL protocol — measures what the encoder learned, not what fine-tuning adds
- **STL-10**: 100K unlabeled images for pretraining, 13K labeled images for evaluation — standard benchmark in SSL literature

---

## 📝 License

This project is for academic/educational purposes (Final Year Project).
