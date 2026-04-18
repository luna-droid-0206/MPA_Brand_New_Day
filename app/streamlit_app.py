"""
Streamlit Demo App — Self-Supervised Colorization Explorer
==========================================================
Interactive app showcasing:
  1. Image Colorization — upload or select images, see colorization results
  2. Image Retrieval — query an image and see most similar images
  3. t-SNE Embedding Visualization — interactive 2D scatter of learned representations
  4. Training Metrics Dashboard — loss curves, PSNR/SSIM, classification results

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torchvision

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder import ColorizationEncoder
from models.unet import ColorizationUNet
from utils.lab_utils import (
    rgb_to_lab, normalize_l, normalize_ab,
    denormalize_l, denormalize_ab, lab_to_rgb,
    tensor_lab_to_rgb,
)
from utils.checkpoints import load_encoder
from eval.retrieval import l2_normalize, cosine_similarity_matrix, retrieve_top_k

# ── Page Config ──
st.set_page_config(
    page_title="Self-Supervised Colorization Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──
CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

CHECKPOINT_DIR = "./checkpoints"
EMBEDDINGS_DIR = os.path.join(CHECKPOINT_DIR, "embeddings")
LOGS_DIR = "./logs"


# ══════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #0D0D0D;
        color: #F0EDE6;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 42px;
        color: #F0EDE6;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .main-title em {
        color: #C8A96E;
        font-style: italic;
    }

    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 16px;
        color: #888580;
        margin-bottom: 32px;
    }

    .metric-card {
        background: #161616;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #555250;
        margin-bottom: 6px;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 32px;
        color: #C8A96E;
    }

    .metric-value.green { color: #7EC8A4; }

    .section-divider {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 40px 0;
    }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #C8A96E;
        margin-bottom: 8px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #F0EDE6;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Model Loading (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_unet_model():
    """Load the full U-Net model for colorization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationUNet(encoder_pretrained=False).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, device, True
    else:
        return model, device, False


@st.cache_resource
def load_embeddings():
    """Load precomputed embeddings."""
    files = {
        "train_emb": "train_embeddings.npy",
        "test_emb": "test_embeddings.npy",
        "train_labels": "train_labels.npy",
        "test_labels": "test_labels.npy",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(EMBEDDINGS_DIR, fname)
        if os.path.exists(path):
            data[key] = np.load(path)
        else:
            return None
    return data


@st.cache_resource
def load_test_dataset():
    """Load STL-10 test images for retrieval visualization."""
    try:
        dataset = torchvision.datasets.STL10(root="./data", split="test", download=False)
        return dataset
    except Exception:
        return None


@st.cache_data
def load_results(subdir, filename):
    """Load JSON results file."""
    path = os.path.join(LOGS_DIR, subdir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════
#  Page: Colorization
# ══════════════════════════════════════════════════════════════

def page_colorization():
    st.markdown('<p class="eyebrow">Colorization Demo</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Grayscale → <em>Color</em></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a grayscale image or select from STL-10 to see the model\'s colorization.</p>', unsafe_allow_html=True)

    model, device, loaded = load_unet_model()

    if not loaded:
        st.warning("⚠️ No pretrained model found. Please run `python -m train.pretrain` first.")
        st.info("Expected checkpoint at: `checkpoints/best_model.pth`")
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        source = st.radio("Image Source", ["Upload Image", "STL-10 Sample"],
                           horizontal=True)

    input_image = None

    if source == "Upload Image":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded:
            input_image = Image.open(uploaded).convert("RGB")
    else:
        test_ds = load_test_dataset()
        if test_ds:
            idx = st.slider("Select image index", 0, len(test_ds) - 1, 0)
            input_image = test_ds[idx][0].convert("RGB")
            label = test_ds[idx][1]
            st.caption(f"Class: **{CLASS_NAMES[label]}**")
        else:
            st.warning("STL-10 test data not found. Run pretraining first to download.")

    if input_image is not None:
        # Resize
        input_image = input_image.resize((96, 96))
        img_np = np.array(input_image, dtype=np.uint8)

        # Convert to Lab
        lab = rgb_to_lab(img_np)
        L = normalize_l(lab[:, :, 0])
        ab_true = normalize_ab(lab[:, :, 1:])

        L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)

        # Predict
        with torch.no_grad():
            ab_pred = model(L_tensor)

        ab_pred_np = ab_pred[0].cpu().numpy()  # (2, H, W)

        # Reconstruct images
        L_display = denormalize_l(L)
        lab_pred = np.stack([
            L_display,
            denormalize_ab(ab_pred_np[0]),
            denormalize_ab(ab_pred_np[1])
        ], axis=-1).astype(np.float32)
        rgb_pred = lab_to_rgb(lab_pred)

        # Display
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Grayscale Input**")
            st.image(L_display / 100.0, clamp=True, use_container_width=True)
        with col_b:
            st.markdown("**Ground Truth**")
            st.image(img_np, use_container_width=True)
        with col_c:
            st.markdown("**Predicted Colorization**")
            st.image(rgb_pred, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  Page: Image Retrieval
# ══════════════════════════════════════════════════════════════

def page_retrieval():
    st.markdown('<p class="eyebrow">Retrieval Demo</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Find <em>Similar</em> Images</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select a query image and find the most visually similar images using learned embeddings.</p>', unsafe_allow_html=True)

    emb_data = load_embeddings()
    test_ds = load_test_dataset()

    if emb_data is None or test_ds is None:
        st.warning("⚠️ Embeddings or test data not found. Run the full pipeline first.")
        return

    test_emb = emb_data["test_emb"]
    test_labels = emb_data["test_labels"]

    # Normalize
    test_emb_norm = l2_normalize(test_emb)
    sim_matrix = cosine_similarity_matrix(test_emb_norm)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Class filter
    class_filter = st.selectbox("Filter by class", ["All"] + CLASS_NAMES)

    if class_filter == "All":
        valid_indices = np.arange(len(test_labels))
    else:
        class_idx = CLASS_NAMES.index(class_filter)
        valid_indices = np.where(test_labels == class_idx)[0]

    query_pos = st.slider("Select query image", 0, len(valid_indices) - 1, 0)
    query_idx = valid_indices[query_pos]

    top_k = st.slider("Number of results", 3, 15, 9)

    # Query image
    query_img = np.array(test_ds[query_idx][0].convert("RGB").resize((96, 96)))
    query_label = test_labels[query_idx]

    st.markdown(f"**Query:** {CLASS_NAMES[query_label]}")
    st.image(query_img, width=150)

    # Retrieve
    top_k_idx = retrieve_top_k(query_idx, sim_matrix, k=top_k)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Retrieved Images:**")

    cols = st.columns(min(top_k, 5))
    for i, idx in enumerate(top_k_idx):
        col = cols[i % len(cols)]
        img = np.array(test_ds[idx][0].convert("RGB").resize((96, 96)))
        lbl = test_labels[idx]
        is_correct = lbl == query_label
        border = "🟢" if is_correct else "🔴"
        with col:
            st.image(img, use_container_width=True)
            st.caption(f"{border} {CLASS_NAMES[lbl]}")

    # Precision
    retrieved_labels = test_labels[top_k_idx]
    precision = sum(1 for l in retrieved_labels if l == query_label) / len(retrieved_labels)
    st.metric(f"Precision@{top_k}", f"{precision:.1%}")


# ══════════════════════════════════════════════════════════════
#  Page: t-SNE Visualization
# ══════════════════════════════════════════════════════════════

def page_tsne():
    st.markdown('<p class="eyebrow">Embedding Space</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title"><em>t-SNE</em> Visualization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">2D projection of learned 512-dimensional embeddings.</p>', unsafe_allow_html=True)

    # Check for precomputed t-SNE plots
    tsne_plots = {
        "Pretrained — True Labels": os.path.join(LOGS_DIR, "clustering", "tsne_pretrained_true_labels.png"),
        "Pretrained — K-Means": os.path.join(LOGS_DIR, "clustering", "tsne_pretrained_kmeans.png"),
        "Random vs. Pretrained": os.path.join(LOGS_DIR, "clustering", "tsne_comparison.png"),
    }

    available = {k: v for k, v in tsne_plots.items() if os.path.exists(v)}

    if available:
        selected = st.selectbox("Select visualization", list(available.keys()))
        st.image(available[selected], use_container_width=True)
    else:
        st.info("No precomputed t-SNE plots found. Run `python -m eval.clustering` to generate them.")

        # Offer live computation
        emb_data = load_embeddings()
        if emb_data is not None:
            if st.button("Compute t-SNE Now (may take a minute)"):
                from sklearn.manifold import TSNE
                from sklearn.preprocessing import StandardScaler

                test_emb = emb_data["test_emb"]
                test_labels = emb_data["test_labels"]

                n_samples = min(3000, len(test_emb))
                rng = np.random.RandomState(42)
                indices = rng.choice(len(test_emb), n_samples, replace=False)

                scaler = StandardScaler()
                emb_scaled = scaler.fit_transform(test_emb[indices])

                with st.spinner("Running t-SNE..."):
                    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                                n_iter=1000, learning_rate="auto", init="pca")
                    emb_2d = tsne.fit_transform(emb_scaled)

                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor("#0D0D0D")
                ax.set_facecolor("#161616")

                for ci in range(10):
                    mask = test_labels[indices] == ci
                    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                               s=8, alpha=0.7, label=CLASS_NAMES[ci])

                ax.legend(fontsize=8, framealpha=0.5,
                          facecolor="#1E1E1E", edgecolor="#555250", labelcolor="#F0EDE6")
                ax.set_title("t-SNE — Pretrained Encoder", color="#F0EDE6", fontsize=14)
                ax.tick_params(colors="#888580")
                st.pyplot(fig)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  Page: Results Dashboard
# ══════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown('<p class="eyebrow">Evaluation Results</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Results <em>Dashboard</em></h1>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Classification Results ──
    st.markdown("### 📊 Classification")
    cls_results = load_results("classification", "classification_results.json")

    if cls_results:
        cols = st.columns(3)
        methods = [
            ("pretrained_linear", "Pretrained (Linear)", "metric-value"),
            ("pretrained_mlp", "Pretrained (MLP)", "metric-value green"),
            ("random_linear", "Random (Baseline)", "metric-value"),
        ]
        for i, (key, label, css_class) in enumerate(methods):
            if key in cls_results:
                acc = cls_results[key]["accuracy"]
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="{css_class}">{acc:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Per-class accuracy chart
        if "pretrained_linear" in cls_results:
            per_class = cls_results["pretrained_linear"]["per_class"]
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0D0D0D")
            ax.set_facecolor("#161616")
            colors = ["#C8A96E" if v >= 0.5 else "#C87E7E" for v in per_class.values()]
            bars = ax.bar(per_class.keys(), per_class.values(), color=colors, alpha=0.85)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy", color="#F0EDE6")
            ax.tick_params(colors="#888580")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        # Confusion matrices
        cm_plots = {
            "Linear Probe": os.path.join(LOGS_DIR, "classification", "pretrained_linear_confusion.png"),
            "MLP Probe": os.path.join(LOGS_DIR, "classification", "pretrained_mlp_confusion.png"),
        }
        available_cm = {k: v for k, v in cm_plots.items() if os.path.exists(v)}
        if available_cm:
            selected_cm = st.selectbox("Confusion Matrix", list(available_cm.keys()))
            st.image(available_cm[selected_cm], use_container_width=True)
    else:
        st.info("Classification results not found. Run `python -m eval.classification`.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Clustering Results ──
    st.markdown("### 🔬 Clustering")
    clust_results = load_results("clustering", "clustering_results.json")

    if clust_results:
        cols = st.columns(4)
        metrics_display = [
            ("Pretrained — Silhouette", clust_results.get("pretrained", {}).get("silhouette_score")),
            ("Pretrained — ARI", clust_results.get("pretrained", {}).get("adjusted_rand_index")),
            ("Random — Silhouette", clust_results.get("random", {}).get("silhouette_score")),
            ("Random — ARI", clust_results.get("random", {}).get("adjusted_rand_index")),
        ]
        for i, (label, value) in enumerate(metrics_display):
            if value is not None:
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Clustering results not found. Run `python -m eval.clustering`.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Retrieval Results ──
    st.markdown("### 🔍 Retrieval")
    ret_results = load_results("retrieval", "retrieval_results.json")

    if ret_results:
        cols = st.columns(2)
        with cols[0]:
            p5 = ret_results.get("mean_precision_at_5", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Precision@5</div>
                <div class="metric-value green">{p5:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            p9 = ret_results.get("mean_precision_at_9", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Precision@9</div>
                <div class="metric-value green">{p9:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Retrieval results not found. Run `python -m eval.retrieval`.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Training Loss Curve ──
    st.markdown("### 📈 Training Loss")
    loss_path = os.path.join(LOGS_DIR, "training_loss.png")
    if os.path.exists(loss_path):
        st.image(loss_path, use_container_width=True)
    else:
        st.info("Training loss curve not found. Run `python -m train.pretrain`.")


# ══════════════════════════════════════════════════════════════
#  Main App Router
# ══════════════════════════════════════════════════════════════

def main():
    load_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎨 Navigation")
        st.markdown("---")
        page = st.radio(
            "Select Page",
            ["🖌️ Colorization", "🔍 Retrieval", "🗺️ t-SNE", "📊 Dashboard"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<p style="font-size: 11px; color: #555250; font-family: JetBrains Mono, monospace;">'
            'Self-Supervised Colorization<br>Final Year Project</p>',
            unsafe_allow_html=True,
        )

    # Route to page
    if page == "🖌️ Colorization":
        page_colorization()
    elif page == "🔍 Retrieval":
        page_retrieval()
    elif page == "🗺️ t-SNE":
        page_tsne()
    elif page == "📊 Dashboard":
        page_dashboard()


if __name__ == "__main__":
    main()
