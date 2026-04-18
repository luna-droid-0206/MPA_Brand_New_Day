"""
Main Runner — Self-Supervised Image Colorization
=================================================
Orchestrates the full pipeline:
  1. Pretraining (colorization on unlabeled STL-10)
  2. Feature extraction (pretrained + random baseline)
  3. Classification evaluation (linear & MLP probes)
  4. Clustering evaluation (K-Means + t-SNE)
  5. Retrieval evaluation (cosine similarity + Precision@K)

Usage:
    python run_pipeline.py [--config configs/config.yaml] [--stage STAGE]

    Stages: all | pretrain | extract | classify | cluster | retrieve

Examples:
    python run_pipeline.py                         # Run full pipeline
    python run_pipeline.py --stage pretrain         # Only pretraining
    python run_pipeline.py --stage extract          # Only feature extraction
    python run_pipeline.py --stage classify         # Only classification eval
"""

import os
import sys
import argparse
import time
import yaml

sys.path.insert(0, os.path.dirname(__file__))


def banner(text):
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Self-Supervised Colorization — Full Pipeline Runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "pretrain", "extract", "classify", "cluster", "retrieve"],
        help="Which stage to run (default: all)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    stages = (
        ["pretrain", "extract", "classify", "cluster", "retrieve"]
        if args.stage == "all"
        else [args.stage]
    )

    total_start = time.time()

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Self-Supervised Image Colorization — Pipeline Runner   ║")
    print("  ║  Learning Visual Representations via Colorization       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Config:  {args.config}")
    print(f"  Stages:  {', '.join(stages)}")
    print(f"  Device:  {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    print()

    embeddings_dir = os.path.join(config["pretraining"]["checkpoint_dir"], "embeddings")
    log_dir = config["pretraining"]["log_dir"]

    # ── Stage 1: Pretraining ──
    if "pretrain" in stages:
        banner("STAGE 1 — Self-Supervised Pretraining")
        t0 = time.time()
        from train.pretrain import main as pretrain_main
        sys.argv = ["pretrain", "--config", args.config]
        pretrain_main()
        print(f"\n  ⏱  Pretraining completed in {time.time() - t0:.1f}s")

    # ── Stage 2: Feature Extraction ──
    if "extract" in stages:
        banner("STAGE 2 — Feature Extraction")
        t0 = time.time()
        from train.extract_features import main as extract_main
        sys.argv = ["extract_features", "--config", args.config]
        extract_main()
        print(f"\n  ⏱  Feature extraction completed in {time.time() - t0:.1f}s")

    # ── Stage 3: Classification Evaluation ──
    if "classify" in stages:
        banner("STAGE 3 — Classification Evaluation")
        t0 = time.time()
        from eval.classification import run_classification_eval
        results_dir = os.path.join(log_dir, "classification")
        run_classification_eval(embeddings_dir, results_dir, config)
        print(f"\n  ⏱  Classification eval completed in {time.time() - t0:.1f}s")

    # ── Stage 4: Clustering Evaluation ──
    if "cluster" in stages:
        banner("STAGE 4 — Clustering & t-SNE Evaluation")
        t0 = time.time()
        from eval.clustering import run_clustering_eval
        results_dir = os.path.join(log_dir, "clustering")
        run_clustering_eval(embeddings_dir, results_dir, config)
        print(f"\n  ⏱  Clustering eval completed in {time.time() - t0:.1f}s")

    # ── Stage 5: Retrieval Evaluation ──
    if "retrieve" in stages:
        banner("STAGE 5 — Image Retrieval Evaluation")
        t0 = time.time()
        from eval.retrieval import run_retrieval_eval
        results_dir = os.path.join(log_dir, "retrieval")
        run_retrieval_eval(
            embeddings_dir, results_dir, config,
            data_dir=config["dataset"]["data_dir"]
        )
        print(f"\n  ⏱  Retrieval eval completed in {time.time() - t0:.1f}s")

    # ── Done ──
    total_time = time.time() - total_start
    mins, secs = divmod(int(total_time), 60)

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                   PIPELINE COMPLETE                     ║")
    print(f"  ║  Total time: {mins:02d}m {secs:02d}s{' ' * 38}║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Next steps:                                            ║")
    print("  ║    • streamlit run app/streamlit_app.py                 ║")
    print("  ║    • Check logs/ for all visualizations                 ║")
    print("  ║    • Check checkpoints/ for saved models                ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
