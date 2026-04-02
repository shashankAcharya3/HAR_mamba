"""
eval.py — Standalone Evaluation for HAR-Mamba.

This script evaluates a trained classifier and generates all
paper-ready artifacts:
    1. metrics.csv           — Accuracy, Macro F1, Precision, Recall
    2. confusion_matrix.png  — Class-level performance
    3. tsne_embeddings.pdf   — Encoder embeddings visualisation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import Config, get_config
from dataset import build_dataloaders
from model import HARClassifier, MaskedMambaAutoencoder
from utils import (
    compute_metrics,
    extract_embeddings,
    log_metrics_csv,
    plot_confusion_matrix,
    plot_tsne,
    set_seed,
)


@torch.no_grad()
def evaluate(
    cfg: Config | None = None,
    classifier_path: Optional[Path] = None,
    pretrained_path: Optional[Path] = None,
) -> None:
    """Full evaluation pipeline.

    Parameters
    ----------
    cfg : Config
    classifier_path : path to saved ``HARClassifier`` state dict.
        If None, loads the default ``classifier_full_frac1.00.pth``.
    pretrained_path : path to pretrained encoder (for pre-FT t-SNE comparison).
        If None, uses the default ``pretrained_encoder.pth``.
    """
    if cfg is None:
        cfg = get_config()

    set_seed(cfg.seed)
    device = cfg.device

    if classifier_path is None:
        classifier_path = cfg.pretrain.checkpoint_dir / "classifier_full_frac1.00.pth"
    if pretrained_path is None:
        pretrained_path = cfg.pretrain.checkpoint_dir / cfg.pretrain.best_model_name

    print(f"\n{'═' * 60}")
    print(f"  HAR-Mamba  ·  Evaluation")
    print(f"  Device: {device}")
    print(f"{'═' * 60}\n")

    # ── Data ──────────────────────────────────────────────────────────
    loaders, _ = build_dataloaders(cfg)
    test_loader = loaders["test"]

    # ── Build & load classifier ───────────────────────────────────────
    autoencoder = MaskedMambaAutoencoder(cfg).to(device)
    classifier = HARClassifier(autoencoder, cfg).to(device)

    state = torch.load(classifier_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.eval()
    print(f"  Loaded classifier from: {classifier_path}")

    # ── Predict on test set ───────────────────────────────────────────
    all_preds, all_labels = [], []
    for patches, labels, _ in test_loader:
        patches = patches.to(device)
        logits = classifier(patches)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n  ┌─── Test Metrics ──────────────────────────────┐")
    for k, v in metrics.items():
        print(f"  │  {k:>12s}: {v:.4f}")
    print(f"  └────────────────────────────────────────────────┘\n")

    # Save metrics.csv
    csv_path = cfg.results_dir / "metrics.csv"
    log_metrics_csv(csv_path, metrics, append=False)
    print(f"  Metrics saved → {csv_path}")

    # ── Confusion Matrix ──────────────────────────────────────────────
    cm_path = cfg.results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=cfg.data.activity_names,
        save_path=cm_path,
        title="HAR-Mamba — Test Set Confusion Matrix",
    )

    # ── t-SNE: Post fine-tuning ───────────────────────────────────────
    emb_ft, lab_ft = extract_embeddings(classifier, test_loader, device)
    plot_tsne(
        emb_ft, lab_ft,
        class_names=cfg.data.activity_names,
        save_path=cfg.results_dir / "tsne_embeddings.pdf",
        title="t-SNE — After Fine-tuning",
    )

    # ── t-SNE: Pre fine-tuning (from pretrained encoder) ──────────────
    if pretrained_path.exists():
        autoencoder_pt = MaskedMambaAutoencoder(cfg).to(device)
        autoencoder_pt.load_state_dict(
            torch.load(pretrained_path, map_location=device, weights_only=True)
        )
        autoencoder_pt.eval()

        emb_pt, lab_pt = extract_embeddings(autoencoder_pt, test_loader, device)
        plot_tsne(
            emb_pt, lab_pt,
            class_names=cfg.data.activity_names,
            save_path=cfg.results_dir / "tsne_pretrained.pdf",
            title="t-SNE — Pre-trained Encoder (before fine-tuning)",
        )

    print(f"\n  All artifacts saved to: {cfg.results_dir.resolve()}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HAR-Mamba Evaluation")
    parser.add_argument(
        "--classifier", type=str, default=None,
        help="Path to trained classifier checkpoint.",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained encoder (for pre-FT t-SNE).",
    )
    args = parser.parse_args()

    cfg = get_config()
    evaluate(
        cfg,
        classifier_path=Path(args.classifier) if args.classifier else None,
        pretrained_path=Path(args.pretrained) if args.pretrained else None,
    )
