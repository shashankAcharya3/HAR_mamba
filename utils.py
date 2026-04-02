"""
utils.py — Shared utilities for HAR-Mamba.

Contains:
    • Reproducibility seeding
    • Learning rate schedulers (cosine w/ warmup)
    • Metric computation
    • Visualisation helpers (confusion matrix, t-SNE, reconstruction plots)
    • CSV logging
"""

from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config


# ════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════
# Learning Rate Scheduler
# ════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup.

    LR schedule:
        epoch < warmup_epochs → linear ramp from 0 to base_lr
        epoch ≥ warmup_epochs → cosine decay from base_lr to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        """Update learning rate and return the new value."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Returns
    -------
    dict with keys: accuracy, macro_f1, precision, recall
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ════════════════════════════════════════════════════════════════════════
# CSV Logging
# ════════════════════════════════════════════════════════════════════════

def log_metrics_csv(
    filepath: Path,
    row: Dict[str, object],
    append: bool = True,
) -> None:
    """Append a row of metrics to a CSV file (create header if new)."""
    file_exists = filepath.exists()
    mode = "a" if append else "w"

    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists or not append:
            writer.writeheader()
        writer.writerow(row)


# ════════════════════════════════════════════════════════════════════════
# Visualisation: Confusion Matrix
# ════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a publication-quality confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] Confusion matrix saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# Visualisation: t-SNE Embeddings
# ════════════════════════════════════════════════════════════════════════

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "t-SNE Embeddings",
    perplexity: int = 30,
) -> None:
    """Generate a 2-D t-SNE scatter plot of encoder embeddings.

    Parameters
    ----------
    embeddings : (N, H) — encoder output after mean pooling
    labels     : (N,)   — ground-truth class indices
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10", len(class_names))

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=[cmap(cls_idx)],
            label=cls_name,
            s=12, alpha=0.7, edgecolors="none",
        )

    ax.legend(
        markerscale=3, fontsize=9,
        bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] t-SNE plot saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# Visualisation: Reconstruction Plot
# ════════════════════════════════════════════════════════════════════════

def plot_reconstruction(
    raw_window: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    save_path: Path,
    channel_idx: int = 0,
    title: str = "PDGM Reconstruction",
) -> None:
    """Visualise the masking and reconstruction of a single sample.

    Draws:
        • The original sensor signal (one channel).
        • Shaded bands showing which patches PDGM selected to mask.
        • The model's reconstruction overlaid.

    Parameters
    ----------
    raw_window     : (T, C)       — original sensor window
    reconstruction : (P, D)       — reconstructed patches
    mask           : (P,)         — True = masked
    patch_size     : int
    save_path      : Path
    channel_idx    : which channel to plot
    """
    T, C = raw_window.shape
    P = len(mask)

    # Un-patchify reconstruction for the chosen channel
    recon_full = reconstruction.reshape(P, patch_size, C)  # (P, ps, C)
    recon_signal = recon_full[:, :, channel_idx].flatten()  # (T,)
    original_signal = raw_window[:, channel_idx]            # (T,)

    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(T)

    # Original
    ax.plot(t, original_signal, color="#2563eb", linewidth=1.2, label="Original", zorder=3)
    # Reconstruction
    ax.plot(t, recon_signal, color="#f97316", linewidth=1.0, linestyle="--",
            label="Reconstruction", alpha=0.9, zorder=4)

    # Shade masked patches
    for p_idx in range(P):
        if mask[p_idx]:
            start = p_idx * patch_size
            end = start + patch_size
            ax.axvspan(start, end, alpha=0.15, color="#ef4444", zorder=1)

    # Legend entry for masked regions
    ax.fill_between([], [], [], color="#ef4444", alpha=0.15, label="Masked by PDGM")

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] Reconstruction plot saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# Extract encoder embeddings (for t-SNE / eval)
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the encoder over a dataset and collect mean-pooled embeddings.

    Works with both ``MaskedMambaAutoencoder`` (via ``.encode``) and
    ``HARClassifier`` (uses patch_embed → blocks → norm).

    Returns
    -------
    embeddings : (N, H)
    labels     : (N,)
    """
    model.eval()
    all_embs, all_labels = [], []

    for patches, labels, _ in dataloader:
        patches = patches.to(device)

        # Use .encode() if available (autoencoder), else forward up to head
        if hasattr(model, "encode"):
            emb = model.encode(patches)           # (B, P, H)
        else:
            # HARClassifier — replicate forward minus the head
            x = model.patch_embed(patches)
            x = x + model.pos_embed[:, :x.size(1)]
            x = model.encoder_dropout(x)
            for block in model.encoder_blocks:
                x = block(x)
            emb = model.encoder_norm(x)            # (B, P, H)

        emb = emb.mean(dim=1)                      # (B, H) global pool
        all_embs.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)
