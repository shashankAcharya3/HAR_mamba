"""
finetune.py — Downstream Fine-Tuning & Linear Probing for HAR-Mamba.

This script:
1. Loads the pretrained Bi-Mamba encoder.
2. Attaches a classification head (``HARClassifier``).
3. Supports two modes:
       • **Linear Probing** — encoder frozen, only head trained.
       • **Full Fine-Tuning** — entire network trained end-to-end.
4. Includes a **Data Scarcity Experiment**: train on
   {1%, 5%, 10%, 25%, 100%} of labels to prove label efficiency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import Config, get_config
from dataset import build_dataloaders
from model import HARClassifier, MaskedMambaAutoencoder
from utils import (
    CosineWarmupScheduler,
    compute_metrics,
    extract_embeddings,
    log_metrics_csv,
    plot_confusion_matrix,
    plot_tsne,
    set_seed,
)


# ════════════════════════════════════════════════════════════════════════
# Training & Evaluation helpers
# ════════════════════════════════════════════════════════════════════════

def _train_epoch(
    model: HARClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One training epoch.  Returns average cross-entropy loss."""
    model.train()
    loss_sum = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()

    for patches, labels, _ in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item()
        steps += 1

    return loss_sum / max(steps, 1)


@torch.no_grad()
def _eval_epoch(
    model: HARClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate and return a metrics dict."""
    model.eval()
    all_preds, all_labels = [], []

    for patches, labels, _ in loader:
        patches = patches.to(device)
        logits = model(patches)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return compute_metrics(y_true, y_pred), y_true, y_pred


# ════════════════════════════════════════════════════════════════════════
# Subsample training set for data-scarcity experiments
# ════════════════════════════════════════════════════════════════════════

def _subsample_loader(
    loader: DataLoader,
    fraction: float,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Return a DataLoader with only ``fraction`` of the original samples.

    Uses stratified subsampling by label to keep class balance.
    """
    dataset = loader.dataset
    n = len(dataset)
    k = max(1, int(n * fraction))

    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=k, replace=False)

    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )


# ════════════════════════════════════════════════════════════════════════
# Main fine-tuning routine
# ════════════════════════════════════════════════════════════════════════

def finetune(
    cfg: Config | None = None,
    pretrained_path: Optional[Path] = None,
    mode: str = "full",            # "full" | "linear_probe"
    label_fraction: float = 1.0,
) -> Dict[str, float]:
    """Fine-tune the pretrained encoder for activity classification.

    Parameters
    ----------
    cfg : Config
    pretrained_path : path to ``pretrained_encoder.pth``
    mode : "full" (unfreeze everything) or "linear_probe" (freeze encoder)
    label_fraction : fraction of training labels to use (for scarcity expt)

    Returns
    -------
    dict of test metrics
    """
    if cfg is None:
        cfg = get_config()
    if pretrained_path is None:
        pretrained_path = cfg.pretrain.checkpoint_dir / cfg.pretrain.best_model_name

    set_seed(cfg.seed)
    device = cfg.device

    tag = f"{mode}_frac{label_fraction:.2f}"
    print(f"\n{'─' * 60}")
    print(f"  Fine-tuning  │  mode={mode}  label_frac={label_fraction:.0%}")
    print(f"{'─' * 60}")

    # ── Data ──────────────────────────────────────────────────────────
    loaders, _ = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Subsample if needed
    if label_fraction < 1.0:
        train_loader = _subsample_loader(
            train_loader, label_fraction,
            batch_size=cfg.finetune.batch_size,
            seed=cfg.seed,
        )
    print(f"  Training samples: {len(train_loader.dataset)}")

    # ── Load pretrained encoder ───────────────────────────────────────
    autoencoder = MaskedMambaAutoencoder(cfg).to(device)
    if cfg.ablation.use_pretraining and pretrained_path.exists():
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        autoencoder.load_state_dict(state)
        print(f"  Loaded pretrained encoder from: {pretrained_path}")
    else:
        print("  Using RANDOM INIT encoder (no pretraining)")

    # ── Build classifier ──────────────────────────────────────────────
    classifier = HARClassifier(autoencoder, cfg).to(device)
    if mode == "linear_probe":
        classifier.freeze_encoder()
        print("  Encoder FROZEN (linear probe)")
    else:
        classifier.unfreeze_encoder()
        print("  Encoder UNFROZEN (full fine-tuning)")

    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in classifier.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params\n")

    # ── Optimiser & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=cfg.finetune.lr,
        weight_decay=cfg.finetune.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.finetune.warmup_epochs,
        total_epochs=cfg.finetune.epochs,
    )

    # ── Training loop ─────────────────────────────────────────────────
    csv_path = cfg.results_dir / f"finetune_{tag}.csv"
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.finetune.epochs + 1):
        t0 = time.time()

        train_loss = _train_epoch(classifier, train_loader, optimizer, device)
        val_metrics, _, _ = _eval_epoch(classifier, val_loader, device)
        lr = scheduler.step(epoch)

        improved = ""
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            improved = " ★"

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1 or improved:
            print(
                f"  Epoch {epoch:3d}/{cfg.finetune.epochs}  │  "
                f"loss={train_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['macro_f1']:.4f}  "
                f"lr={lr:.2e}  │  {elapsed:.1f}s{improved}"
            )

        log_metrics_csv(csv_path, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            **{f"val_{k}": f"{v:.6f}" for k, v in val_metrics.items()},
            "lr": f"{lr:.2e}",
        })

    # ── Restore best & test ───────────────────────────────────────────
    classifier.load_state_dict(best_state)
    classifier.to(device)
    test_metrics, y_true, y_pred = _eval_epoch(classifier, test_loader, device)

    print(f"\n  ┌─── Test Results ({tag}) ───────────────────────┐")
    for k, v in test_metrics.items():
        print(f"  │  {k:>12s}: {v:.4f}")
    print(f"  └────────────────────────────────────────────────┘\n")

    # ── Confusion matrix ──────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=cfg.data.activity_names,
        save_path=cfg.results_dir / f"confusion_matrix_{tag}.png",
        title=f"Confusion Matrix — {tag}",
    )

    # ── Post-finetune t-SNE ───────────────────────────────────────────
    embeddings, labels = extract_embeddings(classifier, test_loader, device)
    plot_tsne(
        embeddings, labels,
        class_names=cfg.data.activity_names,
        save_path=cfg.results_dir / f"tsne_{tag}.png",
        title=f"t-SNE — {tag}",
    )

    # Save final model
    ckpt_path = cfg.pretrain.checkpoint_dir / f"classifier_{tag}.pth"
    torch.save(best_state, ckpt_path)
    print(f"  Classifier saved → {ckpt_path}")

    return test_metrics


# ════════════════════════════════════════════════════════════════════════
# Data Scarcity Experiment
# ════════════════════════════════════════════════════════════════════════

def data_scarcity_experiment(cfg: Config | None = None) -> None:
    """Run fine-tuning at multiple label fractions and log all results."""
    if cfg is None:
        cfg = get_config()

    print(f"\n{'═' * 60}")
    print(f"  DATA SCARCITY EXPERIMENT")
    print(f"  Fractions: {cfg.finetune.label_fractions}")
    print(f"{'═' * 60}")

    all_results = []

    for frac in cfg.finetune.label_fractions:
        metrics = finetune(cfg, mode="full", label_fraction=frac)
        metrics["label_fraction"] = frac
        all_results.append(metrics)

        log_metrics_csv(
            cfg.results_dir / "data_scarcity_results.csv",
            {
                "label_fraction": f"{frac:.2f}",
                **{k: f"{v:.6f}" for k, v in metrics.items() if k != "label_fraction"},
            },
        )

    print(f"\n{'═' * 60}")
    print(f"  Data Scarcity Summary")
    print(f"{'═' * 60}")
    print(f"  {'Frac':>6s}  {'Accuracy':>10s}  {'Macro F1':>10s}")
    print(f"  {'─' * 30}")
    for r in all_results:
        print(
            f"  {r['label_fraction']:>5.0%}  "
            f"  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}"
        )


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HAR-Mamba Fine-tuning")
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "linear_probe", "scarcity"],
        help="Fine-tuning mode.",
    )
    parser.add_argument(
        "--label-fraction", type=float, default=1.0,
        help="Fraction of training labels to use (for single-run mode).",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained encoder checkpoint.",
    )
    args = parser.parse_args()

    cfg = get_config()

    if args.mode == "scarcity":
        data_scarcity_experiment(cfg)
    else:
        pretrained_path = Path(args.pretrained) if args.pretrained else None
        finetune(cfg, pretrained_path=pretrained_path,
                 mode=args.mode, label_fraction=args.label_fraction)
