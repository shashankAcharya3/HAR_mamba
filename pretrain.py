"""
pretrain.py — Self-Supervised Pretraining for HAR-Mamba.

Training Loop
═════════════
1. Feed patchified sensor windows + raw windows into the
   ``MaskedMambaAutoencoder``.
2. PDGM selects high-dynamics patches to mask.
3. Bi-Mamba encoder processes visible patches.
4. Lightweight MLP decoder reconstructs masked patches.
5. Loss = MSE computed **only on masked patches**.

The best encoder checkpoint (lowest validation reconstruction loss)
is saved to ``checkpoints/pretrained_encoder.pth``.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from config import Config, get_config
from dataset import build_dataloaders
from model import MaskedMambaAutoencoder
from utils import (
    CosineWarmupScheduler,
    extract_embeddings,
    log_metrics_csv,
    plot_reconstruction,
    plot_tsne,
    set_seed,
)


def pretrain(cfg: Config | None = None) -> Path:
    """Run self-supervised pretraining.

    Returns
    -------
    Path to the best encoder checkpoint.
    """
    if cfg is None:
        cfg = get_config()

    set_seed(cfg.seed)
    device = cfg.device
    print(f"\n{'═' * 60}")
    print(f"  HAR-Mamba  ·  Self-Supervised Pretraining")
    print(f"  Device: {device}  ·  Seed: {cfg.seed}")
    print(f"{'═' * 60}\n")

    # ── Data ──────────────────────────────────────────────────────────
    loaders, stats = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ── Model ─────────────────────────────────────────────────────────
    model = MaskedMambaAutoencoder(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] MaskedMambaAutoencoder — {total_params:,} params\n")

    # ── Optimiser & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.pretrain.warmup_epochs,
        total_epochs=cfg.pretrain.epochs,
    )

    # ── Checkpoint path ───────────────────────────────────────────────
    ckpt_dir = cfg.pretrain.checkpoint_dir
    best_path = ckpt_dir / cfg.pretrain.best_model_name
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    csv_path = cfg.results_dir / "pretrain_log.csv"

    for epoch in range(1, cfg.pretrain.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for patches, _labels, raw_window in train_loader:
            patches = patches.to(device)
            raw_window = raw_window.to(device)

            _, _, loss = model(patches, raw_window)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        train_loss = train_loss_sum / max(train_steps, 1)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for patches, _labels, raw_window in val_loader:
                patches = patches.to(device)
                raw_window = raw_window.to(device)

                _, _, loss = model(patches, raw_window)
                val_loss_sum += loss.item()
                val_steps += 1

        val_loss = val_loss_sum / max(val_steps, 1)

        # ── LR schedule ──────────────────────────────────────────────
        lr = scheduler.step(epoch)

        elapsed = time.time() - t0
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            improved = " ★ saved"

        print(
            f"  Epoch {epoch:3d}/{cfg.pretrain.epochs}  │  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"lr={lr:.2e}  │  {elapsed:.1f}s{improved}"
        )

        # Log to CSV
        log_metrics_csv(csv_path, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "lr": f"{lr:.2e}",
        })

    print(f"\n[PRETRAIN] Best val loss: {best_val_loss:.5f}")
    print(f"[PRETRAIN] Checkpoint saved → {best_path}\n")

    # ── Generate reconstruction plot ──────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        sample_patches, sample_labels, sample_raw = next(iter(val_loader))
        sample_patches = sample_patches.to(device)
        sample_raw = sample_raw.to(device)

        reconstruction, mask, _ = model(sample_patches, sample_raw)

        # Plot first sample
        idx = 0
        plot_reconstruction(
            raw_window=sample_raw[idx].cpu().numpy(),
            reconstruction=reconstruction[idx].cpu().numpy(),
            mask=mask[idx].cpu().numpy(),
            patch_size=cfg.patch.patch_size,
            save_path=cfg.results_dir / "reconstruction_plot.png",
            channel_idx=0,
            title="PDGM Reconstruction (body_acc_x)",
        )

    # ── Generate pre-finetune t-SNE ───────────────────────────────────
    embeddings, labels = extract_embeddings(model, val_loader, device)
    plot_tsne(
        embeddings, labels,
        class_names=cfg.data.activity_names,
        save_path=cfg.results_dir / "tsne_pretrained.png",
        title="t-SNE — Pre-trained Encoder (before fine-tuning)",
    )

    return best_path


if __name__ == "__main__":
    pretrain()
