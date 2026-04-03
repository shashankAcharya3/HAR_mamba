"""
dataset.py — Data Pipeline for HAR-Mamba.

Responsibilities
================
1.  Load the **raw inertial signals** (body_acc, body_gyro, total_acc)
    from the UCI HAR Dataset.
2.  Apply optional low-pass filtering (gravity removal already done by
    UCI's preprocessing, but we add a configurable Butterworth pass for
    extra noise rejection).
3.  Compute the **Kinematic Interaction Tensor (KIT)**:
        F_{int,t} = a_t × ω_t   (cross-product per timestep)
    and concatenate it to produce a 12-channel input.
4.  Perform **subject-independent splitting** (train / val / test).
5.  **Patchify** the 128-step windows into non-overlapping patches.

The module exposes:
    • ``build_dataloaders(cfg)`` → dict of ``DataLoader``s keyed by split.
    • ``HARDataset``             → a ``torch.utils.data.Dataset``.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, Tuple

from config import Config, get_config


# ════════════════════════════════════════════════════════════════════════
# 1.  Raw I/O helpers
# ════════════════════════════════════════════════════════════════════════

def _load_signal(filepath: Path) -> np.ndarray:
    """Load a single UCI HAR inertial signal file.

    Each row is a fixed-width 128-column vector of floats.

    Returns
    -------
    np.ndarray of shape ``(N, 128)``
    """
    return np.loadtxt(filepath)


def _load_inertial_signals(root: Path, split: str, cfg: Config) -> np.ndarray:
    """Stack all 9 inertial channels into a single tensor.

    Returns
    -------
    np.ndarray of shape ``(N, 128, 9)``  — (samples, timesteps, channels)
    """
    suffix = "train" if split == "train" else "test"
    sig_dir = root / split / "Inertial Signals"

    channels = []
    for sig_name in cfg.data.inertial_signals:
        fpath = sig_dir / f"{sig_name}_{suffix}.txt"
        channels.append(_load_signal(fpath))              # (N, 128)

    # channels: list of 9 × (N, 128) → stack to (N, 128, 9)
    return np.stack(channels, axis=-1)


def _load_labels(root: Path, split: str) -> np.ndarray:
    """Load activity labels (1-indexed) → 0-indexed int array ``(N,)``."""
    suffix = "train" if split == "train" else "test"
    y = np.loadtxt(root / split / f"y_{suffix}.txt", dtype=int)
    return y - 1  # shift to 0-based


def _load_subjects(root: Path, split: str) -> np.ndarray:
    """Load subject IDs ``(N,)``."""
    suffix = "train" if split == "train" else "test"
    return np.loadtxt(root / split / f"subject_{suffix}.txt", dtype=int)


# ════════════════════════════════════════════════════════════════════════
# 2.  Preprocessing: Low-pass Filter
# ════════════════════════════════════════════════════════════════════════

def _butterworth_lowpass(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter channel-wise.

    Parameters
    ----------
    data : np.ndarray of shape ``(N, T, C)``
    cutoff : Hz
    fs : sampling rate Hz
    order : filter order

    Returns
    -------
    Filtered array of same shape.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    sos = butter(order, norm_cutoff, btype="low", output="sos")
    # sosfiltfilt along axis=1 (time)
    return sosfiltfilt(sos, data, axis=1).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 3.  Kinematic Interaction Tensor (KIT)
# ════════════════════════════════════════════════════════════════════════

def compute_kit(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Compute the cross-product F_{int,t} = a_t × ω_t.

    The Kinematic Interaction Tensor captures rotational jerk and
    Coriolis-like effects that encode richer motion semantics than
    a naive concatenation of accelerometer and gyroscope readings.

    Parameters
    ----------
    raw : np.ndarray, shape ``(N, T, C)``  where C ≥ 6
        Channels ordered as [body_acc_x, body_acc_y, body_acc_z,
                              body_gyro_x, body_gyro_y, body_gyro_z, ...]
    cfg : Config

    Returns
    -------
    np.ndarray, shape ``(N, T, 3)``  — the cross-product per timestep.
    """
    acc_idx = list(cfg.data.acc_channel_indices)   # [0,1,2]
    gyro_idx = list(cfg.data.gyro_channel_indices) # [3,4,5]

    a = raw[:, :, acc_idx]   # (N, T, 3)
    omega = raw[:, :, gyro_idx]   # (N, T, 3)

    # Vectorised cross product along last axis
    F_int = np.cross(a, omega)  # (N, T, 3)
    return F_int.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 4.  Patchification
# ════════════════════════════════════════════════════════════════════════

def patchify(x: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape ``(N, T, C)`` → ``(N, num_patches, patch_size * C)``.

    Each patch is a flattened slice of ``patch_size`` consecutive timesteps
    across *all* channels.

    Parameters
    ----------
    x : np.ndarray, shape ``(N, T, C)``
    patch_size : int

    Returns
    -------
    np.ndarray, shape ``(N, T // patch_size, patch_size * C)``
    """
    N, T, C = x.shape
    assert T % patch_size == 0, (
        f"Window length {T} not divisible by patch_size {patch_size}"
    )
    num_patches = T // patch_size
    # (N, num_patches, patch_size, C) → (N, num_patches, patch_size * C)
    x = x.reshape(N, num_patches, patch_size, C)
    x = x.reshape(N, num_patches, patch_size * C)
    return x


# ════════════════════════════════════════════════════════════════════════
# 5.  PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════

class HARDataset(Dataset):
    """UCI HAR inertial-signal dataset with KIT augmentation and patchification.

    Each sample is a tuple ``(patches, label, raw_window)`` where:
        • ``patches``    : FloatTensor ``(num_patches, patch_dim)``
        • ``label``      : LongTensor  scalar
        • ``raw_window`` : FloatTensor ``(T, C_total)`` — kept for
                           reconstruction visualisation.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.raw = data.astype(np.float32)          # (N, T, C)
        self.patches = patchify(data, patch_size).astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.patches[idx]),    # (P, D)
            torch.tensor(self.labels[idx]),          # scalar
            torch.from_numpy(self.raw[idx]),         # (T, C)
        )


# ════════════════════════════════════════════════════════════════════════
# 6.  Build everything: load → preprocess → split → dataloaders
# ════════════════════════════════════════════════════════════════════════

def _preprocess(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Full preprocessing pipeline: low-pass → KIT → concatenate.

    Parameters
    ----------
    raw : ``(N, 128, 9)``

    Returns
    -------
    ``(N, 128, 12)`` — original 9 channels + 3 KIT channels.
    """
    if cfg.data.apply_lowpass:
        raw = _butterworth_lowpass(
            raw,
            cutoff=cfg.data.lowpass_cutoff_hz,
            fs=cfg.data.sampling_rate_hz,
            order=cfg.data.lowpass_order,
        )

    if cfg.ablation.use_kit:
        kit = compute_kit(raw, cfg)  # (N, 128, 3)
        data = np.concatenate([raw, kit], axis=-1)  # (N, 128, 12)
    else:
        data = raw.copy()  # (N, 128, 9) — no physics prior

    # Per-channel z-score normalisation (computed on *this* split for now)
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True) + 1e-8
    data = (data - mean) / std

    return data.astype(np.float32)


def build_dataloaders(
    cfg: Config | None = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
    """Construct train / val / test ``DataLoader``s.

    The UCI HAR Dataset already provides a subject-independent
    train/test split (21 train subjects, 9 test subjects).
    We further carve out ``cfg.data.val_subjects`` from the train
    fold to form a validation set.

    Returns
    -------
    loaders : dict  {"train": …, "val": …, "test": …}
    stats   : dict  with normalisation statistics and metadata
    """
    if cfg is None:
        cfg = get_config()

    root = cfg.data.dataset_root
    print(f"[DATA] Loading UCI HAR from: {root.resolve()}")

    # ── Load raw signals ──────────────────────────────────────────────
    train_raw = _load_inertial_signals(root, "train", cfg)   # (7352, 128, 9)
    test_raw = _load_inertial_signals(root, "test", cfg)     # (2947, 128, 9)

    train_y = _load_labels(root, "train")                    # (7352,)
    test_y = _load_labels(root, "test")                      # (2947,)

    train_subj = _load_subjects(root, "train")               # (7352,)

    # ── Preprocess ────────────────────────────────────────────────────
    train_data = _preprocess(train_raw, cfg)   # (7352, 128, 12)
    test_data = _preprocess(test_raw, cfg)     # (2947, 128, 12)

    # ── Subject-independent val split ─────────────────────────────────
    val_mask = np.isin(train_subj, cfg.data.val_subjects)
    train_mask = ~val_mask

    tr_data = train_data[train_mask]
    tr_y = train_y[train_mask]
    val_data = train_data[val_mask]
    val_y = train_y[val_mask]

    print(f"[DATA] Samples — train: {len(tr_y)}, val: {len(val_y)}, test: {len(test_y)}")
    print(f"[DATA] Val subjects: {cfg.data.val_subjects}")
    print(f"[DATA] Input shape after KIT: (T={cfg.data.window_length}, C={cfg.data.total_input_channels})")
    print(f"[DATA] Patch shape: (P={cfg.patch.num_patches}, D={cfg.patch.patch_size * cfg.data.total_input_channels})")

    # ── Datasets ──────────────────────────────────────────────────────
    ps = cfg.patch.patch_size
    train_ds = HARDataset(tr_data, tr_y, ps)
    val_ds = HARDataset(val_data, val_y, ps)
    test_ds = HARDataset(test_data, test_y, ps)

    # ── DataLoaders ───────────────────────────────────────────────────
    common = dict(num_workers=cfg.num_workers, pin_memory=True)
    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(train_ds, batch_size=cfg.pretrain.batch_size,
                            shuffle=True,  drop_last=True, **common),
        "val":   DataLoader(val_ds,   batch_size=cfg.pretrain.batch_size,
                            shuffle=False, drop_last=False, **common),
        "test":  DataLoader(test_ds,  batch_size=cfg.pretrain.batch_size,
                            shuffle=False, drop_last=False, **common),
    }

    # Normalisation stats for reproducibility logging
    stats = {
        "train_subjects": np.unique(train_subj[train_mask]),
        "val_subjects": np.array(cfg.data.val_subjects),
        "num_train": len(tr_y),
        "num_val": len(val_y),
        "num_test": len(test_y),
    }

    return loaders, stats


# ════════════════════════════════════════════════════════════════════════
# 7.  Quick self-test
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = get_config()
    loaders, stats = build_dataloaders(cfg)

    for split, loader in loaders.items():
        patches, labels, raw = next(iter(loader))
        print(
            f"  [{split:>5}]  patches {patches.shape}  "
            f"labels {labels.shape}  raw {raw.shape}"
        )
