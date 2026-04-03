"""
config.py — Central configuration for HAR-Mamba.

All hyperparameters, dataset paths, training schedules, and architectural
constants are consolidated here to keep the rest of the codebase free
from magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch


def _resolve_device() -> torch.device:
    """Pick the best available accelerator: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DataConfig:
    """Dataset & preprocessing knobs."""

    # ── Paths ──────────────────────────────────────────────────────────
    dataset_root: Path = Path("UCI HAR Dataset")
    inertial_signals: List[str] = field(
        default_factory=lambda: [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
    )

    # ── Signal params ──────────────────────────────────────────────────
    sampling_rate_hz: int = 50
    window_length: int = 128          # timesteps per window (2.56 s)
    num_raw_channels: int = 9         # 3×acc + 3×gyro + 3×total_acc

    # We will use body_acc (3ch) and body_gyro (3ch) for KIT computation.
    # total_acc is kept as additional context.
    acc_channel_indices: Tuple[int, ...] = (0, 1, 2)   # body_acc x,y,z
    gyro_channel_indices: Tuple[int, ...] = (3, 4, 5)  # body_gyro x,y,z

    # After KIT concatenation the input grows by 3 channels.
    kit_channels: int = 3
    @property
    def total_input_channels(self) -> int:
        """Raw channels + KIT cross-product channels."""
        return self.num_raw_channels + self.kit_channels  # 12

    # ── Activity labels ────────────────────────────────────────────────
    num_classes: int = 6
    activity_names: List[str] = field(
        default_factory=lambda: [
            "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
            "SITTING", "STANDING", "LAYING",
        ]
    )

    # ── Subject-independent split ──────────────────────────────────────
    # UCI already splits by subject; we carve a validation set from train.
    val_subjects: List[int] = field(default_factory=lambda: [21, 22, 23, 25])

    # ── Preprocessing ──────────────────────────────────────────────────
    apply_lowpass: bool = True
    lowpass_cutoff_hz: float = 20.0
    lowpass_order: int = 3


@dataclass
class PatchConfig:
    """Patchification parameters."""

    patch_size: int = 16              # each patch covers 16 timesteps
    # Derived at runtime:
    @property
    def num_patches(self) -> int:
        return 128 // self.patch_size  # 128 / 16 = 8


@dataclass
class MambaConfig:
    """Bidirectional Mamba block parameters."""

    d_model: int = 128                # hidden dimension
    d_state: int = 16                 # SSM state dimension
    d_conv: int = 4                   # local convolution width
    expand: int = 2                   # expansion factor for inner dim
    num_layers: int = 4               # stacked Bi-Mamba blocks
    dropout: float = 0.1


@dataclass
class MaskingConfig:
    """PDGM (Probabilistic Dynamics-Guided Masking) settings."""

    mask_ratio: float = 0.50          # fraction of patches to mask
    temperature: float = 1.0          # softmax temperature for sampling
    min_prob: float = 0.05            # floor probability for static patches


@dataclass
class PretrainConfig:
    """Self-supervised pretraining schedule."""

    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    scheduler: str = "cosine"         # 'cosine' | 'step'
    decoder_dim: int = 64
    decoder_layers: int = 1
    checkpoint_dir: Path = Path("checkpoints")
    best_model_name: str = "pretrained_encoder.pth"


@dataclass
class FinetuneConfig:
    """Downstream fine-tuning schedule."""

    epochs: int = 40
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    scheduler: str = "cosine"
    label_fractions: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.35, 0.45, 1.0]
    )
    freeze_encoder: bool = False      # True → linear probe


@dataclass
class AblationConfig:
    """Ablation study flags — toggle individual components on/off."""

    use_kit: bool = True              # Pillar 1: KIT physics prior
    use_pdgm: bool = True             # Pillar 2: dynamics-guided masking
    use_bidirectional: bool = True    # Pillar 3: bidirectional SSM
    use_pretraining: bool = True      # Self-supervised pretraining phase


@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    device: torch.device = field(default_factory=_resolve_device)
    seed: int = 42
    num_workers: int = 0
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        self.pretrain.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ── Convenience singleton ──────────────────────────────────────────────
def get_config() -> Config:
    """Return a default Config instance (override fields as needed)."""
    return Config()
