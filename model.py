"""
model.py — HAR-Mamba Architecture.

This module implements the three core architectural components:

1.  **BiMambaBlock** — A Bidirectional State Space Model (SSM) block that
    processes sequences in *both* forward and backward directions, fusing
    the two via a gated projection.  Unlike Transformers, SSMs operate
    in O(T) linear complexity w.r.t. sequence length.

2.  **ProbabilisticDynamicsGuidedMasking (PDGM)** — A masking strategy that
    preferentially masks high-dynamic patches (measured by L2-norm of
    temporal gradients).  This forces the encoder to reconstruct the
    *most informative* motion transitions from static context.

3.  **MaskedMambaAutoencoder** — The full self-supervised model:
        Encoder  : Patch Embedding → Bi-Mamba stack → latent
        PDGM     : selects which patches to mask
        Decoder  : lightweight MLP reconstructs masked patches

Math & Notation
───────────────
• B = batch size
• P = number of patches
• D = patch_dim = patch_size × total_input_channels
• H = d_model (hidden dim)
• S = d_state (SSM state)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, get_config


# ════════════════════════════════════════════════════════════════════════
# Helper: Causal 1-D Convolution
# ════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """Causal 1-D convolution with left-padding so output T == input T."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        (B, C_out, T)
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# ════════════════════════════════════════════════════════════════════════
# Selective SSM Scan (Simplified S6)
# ════════════════════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model core (S6-style).

    This is the heart of each Mamba block.  For every timestep *t* the
    model computes:
        x̄(t) = A·x(t-1) + B·u(t)
        y(t) = C·x̄(t)
    where A, B, C are *input-dependent* (selective) projections.

    Parameters
    ----------
    d_inner : int
        Inner working dimension (d_model × expand).
    d_state : int
        Latent state dimensionality per channel.
    """

    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Input-dependent projections  B, C, Δ
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_dt = nn.Linear(d_inner, d_inner, bias=True)

        # Learnable log(A) — initialised to HiPPO matrix diagonal
        log_A = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
        ).unsqueeze(0).expand(d_inner, -1)  # (d_inner, d_state)
        self.log_A = nn.Parameter(log_A)

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run the selective scan.

        Parameters
        ----------
        u : (B, T, d_inner)

        Returns
        -------
        y : (B, T, d_inner)
        """
        B_batch, T, D = u.shape

        # ── Compute input-dependent params ────────────────────────────
        B_ssm = self.proj_B(u)                         # (B, T, S)
        C_ssm = self.proj_C(u)                         # (B, T, S)
        dt = F.softplus(self.proj_dt(u))               # (B, T, D)

        # Discretize A
        A = -torch.exp(self.log_A)                     # (D, S)
        dA = torch.exp(dt.unsqueeze(-1) * A)           # (B, T, D, S)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)    # (B, T, D, S)

        # ── Optimised scan (pre-alloc + fused input) ─────────────────
        dB_u = dB * u.unsqueeze(-1)                    # (B, T, D, S)
        x = torch.zeros(B_batch, D, self.d_state, device=u.device, dtype=u.dtype)
        y = torch.empty(B_batch, T, D, device=u.device, dtype=u.dtype)
        for t in range(T):
            x = dA[:, t] * x + dB_u[:, t]             # (B, D, S)
            y[:, t] = (x * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y


# ════════════════════════════════════════════════════════════════════════
# Single Mamba Block (Unidirectional)
# ════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """One Mamba block: Norm → Linear ↑ → Conv1d → SSM → Gate → Linear ↓.

    Architecture per Gu & Dao (Mamba, 2023):
        u → [in_proj → z, x]
             x → conv1d → SiLU → SSM → * gate(z) → out_proj → residual
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv = CausalConv1d(d_inner, d_inner, kernel_size=d_conv)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # ── Split into gate & main branch ─────────────────────────────
        xz = self.in_proj(x)                          # (B, T, 2·d_inner)
        x_branch, z = xz.chunk(2, dim=-1)             # each (B, T, d_inner)

        # ── Conv → activation → SSM ─────────────────────────────────
        x_branch = x_branch.transpose(1, 2)           # (B, d_inner, T)
        x_branch = self.conv(x_branch)
        x_branch = x_branch.transpose(1, 2)           # (B, T, d_inner)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)

        # ── Gated merge ──────────────────────────────────────────────
        x_branch = x_branch * F.silu(z)
        out = self.out_proj(x_branch)

        return out + residual


# ════════════════════════════════════════════════════════════════════════
# Bidirectional Mamba Block
# ════════════════════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: process the sequence **past→future** and
    **future→past**, then fuse with a learnable gated projection.

    This gives the encoder context from both temporal directions —
    critical for reconstructing masked patches that may lie anywhere
    in the window.

    Architecture
    ────────────
        Input ─┬── MambaBlock_fwd ──┐
               │                     ├── Concat → Gate → Linear → Output
               └── Flip → MambaBlock_bwd → Flip ──┘
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.fwd_block = MambaBlock(d_model, d_state, d_conv, expand)
        self.bwd_block = MambaBlock(d_model, d_state, d_conv, expand)

        # Learnable gate that weights fwd vs bwd contributions
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.merge = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, P, d_model)

        Returns
        -------
        (B, P, d_model)
        """
        h_fwd = self.fwd_block(x)                       # (B, P, H)
        h_bwd = self.bwd_block(x.flip(dims=[1])).flip(dims=[1])  # (B, P, H)

        combined = torch.cat([h_fwd, h_bwd], dim=-1)    # (B, P, 2H)
        gate = self.gate(combined)                        # (B, P, H)  ∈ [0,1]
        merged = self.merge(combined)                     # (B, P, H)

        return gate * h_fwd + (1 - gate) * h_bwd + merged * 0.1  # gated residual


# ════════════════════════════════════════════════════════════════════════
# PDGM — Probabilistic Dynamics-Guided Masking
# ════════════════════════════════════════════════════════════════════════

class PDGM(nn.Module):
    """Probabilistic Dynamics-Guided Masking.

    Instead of masking patches *uniformly at random* (as in MAE),
    we mask patches with probability proportional to their **motion
    intensity**: patches that capture rapid accelerations / rotational
    changes are *preferentially* masked.

    Rationale
    ─────────
    If the model can reconstruct the most dynamic portions of a
    movement from the remaining static context, it must learn deep
    kinematic semantics — not just surface-level patterns.

    Algorithm
    ─────────
    1. For each patch *p*, compute the mean L2 norm of the temporal
       gradient:  I_p = mean_t ||Δx_t||₂  where t ∈ patch *p*.
    2. Convert to probabilities via softmax(I / τ) with temperature τ.
    3. Clamp probabilities to ``[min_prob, 1]`` so fully static patches
       still have a non-zero chance of being masked.
    4. Sample ``mask_ratio × P`` patches *without replacement* from the
       categorical distribution.

    Parameters
    ----------
    cfg : Config
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.mask_ratio = cfg.masking.mask_ratio
        self.temperature = cfg.masking.temperature
        self.min_prob = cfg.masking.min_prob
        self.patch_size = cfg.patch.patch_size
        self.total_channels = cfg.data.total_input_channels

    def _compute_patch_intensity(self, raw_window: torch.Tensor) -> torch.Tensor:
        """Compute per-patch motion intensity from the raw window.

        Parameters
        ----------
        raw_window : (B, T, C)   — full raw sensor window

        Returns
        -------
        intensity : (B, P)
        """
        # Temporal gradient: Δx_t = x_{t+1} − x_t
        grad = raw_window[:, 1:, :] - raw_window[:, :-1, :]  # (B, T-1, C)
        grad_norm = grad.norm(dim=-1)                          # (B, T-1)

        # Pad so length stays T (repeat last value)
        grad_norm = F.pad(grad_norm, (0, 1), mode="replicate")  # (B, T)

        # Reshape into patches and take mean intensity per patch
        B, T = grad_norm.shape
        P = T // self.patch_size
        grad_patches = grad_norm.reshape(B, P, self.patch_size)  # (B, P, ps)
        intensity = grad_patches.mean(dim=-1)                     # (B, P)

        return intensity

    def forward(
        self,
        patches: torch.Tensor,
        raw_window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply dynamics-guided masking.

        Parameters
        ----------
        patches    : (B, P, D) — patchified input
        raw_window : (B, T, C) — raw sensor window for intensity calc

        Returns
        -------
        visible_patches : (B, P_vis, D)
        mask             : (B, P)  — bool, True = masked
        masked_indices   : (B, num_masked) — indices of masked patches
        """
        B, P, D = patches.shape
        num_masked = int(self.mask_ratio * P)
        num_visible = P - num_masked

        # ── Step 1: Compute intensity ─────────────────────────────────
        intensity = self._compute_patch_intensity(raw_window)  # (B, P)

        # ── Step 2: Convert to sampling probabilities ─────────────────
        probs = F.softmax(intensity / self.temperature, dim=-1)  # (B, P)
        probs = probs.clamp(min=self.min_prob)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # re-normalise

        # ── Step 3: Sample without replacement ────────────────────────
        # Use Gumbel-top-k trick for differentiability-friendly sampling
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(probs).clamp(min=1e-8)
        ))
        perturbed_log_probs = torch.log(probs + 1e-8) + gumbel_noise
        _, indices = perturbed_log_probs.topk(num_masked, dim=-1)  # (B, num_masked)

        # ── Step 4: Build boolean mask ────────────────────────────────
        mask = torch.zeros(B, P, dtype=torch.bool, device=patches.device)
        mask.scatter_(1, indices, True)

        # ── Step 5: Extract visible patches ───────────────────────────
        visible_mask = ~mask  # (B, P)
        # Gather visible patches (keep order)
        visible_indices = visible_mask.nonzero(as_tuple=False)  # (total_vis, 2)

        # Reshape: (B, num_visible, D)
        visible_patches = patches[~mask].reshape(B, num_visible, D)

        return visible_patches, mask, indices


# ════════════════════════════════════════════════════════════════════════
# Masked Mamba Autoencoder
# ════════════════════════════════════════════════════════════════════════

class MaskedMambaAutoencoder(nn.Module):
    """Self-supervised HAR model with Bi-Mamba encoder and MLP decoder.

    Architecture Overview
    ─────────────────────
    ┌──────────────────────────────────────────────────────────────┐
    │  Input (B, P, D)                                            │
    │       ↓                                                      │
    │  PDGM → visible_patches (B, P_vis, D)                      │
    │       ↓                                                      │
    │  Patch Embedding → (B, P_vis, H)                            │
    │       ↓                                                      │
    │  + Positional Embedding                                     │
    │       ↓                                                      │
    │  Bi-Mamba Encoder ×L → latent (B, P_vis, H)                │
    │       ↓                                                      │
    │  Insert mask tokens at masked positions → (B, P, H)         │
    │       ↓                                                      │
    │  MLP Decoder → reconstructed (B, P, D)                      │
    │       ↓                                                      │
    │  MSE Loss (on masked patches only)                          │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        P = cfg.patch.num_patches
        D = cfg.patch.patch_size * cfg.data.total_input_channels  # patch_dim
        H = cfg.mamba.d_model

        self.patch_dim = D
        self.num_patches = P

        # ── Encoder components ────────────────────────────────────────
        self.patch_embed = nn.Linear(D, H)
        self.pos_embed = nn.Parameter(torch.randn(1, P, H) * 0.02)
        self.pdgm = PDGM(cfg)

        self.encoder_blocks = nn.ModuleList([
            BiMambaBlock(
                d_model=cfg.mamba.d_model,
                d_state=cfg.mamba.d_state,
                d_conv=cfg.mamba.d_conv,
                expand=cfg.mamba.expand,
            )
            for _ in range(cfg.mamba.num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(H)
        self.encoder_dropout = nn.Dropout(cfg.mamba.dropout)

        # ── Mask token ────────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        # ── Decoder ──────────────────────────────────────────────────
        decoder_dim = cfg.pretrain.decoder_dim
        decoder_layers = []
        decoder_layers.append(nn.Linear(H, decoder_dim))
        decoder_layers.append(nn.GELU())
        for _ in range(cfg.pretrain.decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_dim, decoder_dim))
            decoder_layers.append(nn.GELU())
        decoder_layers.append(nn.Linear(decoder_dim, D))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode *all* patches (no masking) — used at fine-tune time.

        Parameters
        ----------
        patches : (B, P, D)

        Returns
        -------
        latent : (B, P, H)
        """
        x = self.patch_embed(patches)          # (B, P, H)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)

    def forward(
        self,
        patches: torch.Tensor,
        raw_window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for self-supervised pretraining.

        Parameters
        ----------
        patches    : (B, P, D)
        raw_window : (B, T, C)

        Returns
        -------
        reconstruction : (B, P, D) — reconstructed patches
        mask           : (B, P)    — True at masked positions
        loss           : scalar    — MSE on masked patches only
        """
        B, P, D = patches.shape

        # ── PDGM masking ────────────────────────────────────────────
        visible_patches, mask, masked_indices = self.pdgm(patches, raw_window)

        # ── Encode visible patches ────────────────────────────────────
        # Get positional indices of visible patches
        visible_pos_ids = (~mask).nonzero(as_tuple=True)[1].reshape(B, -1)  # (B, P_vis)

        x = self.patch_embed(visible_patches)               # (B, P_vis, H)

        # Add position embeddings for visible positions
        pos_emb_vis = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=visible_pos_ids.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )
        x = x + pos_emb_vis
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)                             # (B, P_vis, H)

        # ── Insert mask tokens for full sequence decoder ──────────────
        full_seq = self.mask_token.expand(B, P, -1).clone()  # (B, P, H)

        # Place encoded visible tokens at their positions
        vis_idx_expanded = visible_pos_ids.unsqueeze(-1).expand(-1, -1, x.size(-1))
        full_seq.scatter_(1, vis_idx_expanded, x)

        # Add full positional embeddings
        full_seq = full_seq + self.pos_embed

        # ── Decode ───────────────────────────────────────────────────
        reconstruction = self.decoder(full_seq)              # (B, P, D)

        # ── Loss: MSE on masked patches only ─────────────────────────
        target = patches                                      # (B, P, D)
        # mask shape: (B, P) — expand to (B, P, D)
        mask_expanded = mask.unsqueeze(-1).expand_as(target).float()

        # Masked MSE
        diff = (reconstruction - target) ** 2
        loss = (diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

        return reconstruction, mask, loss


# ════════════════════════════════════════════════════════════════════════
# Classification Head (for fine-tuning)
# ════════════════════════════════════════════════════════════════════════

class HARClassifier(nn.Module):
    """Bi-Mamba Encoder + Classification Head.

    Attach this on top of a pretrained ``MaskedMambaAutoencoder.encode``
    for downstream activity recognition.

    Architecture
    ────────────
        patches → Encoder → mean pool → LayerNorm → MLP → logits
    """

    def __init__(self, encoder: MaskedMambaAutoencoder, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.mamba.d_model

        # Steal encoder components
        self.patch_embed = encoder.patch_embed
        self.pos_embed = encoder.pos_embed
        self.encoder_blocks = encoder.encoder_blocks
        self.encoder_norm = encoder.encoder_norm
        self.encoder_dropout = encoder.encoder_dropout

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(H, cfg.data.num_classes),
        )

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (for linear probing)."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.encoder_norm.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters (for full fine-tuning)."""
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.pos_embed.requires_grad = True
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.encoder_norm.parameters():
            param.requires_grad = True

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patches : (B, P, D)

        Returns
        -------
        logits : (B, num_classes)
        """
        x = self.patch_embed(patches)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)                  # (B, P, H)
        x = x.mean(dim=1)                          # (B, H) — global average pool
        logits = self.head(x)                       # (B, C)
        return logits


# ════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = get_config()
    device = cfg.device
    print(f"Device: {device}")

    B = 4
    P = cfg.patch.num_patches        # 8
    D = cfg.patch.patch_size * cfg.data.total_input_channels  # 16 × 12 = 192
    T = cfg.data.window_length       # 128
    C = cfg.data.total_input_channels  # 12

    patches = torch.randn(B, P, D, device=device)
    raw_window = torch.randn(B, T, C, device=device)

    # ── Test Autoencoder ──────────────────────────────────────────────
    model = MaskedMambaAutoencoder(cfg).to(device)
    reconstruction, mask, loss = model(patches, raw_window)
    print(f"\n=== MaskedMambaAutoencoder ===")
    print(f"  Input patches:  {patches.shape}")
    print(f"  Reconstruction: {reconstruction.shape}")
    print(f"  Mask:           {mask.shape}  (masked={mask.sum().item()}/{mask.numel()})")
    print(f"  Loss:           {loss.item():.4f}")

    # Verify gradients flow
    loss.backward()
    total_grad = sum(
        p.grad.abs().sum().item()
        for p in model.parameters() if p.grad is not None
    )
    print(f"  Total |grad|:   {total_grad:.4f}  ✓ (gradients flow)")

    # ── Test Classifier ───────────────────────────────────────────────
    classifier = HARClassifier(model, cfg).to(device)
    logits = classifier(patches)
    print(f"\n=== HARClassifier ===")
    print(f"  Logits:         {logits.shape}")
    print(f"  Predicted:      {logits.argmax(dim=-1).tolist()}")

    # Count parameters
    enc_params = sum(p.numel() for p in model.parameters())
    cls_params = sum(p.numel() for p in classifier.head.parameters())
    print(f"\n  Encoder params: {enc_params:,}")
    print(f"  Head params:    {cls_params:,}")
    print(f"  Total params:   {enc_params + cls_params:,}")
