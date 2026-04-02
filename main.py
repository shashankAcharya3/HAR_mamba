"""
main.py — HAR-Mamba: Full Pipeline Runner.

Usage
─────
    python main.py                      # Run everything: pretrain → finetune → eval
    python main.py --phase pretrain     # Only pretraining
    python main.py --phase finetune     # Only fine-tuning (full)
    python main.py --phase eval         # Only evaluation
    python main.py --phase scarcity     # Data scarcity experiment
    python main.py --phase linear       # Linear probe only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import get_config
from utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HAR-Mamba: Physics-Informed Self-Supervised HAR via Bidirectional SSMs"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "pretrain", "finetune", "linear", "eval", "scarcity"],
        help="Which phase(s) to run.",
    )
    args = parser.parse_args()

    cfg = get_config()
    set_seed(cfg.seed)

    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║   HAR-Mamba                                               ║
    ║   Physics-Informed Self-Supervised HAR                     ║
    ║   via Bidirectional State Space Models                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    print(f"  Device : {cfg.device}")
    print(f"  Seed   : {cfg.seed}")
    print(f"  Dataset: {cfg.data.dataset_root.resolve()}")
    print()

    t_start = time.time()

    # ── Phase 1: Pretraining ──────────────────────────────────────────
    if args.phase in ("all", "pretrain"):
        from pretrain import pretrain
        pretrain(cfg)

    # ── Phase 2a: Full Fine-Tuning ────────────────────────────────────
    if args.phase in ("all", "finetune"):
        from finetune import finetune
        finetune(cfg, mode="full", label_fraction=1.0)

    # ── Phase 2b: Linear Probing ──────────────────────────────────────
    if args.phase in ("all", "linear"):
        from finetune import finetune
        finetune(cfg, mode="linear_probe", label_fraction=1.0)

    # ── Phase 3: Evaluation ───────────────────────────────────────────
    if args.phase in ("all", "eval"):
        from eval import evaluate
        evaluate(cfg)

    # ── Phase 4: Data Scarcity ────────────────────────────────────────
    if args.phase in ("scarcity",):
        from finetune import data_scarcity_experiment
        data_scarcity_experiment(cfg)

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed / 60:.1f} minutes")
    print(f"  Results in:    {cfg.results_dir.resolve()}")
    print(f"  Checkpoints:   {cfg.pretrain.checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
