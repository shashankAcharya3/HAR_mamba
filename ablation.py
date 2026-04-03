"""
ablation.py — Ablation Study Runner for HAR-Mamba.

Systematically tests the contribution of each novel component by
disabling them one at a time and comparing against the full model.

Experiments
───────────
  1. Full Model          — All components active (baseline)
  2. w/o KIT             — Remove Kinematic Interaction Tensor
  3. w/o PDGM            — Replace dynamics-guided masking with random
  4. Unidirectional      — Remove backward Mamba pass
  5. No Pretraining      — Skip SSL, train from scratch
  6. Vanilla Bi-Mamba    — Remove both KIT and PDGM

Usage
─────
    python ablation.py
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import Config, get_config
from utils import set_seed, log_metrics_csv


# ════════════════════════════════════════════════════════════════════════
# Experiment Definitions
# ════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    {
        "name": "full_model",
        "description": "Full HAR-Mamba (all components)",
        "use_kit": True,
        "use_pdgm": True,
        "use_bidirectional": True,
        "use_pretraining": True,
    },
    {
        "name": "no_kit",
        "description": "w/o KIT (no physics prior)",
        "use_kit": False,
        "use_pdgm": True,
        "use_bidirectional": True,
        "use_pretraining": True,
    },
    {
        "name": "random_masking",
        "description": "w/o PDGM (uniform random masking)",
        "use_kit": True,
        "use_pdgm": False,
        "use_bidirectional": True,
        "use_pretraining": True,
    },
    {
        "name": "unidirectional",
        "description": "Unidirectional Mamba (forward only)",
        "use_kit": True,
        "use_pdgm": True,
        "use_bidirectional": False,
        "use_pretraining": True,
    },
    {
        "name": "no_pretraining",
        "description": "No self-supervised pretraining (random init)",
        "use_kit": True,
        "use_pdgm": True,
        "use_bidirectional": True,
        "use_pretraining": False,
    },
    {
        "name": "vanilla_bimamba",
        "description": "Vanilla Bi-Mamba (no KIT, no PDGM)",
        "use_kit": False,
        "use_pdgm": False,
        "use_bidirectional": True,
        "use_pretraining": True,
    },
]


# ════════════════════════════════════════════════════════════════════════
# Single Experiment Runner
# ════════════════════════════════════════════════════════════════════════

def run_single_experiment(experiment: dict) -> Dict[str, float]:
    """Run one ablation experiment: pretrain → finetune → return test metrics.

    Each experiment gets its own checkpoint and results subdirectory to
    avoid conflicts with other experiments.
    """
    name = experiment["name"]
    print(f"\n{'█' * 64}")
    print(f"  ABLATION: {experiment['description']}")
    print(f"  Config:   KIT={experiment['use_kit']}  PDGM={experiment['use_pdgm']}  "
          f"BiDir={experiment['use_bidirectional']}  Pretrain={experiment['use_pretraining']}")
    print(f"{'█' * 64}\n")

    # ── Build isolated config for this experiment ─────────────────────
    cfg = get_config()
    set_seed(cfg.seed)

    # Set ablation flags
    cfg.ablation.use_kit = experiment["use_kit"]
    cfg.ablation.use_pdgm = experiment["use_pdgm"]
    cfg.ablation.use_bidirectional = experiment["use_bidirectional"]
    cfg.ablation.use_pretraining = experiment["use_pretraining"]

    # When KIT is disabled, adjust channel count so model dimensions match
    if not cfg.ablation.use_kit:
        cfg.data.kit_channels = 0  # total_input_channels → 9

    # Isolate outputs per experiment
    cfg.pretrain.checkpoint_dir = Path(f"checkpoints/ablation_{name}")
    cfg.results_dir = Path(f"results/ablation_{name}")
    cfg.pretrain.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Pretrain (if enabled) ────────────────────────────────
    if cfg.ablation.use_pretraining:
        from pretrain import pretrain
        pretrain(cfg)

    # ── Phase 2: Full Fine-Tune ───────────────────────────────────────
    from finetune import finetune
    test_metrics = finetune(cfg, mode="full", label_fraction=1.0)

    return test_metrics


# ════════════════════════════════════════════════════════════════════════
# Main Runner
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║   HAR-Mamba  ·  Ablation Study                            ║
    ║   Testing contribution of each novel component            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    all_results: List[dict] = []
    summary_csv = Path("results/ablation_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    t_global = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n  ▶ Experiment {i}/{len(EXPERIMENTS)}: {exp['name']}")

        t0 = time.time()
        test_metrics = run_single_experiment(exp)
        elapsed = (time.time() - t0) / 60

        result = {
            "experiment": exp["name"],
            "description": exp["description"],
            "use_kit": exp["use_kit"],
            "use_pdgm": exp["use_pdgm"],
            "use_bidirectional": exp["use_bidirectional"],
            "use_pretraining": exp["use_pretraining"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "time_min": f"{elapsed:.1f}",
        }
        all_results.append(result)

        # Save incrementally
        log_metrics_csv(summary_csv, {
            "experiment": exp["name"],
            "KIT": str(exp["use_kit"]),
            "PDGM": str(exp["use_pdgm"]),
            "BiDir": str(exp["use_bidirectional"]),
            "Pretrain": str(exp["use_pretraining"]),
            "accuracy": f"{test_metrics['accuracy']:.4f}",
            "macro_f1": f"{test_metrics['macro_f1']:.4f}",
            "precision": f"{test_metrics['precision']:.4f}",
            "recall": f"{test_metrics['recall']:.4f}",
            "time_min": f"{elapsed:.1f}",
        })

    # ── Final Summary ─────────────────────────────────────────────────
    total_time = (time.time() - t_global) / 60

    print(f"\n{'═' * 80}")
    print(f"  ABLATION STUDY — FINAL RESULTS")
    print(f"{'═' * 80}")
    print(f"  {'Experiment':<22s}  {'KIT':>4s}  {'PDGM':>5s}  {'BiDir':>5s}  "
          f"{'Pre':>4s}  {'Accuracy':>9s}  {'F1':>7s}")
    print(f"  {'─' * 70}")

    for r in all_results:
        kit_sym = "✅" if r["use_kit"] else "❌"
        pdgm_sym = "✅" if r["use_pdgm"] else "❌"
        bidir_sym = "✅" if r["use_bidirectional"] else "❌"
        pre_sym = "✅" if r["use_pretraining"] else "❌"

        print(f"  {r['experiment']:<22s}  {kit_sym:>4s}  {pdgm_sym:>5s}  {bidir_sym:>5s}  "
              f"{pre_sym:>4s}  {r['accuracy']:>8.4f}  {r['macro_f1']:>7.4f}")

    print(f"\n  Total time: {total_time:.1f} minutes")
    print(f"  Results saved to: {summary_csv.resolve()}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
