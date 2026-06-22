#!/usr/bin/env python3
"""Full test evaluation on the held-out (non-training) dataset.

This is the recommended entrypoint to evaluate generalization on combinations
that were *not present in training*.

Assumption
----------
The dataset root you pass is already in VAE-compatible, normalized format:
- heatmap: (1, 64, 64)
- Imp:     (3, 231)
- Occ_map: (52,)

Run
---
    python evaluation/vae/run_vae_eval_heldout.py \

Outputs
-------
Default output directory: evaluation/vae/heldout/
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path



# =============================================================================
# CONFIGURATION — edit these before running: python evaluation/vae/run_vae_eval_heldout.py
# =============================================================================

CHECKPOINT = Path("experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt")
DATASET_ROOT = Path("datasets/data_eval_norm")
OUT_DIR = Path("evaluation/vae/heldout")
BATCH_SIZE = 64
N_GEN = 2048
SHARED_TEMP = 1.5
SEED = 0
FORCE_CPU = False

# =============================================================================

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "experiments").is_dir() and (p / "datasets").is_dir():
            return p
    return here.parents[2]


def _dataset_ready(root: Path) -> bool:
    return (
        (root / "heatmap").is_dir()
        and (root / "Imp").is_dir()
        and (root / "Occ_map").is_dir()
        and any((root / "Occ_map").glob("sample_*.npy"))
    )


def _import_run_vae_eval(project_root: Path):
    module_path = project_root / "evaluation" / "vae" / "run_vae_eval.py"
    spec = importlib.util.spec_from_file_location("gan_run_vae_eval", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to import run_vae_eval from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    project_root = _project_root()
    os.chdir(project_root)

    dataset_root: Path = DATASET_ROOT

    if not _dataset_ready(dataset_root):
        raise SystemExit(f"Held-out dataset is missing or empty: {dataset_root}")

    print("Running VAE evaluation on FULL held-out set")
    run_vae_eval = _import_run_vae_eval(project_root)
    cfg = run_vae_eval.EvalConfig(
        checkpoint=CHECKPOINT,
        dataset_root=dataset_root,
        out_dir=OUT_DIR,
        n_eval=10**9,
        batch_size=int(BATCH_SIZE),
        n_gen=int(N_GEN),
        shared_temp=float(SHARED_TEMP),
        seed=int(SEED),
        force_cpu=bool(FORCE_CPU),
    )
    run_vae_eval.run_eval(cfg)

    print("\n✓ Held-out evaluation complete")
    print(f"  Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
