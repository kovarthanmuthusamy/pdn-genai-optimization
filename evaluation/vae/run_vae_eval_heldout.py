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
        --checkpoint experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt \
        --dataset-root datasets/data_eval_norm

Outputs
-------
Default output directory: evaluation/vae/heldout/
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt"),
    )

    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/data_eval_norm"),
        help="Path to a held-out dataset root that is already normalized for the VAE.",
    )

    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/vae/heldout"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-gen", type=int, default=2048)
    ap.add_argument("--shared-temp", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--force-cpu", action="store_true")

    args = ap.parse_args()

    project_root = _project_root()
    os.chdir(project_root)

    dataset_root: Path = args.dataset_root

    if not _dataset_ready(dataset_root):
        raise SystemExit(f"Held-out dataset is missing or empty: {dataset_root}")

    print("Running VAE evaluation on FULL held-out set")
    run_vae_eval = _import_run_vae_eval(project_root)
    cfg = run_vae_eval.EvalConfig(
        checkpoint=args.checkpoint,
        dataset_root=dataset_root,
        out_dir=args.out_dir,
        n_eval=10**9,
        batch_size=int(args.batch_size),
        n_gen=int(args.n_gen),
        shared_temp=float(args.shared_temp),
        seed=int(args.seed),
        force_cpu=bool(args.force_cpu),
    )
    run_vae_eval.run_eval(cfg)

    print("\n✓ Held-out evaluation complete")
    print(f"  Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
