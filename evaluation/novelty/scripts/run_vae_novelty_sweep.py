#!/usr/bin/env python3
"""Generate + score novelty for many K values.

This is the "include most K and increase N" runner.

Outputs
- Per-K folders under:
    <out-root>/K{K}/data_sample_i/{heatmap_zscore.npy, occupancy_map.npy, impedance_profile.npy}
  plus per-K `novelty_report.csv`.
- One aggregate CSV:
    <out-root>/novelty_sweep_summary.csv

Example
  python scripts/run_vae_novelty_sweep.py --N 100 --k-start 1 --k-end 52

Tip
- Use `--max-train` to cap dataset size per K for speed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "experiments").is_dir() and (p / "datasets").is_dir():
            return p
    return here.parents[1]


def _load_engine(*, checkpoint: str, latent_dim: int, force_cpu: bool):
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.exp027_sigma_reg_tuning.codes.inference_vae import VAEInference  # noqa: WPS433

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.chdir(project_root)
    engine = VAEInference(checkpoint_path=checkpoint, latent_dim=latent_dim, device=device)
    engine.load_latent_stats(None)
    return engine


def _generate_one_k(
    *,
    engine,
    out_dir: Path,
    k_value: int,
    n: int,
    shared_temp: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        heatmap_zscore, occ_prob, impedance_norm = engine.model.inference(
            n,
            engine.device,
            K=k_value,
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=shared_temp,
        )

        occ_bin = torch.zeros_like(occ_prob)
        if k_value > 0:
            topk_idx = occ_prob.topk(min(k_value, occ_prob.shape[-1]), dim=-1).indices
            occ_bin.scatter_(-1, topk_idx, 1.0)

        impedance_log, _, _, _ = engine._denorm_impedance(impedance_norm)

    hm_np = heatmap_zscore.cpu().numpy()
    occ_np = occ_bin.cpu().numpy().astype(np.int8)
    imp_np = impedance_log.cpu().numpy()

    np.save(out_dir / "occupancy.npy", occ_np)
    for i in range(n):
        sample_dir = out_dir / f"data_sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        np.save(sample_dir / "heatmap_zscore.npy", hm_np[i])
        np.save(sample_dir / "occupancy_map.npy", occ_np[i])
        np.save(sample_dir / "impedance_profile.npy", imp_np[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--N", type=int, default=100, help="Generated samples per K")
    ap.add_argument("--k-start", type=int, default=1)
    ap.add_argument("--k-end", type=int, default=52)
    ap.add_argument("--k-step", type=int, default=1)
    ap.add_argument("--shared-temp", type=float, default=1.5)

    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/data_norm"))
    ap.add_argument("--hm-pool", type=int, default=16)
    ap.add_argument("--baseline-n", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--score-only",
        action="store_true",
        help="Do not generate; only (re)score existing K folders under out-root",
    )

    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root (default: evaluation/novelty/runs/novelty_sweep_N{N})",
    )
    args = ap.parse_args()

    if not args.score_only and args.N <= 0:
        raise SystemExit("--N must be > 0 unless --score-only is set")
    if not (0 <= args.k_start <= 52 and 0 <= args.k_end <= 52 and args.k_step > 0):
        raise SystemExit("Invalid K range")

    out_root = args.out_root
    if out_root is None:
        out_root = Path("evaluation/novelty/runs") / f"novelty_sweep_N{args.N}"

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import scorer (uses K cache internally; building once makes the sweep faster)
    report_path = project_root / "evaluation" / "novelty" / "scripts" / "vae_novelty_report.py"
    spec = importlib.util.spec_from_file_location("vae_novelty_report", report_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import novelty scorer from {report_path}")
    report_mod = importlib.util.module_from_spec(spec)
    sys.modules["vae_novelty_report"] = report_mod
    spec.loader.exec_module(report_mod)
    _load_or_build_k_cache = getattr(report_mod, "_load_or_build_k_cache")
    score_generated_against_dataset = getattr(report_mod, "score_generated_against_dataset")

    print(f"Out root: {out_root}")
    print("Building / loading dataset K cache...")
    _load_or_build_k_cache(args.dataset_root)

    engine = None
    if not args.score_only:
        print("Loading VAE inference engine...")
        engine = _load_engine(checkpoint=args.checkpoint, latent_dim=args.latent_dim, force_cpu=bool(args.force_cpu))

    ks = list(range(args.k_start, args.k_end + 1, args.k_step))
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for idx, k in enumerate(ks, start=1):
        k_dir = out_root / f"K{k}"
        if not args.score_only:
            print(f"\n[{idx}/{len(ks)}] K={k}: generating N={args.N}")
            _generate_one_k(engine=engine, out_dir=k_dir, k_value=k, n=args.N, shared_temp=args.shared_temp)
        else:
            if not k_dir.exists():
                print(f"\n[{idx}/{len(ks)}] K={k}: missing folder, skipping: {k_dir}")
                continue

        print(f"[{idx}/{len(ks)}] K={k}: scoring")
        summary = score_generated_against_dataset(
            gen_dir=k_dir,
            dataset_root=args.dataset_root,
            k_filter=int(k),
            hm_pool=args.hm_pool,
            baseline_n=args.baseline_n,
            seed=args.seed,
            max_gen=None,
            max_train=args.max_train,
            out_csv=k_dir / "novelty_report.csv",
        )

        base_med = summary.baseline_combined["median"]
        gen_med = summary.gen_to_train_combined["median"]
        ratio = (gen_med / base_med) if (base_med not in (None, 0.0) and gen_med is not None) else None

        train_to_gen = getattr(summary, "train_to_gen_combined", None)
        train_to_gen_med = train_to_gen["median"] if train_to_gen is not None else None

        summary_rows.append(
            {
                "K": k,
                "n_gen": summary.n_gen,
                "n_train": summary.n_train,
                "baseline_combined_median": base_med,
                "gen_to_train_combined_median": gen_med,
                "median_ratio_gen_over_base": ratio,
                "train_to_gen_combined_median": train_to_gen_med,
                "train_to_gen_cov_at_base_p90": getattr(summary, "train_to_gen_coverage_at_base_p90", None),
                "mem_suspect_rate_p01": getattr(summary, "mem_suspect_rate_p01", None),
                "mem_suspect_rate_p05": getattr(summary, "mem_suspect_rate_p05", None),
                "mem_suspect_rate_p10": getattr(summary, "mem_suspect_rate_p10", None),
                "occ_exact": summary.occupancy_exact_count,
                "occ_ham_median": summary.gen_to_train_occupancy_hamming["median"],
                "hm_mse_median": summary.gen_to_train_heatmap_mse["median"],
                "imp_mse_median": summary.gen_to_train_impedance_mse["median"],
                "train_occ_unique": getattr(summary, "train_occ_unique", None),
                "gen_occ_unique": getattr(summary, "gen_occ_unique", None),
                "gen_occ_new_unique": getattr(summary, "gen_occ_new_unique", None),
                "gen_occ_new_rate": getattr(summary, "gen_occ_new_rate", None),
                "gen_occ_mode_frac": getattr(summary, "gen_occ_mode_frac", None),
                "occ_jsd": getattr(summary, "occ_jsd", None),
            }
        )

    if not summary_rows:
        raise SystemExit("No K folders were scored (check --out-root and K range)")

    summary_path = out_root / "novelty_sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\n✓ Sweep complete")
    print(f"  Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
