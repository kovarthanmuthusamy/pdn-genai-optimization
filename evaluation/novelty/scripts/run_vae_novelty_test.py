#!/usr/bin/env python3
"""End-to-end novelty test: generate N samples + score vs dataset.

This script:
1) Uses the exp027 inference code to generate a small batch of samples for a given K.
2) Saves them in the same on-disk layout as `scrap/generate_samples_and_peb.py`.
3) Runs `scripts/vae_novelty_report.py` to compute nearest-neighbor scores and write a CSV.
4) Writes a short markdown summary next to the CSV.

Example
  python scripts/run_vae_novelty_test.py --K 5 --N 50

Notes
- This does NOT require ECADStar / .peb.
- For scoring, we strongly recommend filtering the dataset to the same K.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
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


def _python_exe() -> str:
    return sys.executable


def _save_generated_samples(
    *,
    out_dir: Path,
    checkpoint_path: str,
    latent_dim: int,
    k_value: int,
    num_samples: int,
    shared_temp: float,
    force_cpu: bool,
) -> None:
    # Ensure project root is importable
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.exp027_sigma_reg_tuning.codes.inference_vae import VAEInference  # noqa: WPS433

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.chdir(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = VAEInference(checkpoint_path=checkpoint_path, latent_dim=latent_dim, device=device)
    engine.load_latent_stats(None)

    with torch.no_grad():
        heatmap_zscore, occ_prob, impedance_norm = engine.model.inference(
            num_samples,
            engine.device,
            K=k_value,
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=shared_temp,
        )

        # Activate exactly K highest-probability slots per sample
        occ_bin = torch.zeros_like(occ_prob)
        if k_value > 0:
            topk_idx = occ_prob.topk(min(k_value, occ_prob.shape[-1]), dim=-1).indices
            occ_bin.scatter_(-1, topk_idx, 1.0)

        impedance_log, _, _, _ = engine._denorm_impedance(impedance_norm)

    hm_np = heatmap_zscore.cpu().numpy()
    occ_np = occ_bin.cpu().numpy().astype(np.int8)
    imp_np = impedance_log.cpu().numpy()

    np.save(out_dir / "occupancy.npy", occ_np)

    for i in range(num_samples):
        sample_dir = out_dir / f"data_sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        np.save(sample_dir / "heatmap_zscore.npy", hm_np[i])
        np.save(sample_dir / "occupancy_map.npy", occ_np[i])
        np.save(sample_dir / "impedance_profile.npy", imp_np[i])


def _run_report(
    *,
    gen_dir: Path,
    dataset_root: Path,
    k_value: int,
    hm_pool: int,
    baseline_n: int,
    max_train: int | None,
) -> tuple[int, str]:
    project_root = _project_root()
    cmd = [
        _python_exe(),
        str(project_root / "evaluation" / "novelty" / "scripts" / "vae_novelty_report.py"),
        "--gen-dir",
        str(gen_dir),
        "--dataset-root",
        str(dataset_root),
        "--K",
        str(k_value),
        "--hm-pool",
        str(hm_pool),
        "--baseline-n",
        str(baseline_n),
    ]
    if max_train is not None:
        cmd += ["--max-train", str(max_train)]

    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()


def _write_summary(*, out_dir: Path, params: dict, report_stdout: str) -> Path:
    summary_path = out_dir / "novelty_report_summary.md"
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(f"# VAE novelty test summary\n")
    lines.append(f"Generated at: {ts}\n")
    lines.append("## Parameters\n")
    for k, v in params.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Report output\n")
    lines.append("```\n" + report_stdout + "\n```\n")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--shared-temp", type=float, default=1.5)
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/data_norm"))
    ap.add_argument("--hm-pool", type=int, default=16)
    ap.add_argument("--baseline-n", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: evaluation/novelty/runs/K{K}_noveltyN{N})",
    )
    args = ap.parse_args()

    if not (0 <= args.K <= 52):
        raise SystemExit("--K must be in [0,52]")
    if args.N <= 0:
        raise SystemExit("--N must be > 0")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("evaluation/novelty/runs") / f"K{args.K}_noveltyN{args.N}"

    print(f"[1/3] Generating N={args.N} samples at K={args.K} → {out_dir}")
    _save_generated_samples(
        out_dir=out_dir,
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        k_value=args.K,
        num_samples=args.N,
        shared_temp=args.shared_temp,
        force_cpu=bool(args.force_cpu),
    )

    print("[2/3] Scoring generated samples vs dataset (nearest-neighbor)")
    rc, report_out = _run_report(
        gen_dir=out_dir,
        dataset_root=args.dataset_root,
        k_value=args.K,
        hm_pool=args.hm_pool,
        baseline_n=args.baseline_n,
        max_train=args.max_train,
    )
    if rc != 0:
        print(report_out)
        raise SystemExit(rc)

    print("[3/3] Writing summary markdown")
    params = {
        "checkpoint": args.checkpoint,
        "latent_dim": args.latent_dim,
        "K": args.K,
        "N": args.N,
        "shared_temp": args.shared_temp,
        "dataset_root": str(args.dataset_root),
        "hm_pool": args.hm_pool,
        "baseline_n": args.baseline_n,
        "max_train": args.max_train,
        "force_cpu": args.force_cpu,
        "out_dir": str(out_dir),
    }
    summary_path = _write_summary(out_dir=out_dir, params=params, report_stdout=report_out)

    print("\n✓ Done")
    print(f"  CSV:     {out_dir / 'novelty_report.csv'}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
