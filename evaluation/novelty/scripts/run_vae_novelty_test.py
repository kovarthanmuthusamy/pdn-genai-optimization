#!/usr/bin/env python3
"""End-to-end novelty test: generate N samples + score vs dataset.

This script:
1) Uses the exp027 inference code to generate a small batch of samples for a given K.
2) Saves them in the same on-disk layout as `scrap/generate_samples_and_peb.py`.
3) Runs `scripts/vae_novelty_report.py` to compute nearest-neighbor scores and write a CSV.
4) Writes a short markdown summary next to the CSV.

Example
  python scripts/run_vae_novelty_test.py

Notes
- This does NOT require ECADStar / .peb.
- For scoring, we strongly recommend filtering the dataset to the same K.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch



# =============================================================================
# CONFIGURATION — edit these before running: python evaluation/novelty/scripts/run_vae_novelty_test.py
# =============================================================================

CHECKPOINT = "experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt"
LATENT_DIM = 32
K_VALUE = 5
NUM_SAMPLES = 50
SHARED_TEMP = 1.5
DATASET_ROOT = Path("datasets/data_norm")
HM_POOL = 16
BASELINE_N = 200
MAX_TRAIN: int | None = None
FORCE_CPU = False
OUT_DIR: Path | None = None  # None → evaluation/novelty/runs/K{K}_noveltyN{N}

# =============================================================================

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
    import io
    from contextlib import redirect_stdout, redirect_stderr
    import evaluation.novelty.scripts.vae_novelty_report as report_mod
    report_mod.GEN_DIR = gen_dir
    report_mod.DATASET_ROOT = dataset_root
    report_mod.K_FILTER = k_value
    report_mod.HM_POOL = hm_pool
    report_mod.BASELINE_N = baseline_n
    report_mod.MAX_TRAIN = max_train
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            report_mod.main()
            return 0, buf.getvalue().strip()
        except SystemExit as e:
            return int(e.code or 1), buf.getvalue().strip()


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
    if not (0 <= K_VALUE <= 52):
        raise SystemExit("K_VALUE must be in [0,52]")
    if NUM_SAMPLES <= 0:
        raise SystemExit("NUM_SAMPLES must be > 0")

    out_dir = OUT_DIR or (Path("evaluation/novelty/runs") / f"K{K_VALUE}_noveltyN{NUM_SAMPLES}")

    print(f"[1/3] Generating N={NUM_SAMPLES} samples at K={K_VALUE} → {out_dir}")
    _save_generated_samples(
        out_dir=out_dir,
        checkpoint_path=CHECKPOINT,
        latent_dim=LATENT_DIM,
        k_value=K_VALUE,
        num_samples=NUM_SAMPLES,
        shared_temp=SHARED_TEMP,
        force_cpu=bool(FORCE_CPU),
    )

    print("[2/3] Scoring generated samples vs dataset (nearest-neighbor)")
    rc, report_out = _run_report(
        gen_dir=out_dir,
        dataset_root=DATASET_ROOT,
        k_value=K_VALUE,
        hm_pool=HM_POOL,
        baseline_n=BASELINE_N,
        max_train=MAX_TRAIN,
    )
    if rc != 0:
        print(report_out)
        raise SystemExit(rc)

    print("[3/3] Writing summary markdown")
    params = {
        "checkpoint": CHECKPOINT,
        "latent_dim": LATENT_DIM,
        "K": K_VALUE,
        "N": NUM_SAMPLES,
        "shared_temp": SHARED_TEMP,
        "dataset_root": str(DATASET_ROOT),
        "hm_pool": HM_POOL,
        "baseline_n": BASELINE_N,
        "max_train": MAX_TRAIN,
        "force_cpu": FORCE_CPU,
        "out_dir": str(out_dir),
    }
    summary_path = _write_summary(out_dir=out_dir, params=params, report_stdout=report_out)

    print("\n✓ Done")
    print(f"  CSV:     {out_dir / 'novelty_report.csv'}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
