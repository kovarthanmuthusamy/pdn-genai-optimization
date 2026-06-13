from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from active_learning_pi.al.candidates import Candidate
from active_learning_pi.al.robust_stats import metric_from_stats, robust_peak_stats


def _load_engine(cfg: dict, groot: Path, device: torch.device) -> Any:
    exp = cfg["experiment_dir"].replace("\\", "/")
    mod_path = exp.replace("/", ".") + ".codes.inference_vae"
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))
    VAEInference = importlib.import_module(mod_path).VAEInference
    ckpt = groot / cfg["checkpoint_path"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    engine = VAEInference(checkpoint_path=str(ckpt), device=device)
    stats_auto = groot / exp / "metrics" / "latent_stats.json"
    engine.load_latent_stats(str(stats_auto) if stats_auto.is_file() else None)
    return engine


def _hm_physical(hm_z: torch.Tensor, hm_log_mean: float, hm_log_std: float) -> np.ndarray:
    return (torch.exp(hm_z * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0).cpu().numpy()


def predict_candidates(
    cfg: dict,
    candidates: list[Candidate],
    groot: Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """
    Run stochastic decode for each candidate; return scores + heatmap stats.
    Uncertainty = variance of robust p99 (or p95) across mc_passes latent samples.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = _load_engine(cfg, groot, device)
    mask = engine.binary_mask
    mc = int(cfg.get("mc_passes", 6))
    mode = cfg.get("inference_mode", "anchor_blend")
    pi_ref = float(cfg.get("pi_ref_mhz", 200.0))
    shared_temp = float(cfg.get("shared_temp", 1.5))
    mhz_priority = set(float(x) for x in cfg.get("mhz_priority", []))
    priority_boost = float(cfg.get("mhz_priority_boost", 0.15))

    from experiments.exp038_true_multi.codes.freq_inference_utils import (  # noqa: E402
        calibrate_heatmap_physical,
        interp_anchor_value,
        is_training_anchor,
        load_anchor_fg_max_table,
    )

    calibrate = bool(cfg.get("calibrate_fg_max", True))
    fg_table = load_anchor_fg_max_table() if calibrate else {}

    rows: list[dict[str, Any]] = []
    n_total = len(candidates)
    print(f"  Running inference on {n_total} candidate(s), {mc} MC pass(es) each...")

    for idx, cand in enumerate(candidates):
        print(f"    [{idx + 1}/{n_total}] candidate_id={cand.candidate_id}  {cand.mhz:.1f} MHz  K={cand.k}", flush=True)
        occ = torch.tensor([cand.occupancy], dtype=torch.float32, device=device)
        k = cand.k
        mhz = cand.mhz
        pass_stats: list[dict[str, float]] = []

        with torch.no_grad():
            for _ in range(mc):
                inf_kw: dict[str, Any] = dict(
                    latent_stats=engine.latent_stats,
                    per_K_latent_stats=engine.per_K_latent_stats,
                    shared_temp=shared_temp,
                    mode=mode,
                    pi_ref_mhz=pi_ref,
                )
                hm_z, _, imp_norm = engine.model.inference(
                    1,
                    device,
                    K=k,
                    PI_freq=mhz,
                    pi_freq_unit="mhz",
                    occupancy=occ,
                    **inf_kw,
                )
                hm_phys = _hm_physical(hm_z, engine.hm_log_mean, engine.hm_log_std)[0]
                if calibrate and fg_table and not is_training_anchor(mhz):
                    target_max = interp_anchor_value(mhz, fg_table)
                    hm_phys = calibrate_heatmap_physical(hm_phys, mask, target_max)
                pass_stats.append(robust_peak_stats(hm_phys, mask))

        metric_key = "p99"
        if cfg.get("score_metric", "p99_var") == "p95_var":
            metric_key = "p95"
        elif cfg.get("score_metric") == "topk_mean_var":
            metric_key = "topk_mean"

        uncertainty = metric_from_stats(pass_stats, metric_key)
        score = uncertainty
        if mhz in mhz_priority:
            score += priority_boost

        imp_np = imp_norm[0].detach().cpu().numpy().astype(np.float32)
        if imp_np.ndim == 3:
            imp_np = imp_np[0]
        imp_np = imp_np.reshape(-1)

        row = {
            **cand.to_dict(),
            "uncertainty": uncertainty,
            "acquisition_score": score,
            "badness": uncertainty,
            "quality_score": -uncertainty,
            "pred_p99_mean": float(np.mean([s["p99"] for s in pass_stats])),
            "pred_p99_std": float(np.std([s["p99"] for s in pass_stats])),
            "pred_p95_mean": float(np.mean([s["p95"] for s in pass_stats])),
            "pred_impedance_norm": imp_np.tolist(),
        }
        rows.append(row)
        print(
            f"      uncertainty={uncertainty:.6f}  pred_p99={row['pred_p99_mean']:.4f}",
            flush=True,
        )

    return rows
