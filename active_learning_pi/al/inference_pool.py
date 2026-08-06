"""VAE Monte-Carlo inference over the candidate pool (uncertainty scoring).

Run:
    python active_learning_pi/al/inference_pool.py"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from active_learning_pi.al.candidates import Candidate
from active_learning_pi.al.robust_stats import (
    auto_acquire_badness,
    heatmap_mc_rce,
    heatmap_mc_sample_mse,
    mc_p99_range,
    metric_from_stats,
    peak_amplitude_bias_score,
    peak_loc_spread,
    relative_prior_deficit,
    robust_peak_stats,
)


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
    """Legacy global log-z denorm. Prefer :func:`_hm_physical_engine` for per-MHz stats."""
    return (torch.exp(hm_z * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0).cpu().numpy()


def _hm_physical_engine(engine: Any, hm_z: torch.Tensor, *, mhz: float) -> np.ndarray:
    """Denorm model heatmap to physical Ω using the engine's norm stats (per-MHz aware)."""
    denorm = getattr(engine, "denorm_heatmap_physical", None)
    if callable(denorm):
        out = denorm(hm_z, mhz=float(mhz))
        if isinstance(out, torch.Tensor):
            arr = out.detach().cpu().numpy()
        else:
            arr = np.asarray(out)
        # engine returns (B,C,H,W) or (B,H,W); callers expect a single layout (C,H,W) or (H,W)
        if arr.ndim == 4:
            return arr[0]
        if arr.ndim == 3 and arr.shape[0] <= 4:
            return arr[0] if arr.shape[0] == 1 else arr
        return arr
    return _hm_physical(hm_z, float(engine.hm_log_mean), float(engine.hm_log_std))[0]


def _mhz_in_list(mhz: float, items: list[float] | None, tol: float = 0.5) -> bool:
    if not items:
        return False
    return any(abs(float(mhz) - float(x)) <= tol for x in items)


def _should_calibrate(
    mhz: float,
    cfg: dict,
    *,
    is_training_anchor_fn,
) -> bool:
    if not bool(cfg.get("calibrate_fg_max", True)):
        return False
    exclude = [float(x) for x in cfg.get("calibrate_fg_max_exclude_mhz", [])]
    if _mhz_in_list(mhz, exclude):
        return False
    mode = str(cfg.get("calibrate_fg_max_mode", "off_anchor")).lower()
    if mode in ("none", "false", "off"):
        return False
    if mode in ("all", "always"):
        return True
    # default: off-anchor only
    return not is_training_anchor_fn(mhz)


def _apply_calibration(
    hm_phys: np.ndarray,
    mask: np.ndarray,
    mhz: float,
    fg_table: dict[float, float],
    cfg: dict,
    *,
    calibrate_fn,
    interp_fn,
) -> np.ndarray:
    target_max = interp_fn(mhz, fg_table)
    scale_up_only = bool(cfg.get("calibrate_fg_max_scale_up_only", True))
    return calibrate_fn(
        hm_phys,
        mask,
        target_max,
        scale_up_only=scale_up_only,
    )


def _score_weights(cfg: dict) -> dict[str, float]:
    raw = cfg.get("score_weights") or {}
    defaults = {
        "decoder_dropout_rce": 1.0,
        "latent_heatmap_mse": 1.0,
        "peak_bias": 1.0,
        "p99_var": 1.0,
        "peak_loc_spread": 0.5,
        "p99_range": 0.5,
        "peak_cv": 0.25,
        "prior_tension": 0.35,
    }
    out = dict(defaults)
    for k, v in raw.items():
        out[k] = float(v)
    return out


def _resolve_uncertainty(
    score_metric: str,
    *,
    decoder_rce: float,
    latent_mse: float,
    peak_bias: float,
    p99_var: float,
    spatial_spread: float,
    weights: dict[str, float],
    auto_badness: float | None = None,
) -> float:
    if score_metric in ("auto_bad", "auto", "automatic"):
        return float(auto_badness if auto_badness is not None else 0.0)
    if score_metric in ("combined_al", "al_combined", "combined"):
        return (
            weights["decoder_dropout_rce"] * decoder_rce
            + weights["latent_heatmap_mse"] * latent_mse
            + weights["peak_bias"] * peak_bias
            + weights["p99_var"] * p99_var
            + weights["peak_loc_spread"] * spatial_spread
        )
    if score_metric in ("decoder_mc_dropout_rce", "decoder_rce", "mc_dropout_rce"):
        return decoder_rce
    if score_metric in ("heatmap_mse", "hm_mse", "mc_heatmap_mse"):
        return latent_mse
    if score_metric in ("peak_bias", "peak_amplitude_bias"):
        return peak_bias
    if score_metric == "peak_loc_spread":
        return spatial_spread
    if score_metric == "p99_var":
        return p99_var
    return decoder_rce


def predict_candidates(
    cfg: dict,
    candidates: list[Candidate],
    groot: Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """
    Score candidates for ECAD selection.

    Default ``acquisition_mode=mc`` (or unset): occ-only layout inference with MC
    decoder dropout + ``auto_bad`` disagreement scores.

    ``acquisition_mode=gp_error``: fit a pred+SVGP error surrogate on labeled data,
    then rank candidates by UCB ``mu_e + kappa * sigma_e`` (see ``gp_error_surrogate``).
    """
    mode = str(cfg.get("acquisition_mode", "mc")).strip().lower()
    if mode in ("gp_error", "gp", "error_gp", "svgp"):
        return _predict_candidates_gp_error(cfg, candidates, groot, device=device)
    if mode in ("random", "rand", "baseline_random"):
        return _predict_candidates_random(cfg, candidates, groot, device=device)
    return _predict_candidates_mc(cfg, candidates, groot, device=device)


def _predict_candidates_random(
    cfg: dict,
    candidates: list[Candidate],
    groot: Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Equal-budget random baseline: uniform scores (selection shuffles via badness)."""
    seed = int((cfg.get("gp_error") or {}).get("seed", cfg.get("candidate_seed", 0)))
    rng = np.random.default_rng(seed + int(cfg.get("_iteration", 0)))
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        score = float(rng.random())
        rows.append({
            **cand.to_dict(),
            "uncertainty": score,
            "acquisition_score": score,
            "badness": score,
            "quality_score": -score,
            "acquisition_mode": "random",
            "heatmap_mc_mse": 0.0,
            "decoder_mc_rce": 0.0,
            "peak_bias_score": 0.0,
            "epistemic_score": score,
            "disagreement_score": 0.0,
            "prior_deficit": 0.0,
            "prior_tension_score": 0.0,
            "p99_mc_range": 0.0,
            "p99_var": 0.0,
            "peak_loc_spread_px": 0.0,
            "pred_peak_row_mean": 0.0,
            "pred_peak_col_mean": 0.0,
            "pred_p99_mean": 0.0,
            "pred_p99_std": 0.0,
            "pred_p95_mean": 0.0,
            "pred_impedance_norm": [],
        })
    print(f"  Random acquisition scores for {len(rows)} candidate(s) (seed={seed})", flush=True)
    return rows


def _predict_candidates_gp_error(
    cfg: dict,
    candidates: list[Candidate],
    groot: Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    from active_learning_pi.al.gp_error_surrogate import (
        build_fit_loader,
        fit_error_gp,
        pred_bootstrap_features,
        resolve_gp_error_cfg,
        save_artifact_meta,
    )
    from active_learning_pi.al.paths import iteration_dir
    from src_vae.others.pi_freq_utils import pi_freq_norm_for_model

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_cfg = resolve_gp_error_cfg(cfg)

    artifact = cfg.get("_gp_error_artifact")
    engine = cfg.get("_gp_error_engine")
    if engine is None:
        engine = _load_engine(cfg, groot, device)
        cfg["_gp_error_engine"] = engine
    model = engine.model
    model.eval()

    if artifact is None:
        exp = cfg["experiment_dir"].replace("\\", "/")
        import importlib
        import sys

        if str(groot) not in sys.path:
            sys.path.insert(0, str(groot))
        inf = importlib.import_module(exp.replace("/", ".") + ".codes.inference_vae")
        exp_yaml = inf.load_experiment_config(Path(exp) / "config.yaml")
        cfg["_gp_error_exp_yaml"] = exp_yaml

        fit_loader = build_fit_loader(
            cfg["experiment_dir"],
            cfg.get("data_dir"),
            groot,
            cfg_yaml=exp_yaml,
        )
        artifact = fit_error_gp(
            engine,
            fit_loader,
            path=gp_cfg["path"],  # type: ignore[arg-type]
            target=gp_cfg["target"],  # type: ignore[arg-type]
            gp=gp_cfg["gp"],
            n_fit=gp_cfg["n_fit"],
            n_inducing=gp_cfg["n_inducing"],
            svgp_epochs=gp_cfg["svgp_epochs"],
            kappa=gp_cfg["kappa"],
            novelty_weight=gp_cfg["novelty_weight"],
            novelty_k=gp_cfg["novelty_k"],
            score_mode=gp_cfg["score_mode"],
            joint_alpha_hm=gp_cfg["joint_alpha_hm"],
            joint_alpha_imp=gp_cfg["joint_alpha_imp"],
            device=device,
            cfg=exp_yaml,
            verbose=gp_cfg["verbose"],
            seed=gp_cfg["seed"],
        )
        cfg["_gp_error_artifact"] = artifact

        try:
            it = int(cfg.get("_iteration", cfg.get("iteration", -1)))
            if it >= 0 and cfg.get("run_name"):
                meta_path = iteration_dir(cfg, it, groot) / "gp_error_surrogate_meta.json"
                save_artifact_meta(artifact, meta_path)
                print(f"  Wrote GP meta → {meta_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip GP meta write: {exc})", flush=True)
    else:
        print(
            f"  Reusing fitted error-GP (n_fit={artifact.n_fit}, path={artifact.path})",
            flush=True,
        )

    n_total = len(candidates)
    print(
        f"  GP-error scoring {n_total} candidate(s): "
        f"path={artifact.path} target={artifact.target} kappa={artifact.kappa} "
        f"novelty_w={artifact.novelty_weight} gp={artifact.gp_kind} n_fit={artifact.n_fit}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    z_rows: list[np.ndarray] = []
    hm_phys_cache: list[np.ndarray] = []
    imp_cache: list[np.ndarray] = []

    for idx, cand in enumerate(candidates):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(
                f"    [{idx + 1}/{n_total}] candidate_id={cand.candidate_id}  "
                f"{cand.mhz:.1f} MHz  K={cand.k}",
                flush=True,
            )
        occ = torch.tensor([cand.occupancy], dtype=torch.float32, device=device)
        K_tensor = torch.full((1,), int(cand.k), dtype=torch.long, device=device)
        pi_t = pi_freq_norm_for_model(cand.mhz, 1, unit="mhz", device=device)

        z_pred, hm_z, imp_norm = pred_bootstrap_features(
            model,
            occupancy=occ,
            K=K_tensor,
            pi=pi_t,
            path=artifact.path,  # type: ignore[arg-type]
        )
        z_rows.append(z_pred[0].detach().cpu().numpy())
        hm_phys = _hm_physical_engine(engine, hm_z, mhz=float(cand.mhz))
        hm_phys_cache.append(hm_phys)
        imp_np = (
            imp_norm[0].detach().cpu().numpy().astype(np.float32)
            if imp_norm is not None
            else np.zeros(231, np.float32)
        )
        if imp_np.ndim == 3:
            imp_np = imp_np[0]
        imp_cache.append(imp_np.reshape(-1))

    Z = np.stack(z_rows, axis=0)
    mu, sigma, ucb, acq = artifact.score_z(Z)
    # novelty component for logging
    novelty = acq - ucb

    mask = engine.binary_mask
    for i, cand in enumerate(candidates):
        score = float(acq[i])
        stats = robust_peak_stats(hm_phys_cache[i], mask)
        row = {
            **cand.to_dict(),
            "uncertainty": score,
            "acquisition_score": score,
            "badness": score,
            "quality_score": -score,
            "acquisition_mode": "gp_error",
            "gp_mu": float(mu[i]),
            "gp_sigma": float(sigma[i]),
            "gp_ucb": float(ucb[i]),
            "gp_novelty": float(novelty[i]),
            "gp_path": artifact.path,
            "gp_target": artifact.target,
            "gp_kappa": artifact.kappa,
            "gp_novelty_weight": artifact.novelty_weight,
            "heatmap_mc_mse": 0.0,
            "decoder_mc_rce": 0.0,
            "peak_bias_score": 0.0,
            "epistemic_score": float(mu[i]),
            "disagreement_score": float(sigma[i]),
            "prior_deficit": 0.0,
            "prior_tension_score": float(novelty[i]),
            "p99_mc_range": 0.0,
            "p99_var": 0.0,
            "peak_loc_spread_px": 0.0,
            "pred_peak_row_mean": float(stats["peak_row"]),
            "pred_peak_col_mean": float(stats["peak_col"]),
            "pred_p99_mean": float(stats["p99"]),
            "pred_p99_std": 0.0,
            "pred_p95_mean": float(stats["p95"]),
            "pred_impedance_norm": imp_cache[i].tolist(),
        }
        rows.append(row)

    print(
        f"  GP scores: acq mean={float(np.mean(acq)):.4f} "
        f"max={float(np.max(acq)):.4f} mu_mean={float(np.mean(mu)):.4f} "
        f"novelty_mean={float(np.mean(novelty)):.4f}",
        flush=True,
    )
    return rows


def _predict_candidates_mc(
    cfg: dict,
    candidates: list[Candidate],
    groot: Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """
    Run occ-only layout inference with MC decoder dropout + badness scoring.

    Default ``score_metric=auto_bad`` ranks by MHz-agnostic model self-disagreement
    (decoder RCE, latent MSE, peak variance/spread). Training-set prior shortfall
    only amplifies already-uncertain layouts — no per-MHz weights or allowlists.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = _load_engine(cfg, groot, device)
    model = engine.model
    mask = engine.binary_mask
    mc_latent = int(cfg.get("mc_passes", cfg.get("mc_latent_passes", 4)))
    mc_decoder = int(cfg.get("mc_decoder_passes", cfg.get("mc_passes", 6)))
    use_decoder_dropout = bool(cfg.get("mc_decoder_dropout", True))
    mode = cfg.get("inference_mode", "layout")
    pi_ref = float(cfg.get("pi_ref_mhz", 200.0))
    shared_temp = float(cfg.get("shared_temp", 1.5))
    score_metric = str(cfg.get("score_metric", "auto_bad"))
    weights = _score_weights(cfg)
    use_legacy_peak_bias = score_metric in ("combined_al", "al_combined", "combined", "peak_bias", "peak_amplitude_bias")
    peak_bias_mhz = {float(x) for x in cfg.get("peak_bias_mhz", [])}
    peak_bias_weights = {
        float(k): float(v)
        for k, v in (cfg.get("peak_bias_mhz_weights") or {}).items()
    }

    from experiments.exp038_true_multi.codes.freq_inference_utils import (  # noqa: E402
        calibrate_heatmap_physical,
        interp_anchor_value,
        is_training_anchor,
        load_anchor_fg_max_table,
    )

    calibrate_enabled = bool(cfg.get("calibrate_fg_max", True))
    fg_table = load_anchor_fg_max_table() if calibrate_enabled else {}

    rows: list[dict[str, Any]] = []
    n_total = len(candidates)
    print(
        f"  Running inference on {n_total} candidate(s): "
        f"latent_mc={mc_latent} decoder_mc={mc_decoder} dropout={use_decoder_dropout} "
        f"metric={score_metric}",
    )

    from src_vae.others.pi_freq_utils import pi_freq_norm_for_model

    for idx, cand in enumerate(candidates):
        print(f"    [{idx + 1}/{n_total}] candidate_id={cand.candidate_id}  {cand.mhz:.1f} MHz  K={cand.k}", flush=True)
        occ = torch.tensor([cand.occupancy], dtype=torch.float32, device=device)
        k = cand.k
        mhz = cand.mhz
        K_tensor = torch.full((1,), int(k), dtype=torch.long, device=device)
        pi_t = pi_freq_norm_for_model(mhz, 1, unit="mhz", device=device)
        pi_ref_t = pi_freq_norm_for_model(pi_ref, 1, unit="mhz", device=device)

        latent_stats: list[dict[str, float]] = []
        latent_hm_passes: list[np.ndarray] = []
        decoder_hm_passes: list[np.ndarray] = []

        do_cal = calibrate_enabled and fg_table and _should_calibrate(
            mhz, cfg, is_training_anchor_fn=is_training_anchor,
        )

        imp_norm = None
        with torch.no_grad():
            model.eval()
            # Latent MC (occ-only encode + deterministic decode per sample)
            for _ in range(mc_latent):
                z = model.encode_occupancy_latent(occ, K_tensor, pi_ref_t, sample=True)
                hm_z, _, imp_norm = model.decode(z, K_tensor, pi_t, occupancy=occ)
                hm_phys = _hm_physical_engine(engine, hm_z, mhz=float(mhz))
                if do_cal:
                    hm_phys = _apply_calibration(
                        hm_phys, mask, mhz, fg_table, cfg,
                        calibrate_fn=calibrate_heatmap_physical,
                        interp_fn=interp_anchor_value,
                    )
                latent_hm_passes.append(hm_phys)
                latent_stats.append(robust_peak_stats(hm_phys, mask))

            # Decoder MC dropout at fixed latent mu (AL-aligned epistemic signal)
            z_mu = model.encode_occupancy_latent(occ, K_tensor, pi_ref_t, sample=False)
            if use_decoder_dropout and hasattr(model, "decode_heatmap_mc_dropout"):
                hm_z_list = model.decode_heatmap_mc_dropout(
                    z_mu, K_tensor, pi_t, occ, mc_decoder,
                )
                for hm_z in hm_z_list:
                    hm_phys = _hm_physical_engine(engine, hm_z, mhz=float(mhz))
                    if do_cal:
                        hm_phys = _apply_calibration(
                            hm_phys, mask, mhz, fg_table, cfg,
                            calibrate_fn=calibrate_heatmap_physical,
                            interp_fn=interp_anchor_value,
                        )
                    decoder_hm_passes.append(hm_phys)
            else:
                decoder_hm_passes = list(latent_hm_passes)

        decoder_stats = [robust_peak_stats(h, mask) for h in decoder_hm_passes]
        latent_mse = heatmap_mc_sample_mse(latent_hm_passes, mask)
        decoder_rce = heatmap_mc_rce(decoder_hm_passes, mask)
        p99_var = metric_from_stats(decoder_stats, "p99")
        spatial_spread = peak_loc_spread(decoder_stats)
        p99_range = mc_p99_range(decoder_stats)

        pred_p99 = float(np.mean([s["p99"] for s in decoder_stats]))
        pred_p99_std = float(np.std([s["p99"] for s in decoder_stats]))

        prior_deficit = 0.0
        if fg_table:
            prior_deficit = relative_prior_deficit(
                pred_p99, mhz, fg_table, interp_fn=interp_anchor_value,
            )

        auto_bad, auto_parts = auto_acquire_badness(
            decoder_rce=decoder_rce,
            latent_mse=latent_mse,
            p99_var=p99_var,
            spatial_spread=spatial_spread,
            p99_range=p99_range,
            pred_p99_mean=pred_p99,
            pred_p99_std=pred_p99_std,
            prior_deficit=prior_deficit,
            weights=weights,
        )

        peak_bias = 0.0
        if use_legacy_peak_bias and (float(mhz) in peak_bias_mhz or not peak_bias_mhz):
            peak_bias = peak_amplitude_bias_score(
                pred_p99,
                mhz,
                fg_table,
                mhz_weights=peak_bias_weights or None,
                interp_fn=interp_anchor_value,
            )

        uncertainty = _resolve_uncertainty(
            score_metric,
            decoder_rce=decoder_rce,
            latent_mse=latent_mse,
            peak_bias=peak_bias,
            p99_var=p99_var,
            spatial_spread=spatial_spread,
            weights=weights,
            auto_badness=auto_bad,
        )

        score = uncertainty

        imp_np = imp_norm[0].detach().cpu().numpy().astype(np.float32) if imp_norm is not None else np.zeros(231, np.float32)
        if imp_np.ndim == 3:
            imp_np = imp_np[0]
        imp_np = imp_np.reshape(-1)

        row = {
            **cand.to_dict(),
            "uncertainty": uncertainty,
            "acquisition_score": score,
            "badness": score,
            "quality_score": -score,
            "acquisition_mode": "mc",
            "heatmap_mc_mse": latent_mse,
            "decoder_mc_rce": decoder_rce,
            "peak_bias_score": peak_bias,
            "epistemic_score": auto_parts["epistemic_score"],
            "disagreement_score": auto_parts["disagreement_score"],
            "prior_deficit": auto_parts["prior_deficit"],
            "prior_tension_score": auto_parts["prior_tension_score"],
            "p99_mc_range": p99_range,
            "p99_var": p99_var,
            "peak_loc_spread_px": spatial_spread,
            "pred_peak_row_mean": float(np.mean([s["peak_row"] for s in decoder_stats])),
            "pred_peak_col_mean": float(np.mean([s["peak_col"] for s in decoder_stats])),
            "pred_p99_mean": pred_p99,
            "pred_p99_std": float(np.std([s["p99"] for s in decoder_stats])),
            "pred_p95_mean": float(np.mean([s["p95"] for s in decoder_stats])),
            "pred_impedance_norm": imp_np.tolist(),
        }
        rows.append(row)
        print(
            f"      score={score:.4f}  epistemic={auto_parts['epistemic_score']:.4f}  "
            f"disagree={auto_parts['disagreement_score']:.4f}  prior_tension={auto_parts['prior_tension_score']:.4f}  "
            f"pred_p99={pred_p99:.4f}",
            flush=True,
        )

    return rows
