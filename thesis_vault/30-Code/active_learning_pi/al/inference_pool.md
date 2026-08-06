---
title: inference_pool
type: code
path: active_learning_pi/al/inference_pool.py
group: active_learning_pi/al
loc: 596
tags: [code, active_learning_pi]
---

# inference_pool

> VAE Monte-Carlo inference over the candidate pool (uncertainty scoring).

**Source:** `active_learning_pi/al/inference_pool.py` · 596 lines

## Purpose

```text
VAE Monte-Carlo inference over the candidate pool (uncertainty scoring).

Run:
    python active_learning_pi/al/inference_pool.py
```

## Functions

- **`_load_engine(cfg: dict, groot: Path, device: torch.device)`**
- **`_hm_physical(hm_z: torch.Tensor, hm_log_mean: float, hm_log_std: float)`** — Legacy global log-z denorm. Prefer :func:`_hm_physical_engine` for per-MHz stats.
- **`_hm_physical_engine(engine: Any, hm_z: torch.Tensor, *, mhz: float)`** — Denorm model heatmap to physical Ω using the engine's norm stats (per-MHz aware).
- **`_mhz_in_list(mhz: float, items: list[float] | None, tol: float=0.5)`**
- **`_should_calibrate(mhz: float, cfg: dict, *, is_training_anchor_fn)`**
- **`_apply_calibration(hm_phys: np.ndarray, mask: np.ndarray, mhz: float, fg_table: dict[float, float], cfg: dict, *, calibrate_fn, interp_fn)`**
- **`_score_weights(cfg: dict)`**
- **`_resolve_uncertainty(score_metric: str, *, decoder_rce: float, latent_mse: float, peak_bias: float, p99_var: float, spatial_spread: float, weights: dict[str, float], auto_badness: float | None=None)`**
- **`predict_candidates(cfg: dict, candidates: list[Candidate], groot: Path, device: torch.device | None=None)`** — Score candidates for ECAD selection.
- **`_predict_candidates_random(cfg: dict, candidates: list[Candidate], groot: Path, device: torch.device | None=None)`** — Equal-budget random baseline: uniform scores (selection shuffles via badness).
- **`_predict_candidates_gp_error(cfg: dict, candidates: list[Candidate], groot: Path, device: torch.device | None=None)`**
- **`_predict_candidates_mc(cfg: dict, candidates: list[Candidate], groot: Path, device: torch.device | None=None)`** — Run occ-only layout inference with MC decoder dropout + badness scoring.

## Imports

- [[active_learning_pi.al.paths]]
- [[candidates]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[gp_error_surrogate]]
- [[pi_freq_utils]]
- [[robust_stats]]

## Imported by

- [[evaluate_cycle]]
- [[evaluate_hole_finding]]
- [[per_k_acquire]]
- [[pipeline]]
- [[validate_acquisition_ab]]

## External dependencies

`experiments`, `importlib`, `numpy`, `src_vae`, `torch`
