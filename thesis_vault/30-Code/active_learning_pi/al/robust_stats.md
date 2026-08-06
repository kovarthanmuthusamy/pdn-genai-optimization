---
title: robust_stats
type: code
path: active_learning_pi/al/robust_stats.py
group: active_learning_pi/al
loc: 236
tags: [code, active_learning_pi]
---

# robust_stats

> Robust foreground peak statistics on physical heatmaps.

**Source:** `active_learning_pi/al/robust_stats.py` · 236 lines

## Purpose

```text
Robust foreground peak statistics on physical heatmaps.

Run:
    Import only — used by ``inference_pool`` and ``evaluate_off_anchor``.
```

## Functions

- **`_peak_location_2d(arr: np.ndarray, mask: np.ndarray | None=None)`** — Argmax (row, col) within mask; ties → first occurrence.
- **`foreground_mask(hm: np.ndarray, bg_margin: float=0.5)`** — hm: (2,H,W) z-score or (H,W) single channel — use channel 0.
- **`robust_peak_stats(hm_phys: np.ndarray, mask_board: np.ndarray | None=None)`** — hm_phys: (2,H,W) physical impedance × mask or (H,W).
- **`metric_from_stats(stats_list: list[dict[str, float]], key: str='p99')`**
- **`peak_loc_spread(stats_list: list[dict[str, float]])`** — RMS spread (px) of argmax peak location across MC passes.
- **`combined_uncertainty(stats_list: list[dict[str, float]], *, magnitude_key: str='p99', spatial_weight: float=1.0)`** — Return (total, magnitude_var, peak_loc_spread_px).
- **`_heatmap_board_flat(hm_phys: np.ndarray, mask_board: np.ndarray | None)`** — Flatten all heatmap channels over the board mask for whole-map comparison.
- **`heatmap_mc_sample_mse(heatmaps: list[np.ndarray], mask_board: np.ndarray | None=None)`** — Mean pairwise MSE across MC heatmap samples on the board region (all channels).
- **`heatmap_mc_rce(heatmaps: list[np.ndarray], mask_board: np.ndarray | None=None)`** — Reconstruction consistency error: mean MSE of each sample vs the MC mean.
- **`mc_p99_range(stats_list: list[dict[str, float]])`** — Peak p99 spread across MC decoder passes (MHz-agnostic disagreement).
- **`relative_prior_deficit(pred_p99: float, mhz: float, fg_max_table: dict[float, float], *, interp_fn=None)`** — Unitless shortfall vs training-set fg-max prior at ``mhz`` (0 = at/above prior).
- **`peak_amplitude_bias_score(pred_p99: float, mhz: float, fg_max_table: dict[float, float], *, mhz_weights: dict[float, float] | None=None, interp_fn=None)`** — Legacy squared deficit — prefer ``auto_acquire_badness`` for new runs.
- **`auto_acquire_badness(*, decoder_rce: float, latent_mse: float, p99_var: float, spatial_spread: float, p99_range: float, pred_p99_mean: float, pred_p99_std: float, prior_deficit: float, weights: dict[str, float])`** — MHz-agnostic acquisition badness from model self-disagreement only.

## Imports

- [[experiments.exp038_true_multi.codes.freq_inference_utils]]

## Imported by

- [[evaluate_off_anchor]]
- [[gp_error_surrogate]]
- [[inference_pool]]

## External dependencies

`experiments`, `numpy`
