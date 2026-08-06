---
title: heatmap_gmax_norm
type: code
path: src_vae/others/heatmap_gmax_norm.py
group: src_vae/others
loc: 191
tags: [code, src_vae]
---

# heatmap_gmax_norm

> Global-max heatmap normalization utilities.

**Source:** `src_vae/others/heatmap_gmax_norm.py` · 191 lines

## Purpose

```text
Global-max heatmap normalization utilities.

Run: Imported by training scripts and ``datasets/build_multifreq_gmax_dataset.py``.
```

## Constants

| Name | Value |
|------|-------|
| `LOG1P_TRAIN_SPACE` | `'log1p_gmax'` |
| `LINEAR_TRAIN_SPACE` | `'linear'` |

## Functions

- **`is_global_max_stats(stats: dict[str, Any])`**
- **`load_gmax_from_stats(stats: dict[str, Any])`** — Return (global_max_ohm, background_value).
- **`zscore_to_physical(z: np.ndarray | torch.Tensor, *, log_mean: float, log_std: float)`** — Invert log-z-score multifreq heatmaps to Ω.
- **`physical_to_gmax_norm(phys: np.ndarray, *, global_max_ohm: float, bg_ohm: float)`** — Map physical Ω to [0, 1] with background forced to 0.
- **`gmax_norm_to_physical(norm: torch.Tensor, global_max_ohm: float)`**
- **`log1p_gmax_denom(global_max_ohm: float)`**
- **`linear_norm_to_log1p_train(norm: np.ndarray | torch.Tensor, global_max_ohm: float)`** — Map on-disk linear norm (phys/gmax) → log1p train space in [0, 1].
- **`log1p_train_to_linear_norm(train: np.ndarray | torch.Tensor, global_max_ohm: float)`** — Inverse: log1p train space → on-disk linear norm.
- **`log1p_train_to_physical(train: torch.Tensor, global_max_ohm: float)`** — log1p train space → Ω (skips linear norm intermediate).
- **`linear_threshold_to_train(thr_linear: float, global_max_ohm: float)`**
- **`linear_clip_bounds_to_train(clip_lo: float, clip_hi: float, global_max_ohm: float)`**
- **`is_log1p_train_space(c)`**
- **`disk_to_train_space(hm: torch.Tensor, c)`** — On-disk linear gmax norm → model train space.
- **`train_to_disk_space(hm: torch.Tensor, c)`** — Model train space → on-disk linear gmax norm.
- **`heatmap_model_to_physical(hm: torch.Tensor, c)`** — Model output in train space → physical Ω.
- **`heatmap_fg_threshold(c)`** — Norm-space FG threshold (matches ``background_value + 0.5`` convention).
- **`heatmap_phys_amplitude_loss_gmax(recon: torch.Tensor, target: torch.Tensor, global_max_ohm: float, c, *, downsample_2x=None, percentile_fn=None)`** — FG p99 in Ω — for global-max normalized heatmaps.
- **`load_stats(data_dir: str | Path)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]

## Imported by

- [[_audit_decap_locality]]
- [[_audit_physics]]
- [[_bench_physics_overhead]]
- [[_bench_train_step]]
- [[_breakdown_hm_loss]]
- [[_breakdown_hm_real]]
- [[check_log1p_train_space]]
- [[diagnose_sweep_vs_dataset]]
- [[eval_cross_freq_gmax]]
- [[experiments.exp043.codes.inference_vae]]
- [[gmax_heatmap_loss]]
- [[gmax_training_patch]]
- [[multifreq]]
- [[verify_peb_gmax_match]]

## External dependencies

`experiments`, `numpy`, `torch`
