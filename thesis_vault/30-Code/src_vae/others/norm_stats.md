---
title: norm_stats
type: code
path: src_vae/others/norm_stats.py
group: src_vae/others
loc: 413
tags: [code, src_vae]
---

# norm_stats

> Centralized normalization / denormalization from ``normalization_stats.json``.

**Source:** `src_vae/others/norm_stats.py` · 413 lines

## Purpose

```text
Centralized normalization / denormalization from ``normalization_stats.json``.

Supports:
  - ``log_zscore`` / ``log_zscore_unbounded`` (legacy mean/std on log1p)
  - ``robust_log1p_per_mhz`` / ``robust_log1p_per_mhz_unbounded`` (median/IQR per anchor)

All heatmap and impedance scale/denorm should go through this module.
```

## Classes

- **`HeatmapBinStats`**
- **`HeatmapNormStats`**
- **`ImpedanceNormStats`**
- **`NormStatsBundle`**

## Functions

- **`_mhz_key(mhz: float)`**
- **`pi_norm_to_mhz(pi_norm: float | np.ndarray | torch.Tensor)`**
- **`load_norm_stats(data_dir: str | Path, *, stats_path: str | Path | None=None)`**
- **`load_heatmap_stats(data_dir: str | Path, *, stats_path: str | Path | None=None)`**
- **`load_heatmap_clip_bounds(data_dir: str | Path, *, stats_path: str | Path | None=None)`** — Global clip bounds for training. Returns None when stats are unbounded.

## Imports

- [[multifreq_anchors]]
- [[pi_freq_utils]]

## Imported by

- [[dataloader]]
- [[debug_clip_spatial]]
- [[debug_inspect_robust_stats]]
- [[experiments.exp048.codes.eval_real_data_sweep]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp048.codes.run_epoch_encode]]
- [[experiments.exp048.codes.train_vae_simple]]
- [[experiments.exp049.codes.eval_real_data_sweep]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp049.codes.run_epoch_encode]]
- [[experiments.exp049.codes.train_vae_simple]]
- [[experiments.exp050.codes.eval_real_data_sweep]]
- [[experiments.exp050.codes.heatmap_peak_losses]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp050.codes.train_vae_simple]]
- [[experiments.exp051_new_datas_appended.codes.eval_real_data_sweep]]
- [[experiments.exp051_new_datas_appended.codes.heatmap_peak_losses]]
- [[experiments.exp051_new_datas_appended.codes.inference_vae]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.eval_real_data_sweep]]
- [[experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_real_data_sweep]]
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]
- [[experiments.exp054_K_30.codes.heatmap_peak_losses]]
- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]
- [[experiments.exp055_hard_occ.codes.heatmap_peak_losses]]
- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
- [[experiments.exp056_graph_vae.codes.heatmap_peak_losses]]
- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
- [[experiments.exp057_structured_graph.codes.heatmap_peak_losses]]
- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
- [[experiments.exp058_asymmetric_kl.codes.heatmap_peak_losses]]
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
- [[experiments.exp059_capacity_freq.codes.heatmap_peak_losses]]
- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]
- [[experiments.exp060_multitype_occ.codes.heatmap_peak_losses]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
- [[gp_error_surrogate]]
- [[gp_error_surrogate_test]]
- [[heatmap_z_clip]]

## External dependencies

`numpy`, `torch`
