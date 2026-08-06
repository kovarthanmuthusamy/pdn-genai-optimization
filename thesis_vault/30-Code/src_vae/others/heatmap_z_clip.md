---
title: heatmap_z_clip
type: code
path: src_vae/others/heatmap_z_clip.py
group: src_vae/others
loc: 119
tags: [code, src_vae]
---

# heatmap_z_clip

> Log-z heatmap clipping and denormalization helpers.

**Source:** `src_vae/others/heatmap_z_clip.py` · 119 lines

## Purpose

```text
Log-z heatmap clipping and denormalization helpers.

Delegates scale/denorm to ``norm_stats`` (single source of truth from normalization_stats.json).
```

## Functions

- **`load_heatmap_z_clip_bounds(data_dir: str | Path, *, stats_path: str | Path | None=None)`**
- **`clip_heatmap_z(x: torch.Tensor | np.ndarray, lo: float, hi: float)`**
- **`clip_fraction(x: torch.Tensor | np.ndarray, lo: float, hi: float)`**
- **`heatmap_z_to_physical(hm_z: torch.Tensor, log_mean: float, log_std: float, *, clip_lo: float | None=None, clip_hi: float | None=None, hm_stats: HeatmapNormStats | dict[str, Any] | None=None, mhz: float | torch.Tensor | None=None, pi_norm: torch.Tensor | None=None)`** — Denorm heatmap to physical Ω.
- **`heatmap_norm_to_physical(hm_norm: torch.Tensor, hm_stats: dict[str, Any] | HeatmapNormStats, *, clip_lo: float | None=None, clip_hi: float | None=None, mhz: float | torch.Tensor | None=None, pi_norm: torch.Tensor | None=None)`** — Denorm heatmap tensor to Ω (z-score, robust per-MHz, or global-max).
- **`describe_clip_bounds(data_dir: str | Path)`**

## Imports

- [[norm_stats]]

## Imported by

- [[check_exp044_norm_vs_train]]
- [[dataloader]]
- [[exp043_eval_common]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.inference_vae]]
- [[experiments.exp041.codes.inference_vae]]
- [[experiments.exp042.codes.inference_vae]]
- [[experiments.exp043.codes.inference_vae]]
- [[experiments.exp043.codes.train_vae_simple]]
- [[experiments.exp044.codes.inference_vae]]
- [[experiments.exp045.codes.inference_vae]]
- [[experiments.exp046.codes.eval_real_data_sweep]]
- [[experiments.exp046.codes.inference_vae]]
- [[experiments.exp047.codes.eval_real_data_sweep]]
- [[experiments.exp047.codes.inference_vae]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp048.codes.train_vae_simple]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp049.codes.train_vae_simple]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp050.codes.train_vae_simple]]
- [[experiments.exp051_new_datas_appended.codes.inference_vae]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]
- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
- [[inspect_sweep_artifacts]]
- [[run_multifreq_heatmap_sweep]]
- [[sweep_qc_eval]]

## External dependencies

`numpy`, `torch`
