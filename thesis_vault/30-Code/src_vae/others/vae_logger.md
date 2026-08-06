---
title: vae_logger
type: code
path: src_vae/others/vae_logger.py
group: src_vae/others
loc: 728
tags: [code, src_vae]
---

# vae_logger

> VAE training metrics logger (CSV, plots, checkpoints, console).

**Source:** `src_vae/others/vae_logger.py` · 728 lines

## Purpose

```text
VAE training metrics logger (CSV, plots, checkpoints, console).

Run: Instantiated in experiment training scripts as ``VAETrainingLogger(log_dir=...)``.
```

## Constants

| Name | Value |
|------|-------|
| `CSV_HEADER` | `['epoch', 'train_total_loss', 'train_recon_loss', 'train_kl_loss', 'train_heatmap_loss', …` |
| `LATENT_CSV_HEADER` | `['epoch', 'beta', 'hm_mu_mean', 'hm_sigma_mean', 'occ_mu_mean', 'occ_sigma_mean', 'imp_mu…` |

## Classes

- **`VAETrainingLogger`** — Logger for tracking and visualizing VAE training metrics

## Functions

- **`_format_epoch_k(x: float, _pos: int)`** — Format epoch ticks: plain below 1000, 'k' suffix at/above 1000.
- **`_epoch_tick_step(n_epochs: int)`** — Pick a round epoch step that keeps ~6–12 readable x-axis labels.
- **`_apply_dense_ticks(ax, epochs: np.ndarray, *, y_nbins: int=16, x_nbins: int=24)`** — Dense y-axis ticks; x-axis adapts so long runs (>1k epochs) stay readable.

## Imported by

- [[experiments.exp037_lat_change.codes.train_vae_simple]]
- [[experiments.exp038_true_multi.codes.metrics_csv_utils]]
- [[experiments.exp038_true_multi.codes.refresh_exp038_plots]]
- [[experiments.exp038_true_multi.codes.save_epoch1_losses]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.metrics_csv_utils]]
- [[experiments.exp039_improved_heatmap.codes.codes.refresh_exp038_plots]]
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.metrics_csv_utils]]
- [[experiments.exp039_improved_heatmap.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.metrics_csv_utils]]
- [[experiments.exp040.codes.codes.refresh_exp038_plots]]
- [[experiments.exp040.codes.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.metrics_csv_utils]]
- [[experiments.exp040.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.metrics_csv_utils]]
- [[experiments.exp041.codes.codes.refresh_exp038_plots]]
- [[experiments.exp041.codes.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.metrics_csv_utils]]
- [[experiments.exp041.codes.save_epoch1_losses]]
- [[experiments.exp042.codes.metrics_csv_utils]]
- [[experiments.exp043.codes.metrics_csv_utils]]
- [[experiments.exp043.codes.train_vae_simple]]
- [[experiments.exp044.codes.metrics_csv_utils]]
- [[experiments.exp045.codes.metrics_csv_utils]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.train_core]]

## External dependencies

`matplotlib`, `numpy`, `yaml`
