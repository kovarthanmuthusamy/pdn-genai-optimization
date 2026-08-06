---
title: eval_real_data_sweep
type: code
path: experiments/exp046/codes/eval_real_data_sweep.py
group: experiments/exp046/codes
experiment: exp046
loc: 356
tags: [code, exp046]
---

# eval_real_data_sweep

> Evaluate exp046 heatmap quality using REAL dataset layouts.

**Source:** `experiments/exp046/codes/eval_real_data_sweep.py` · 356 lines
**Experiment:** [[exp046]]

## Purpose

```text
Evaluate exp046 heatmap quality using REAL dataset layouts.

Loads real (occ, imp, heatmap) from val split and evaluates:
  1. encode  : encode(hm, occ, imp, K, pi) → decode → compare with real hm
  2. layout  : encode_layout_latent(occ, imp, K, pi) → decode → compare with real hm

Reports per-frequency: MAE, FG-MSE, Pearson r, peak-loc error, p95, p99.9, max.
Saves side-by-side comparison images (real vs generated).

Run:
  cd /home/ubuntu/genai_pdn
  python -m experiments.exp046.codes.eval_real_data_sweep
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `MAX_SAMPLES_PER_MHZ` | `8` |
| `MAX_BATCHES` | `0` |
| `NUM_PLOT_SAMPLES` | `2` |
| `OUTPUT_DIR_NAME` | `'eval_real_data_sweep'` |

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float=0.5)`**
- **`evaluate(model, val_loader: DataLoader, *, bg: float, eval_mhz: list[float], log_mean: float, log_std: float, clip_lo: float | None, clip_hi: float | None, mask: np.ndarray, max_samples: int, max_batches: int, device: torch.device)`** — Run eval on val set, return per-(mhz, mode) stats + sample data for plots.
- **`_summarise(buckets: dict)`**
- **`_write_csv(rows: list[dict], path: Path)`**
- **`_save_plots(plot_samples: dict, mask: np.ndarray, out_dir: Path)`**
- **`main()`**

## Imports

- [[exp046_eval_common]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp046.codes.spatial_metrics]]
- [[heatmap_z_clip]]
- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`
