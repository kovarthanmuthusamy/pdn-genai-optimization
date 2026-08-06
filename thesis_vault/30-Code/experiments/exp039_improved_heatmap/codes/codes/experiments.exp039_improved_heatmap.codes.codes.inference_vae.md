---
title: inference_vae
type: code
path: experiments/exp039_improved_heatmap/codes/codes/inference_vae.py
group: experiments/exp039_improved_heatmap/codes/codes
experiment: exp039_improved_heatmap
loc: 353
tags: [code, exp039_improved_heatmap, runnable]
---

# inference_vae

> Inference script for Multi-Input VAE — exp038_true_multi.

**Source:** `experiments/exp039_improved_heatmap/codes/codes/inference_vae.py` · 353 lines
**Experiment:** [[exp039_improved_heatmap]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp039_improved_heatmap/codes/codes/inference_vae.py`

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_EXP_DIR` | `Path(__file__).resolve().parents[1]` |
| `_CONFIG_PATH` | `_EXP_DIR / 'config.yaml'` |
| `CHECKPOINT_PATH` | `f'{exp}/checkpoints/last_model.pt'` |
| `LATENT_STATS_PATH` | `f'{exp}/metrics/latent_stats.json'` |
| `MODEL_LATENT_DIM` | `32` |
| `NUM_SAMPLES` | `3` |
| `K_VALUE` | `5` |
| `PI_FREQ_MHZ` | `100` |
| `OUTPUT_DIR` | `f'{exp}/visuals_K5'` |
| `SAVE_DATA` | `True` |
| `SAVE_PLOTS` | `True` |
| `USE_CUDA` | `True` |
| `SHARED_TEMP` | `1.5` |

## Classes

- **`VAEInference`** — Inference engine for generating samples from the exp038 trained VAE decoder.

## Functions

- **`_default_data_dir()`** — Dataset dir from exp038 config.yaml, else multifreq_norm default.
- **`_norm_stats_path()`**
- **`launch_interactive_viewer(plot_paths)`** — Interactive plot viewer: ← → to navigate, Esc to exit.
- **`main()`**

## Imports

- [[csv_to_occupancy]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[repo_paths]]

## External dependencies

`concurrent`, `libs`, `matplotlib`, `numpy`, `repo_paths`, `torch`
