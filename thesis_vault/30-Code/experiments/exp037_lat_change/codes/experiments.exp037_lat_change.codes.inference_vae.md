---
title: inference_vae
type: code
path: experiments/exp037_lat_change/codes/inference_vae.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 315
tags: [code, exp037_lat_change, runnable]
---

# inference_vae

> Inference script for Multi-Input VAE — exp037_lat_change.

**Source:** `experiments/exp037_lat_change/codes/inference_vae.py` · 315 lines
**Experiment:** [[exp037_lat_change]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp037_lat_change/codes/inference_vae.py`

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `f'{exp}/checkpoints/last_model.pt'` |
| `LATENT_STATS_PATH` | `f'{exp}/metrics/latent_stats.json'` |
| `MODEL_LATENT_DIM` | `32` |
| `NUM_SAMPLES` | `3` |
| `K_VALUE` | `5` |
| `OUTPUT_DIR` | `f'{exp}/visuals_K5'` |
| `SAVE_DATA` | `True` |
| `SAVE_PLOTS` | `True` |
| `USE_CUDA` | `True` |
| `SHARED_TEMP` | `1.5` |

## Classes

- **`VAEInference`** — Inference engine for generating samples from the exp037_lat_change trained VAE decoder.

## Functions

- **`launch_interactive_viewer(plot_paths)`** — Interactive plot viewer: ← → to navigate, Esc to exit.
- **`main()`**

## Imports

- [[csv_to_occupancy]]
- [[experiments.exp037_lat_change.codes.vae_multi_input_simple]]
- [[repo_paths]]

## External dependencies

`concurrent`, `libs`, `matplotlib`, `numpy`, `repo_paths`, `torch`
