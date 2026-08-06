---
title: inference_vae
type: code
path: experiments/exp043/codes/inference_vae.py
group: experiments/exp043/codes
experiment: exp043
loc: 500
tags: [code, exp043, runnable]
---

# inference_vae

> VAE inference for exp043 — generate heatmap, occupancy, impedance samples.

**Source:** `experiments/exp043/codes/inference_vae.py` · 500 lines
**Experiment:** [[exp043]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp043/codes/inference_vae.py`

## Purpose

```text
VAE inference for exp043 — generate heatmap, occupancy, impedance samples.

Run:
    python experiments/exp043/codes/inference_vae.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_EXP_DIR` | `Path(__file__).resolve().parents[1]` |
| `_CONFIG_PATH` | `_EXP_DIR / 'config.yaml'` |
| `CHECKPOINT_PATH` | `str(_EXP_DIR / 'checkpoints/last_model.pt')` |
| `LATENT_STATS_PATH` | `str(_EXP_DIR / 'metrics/latent_stats.json')` |
| `MODEL_LATENT_DIM` | `42` |
| `NUM_SAMPLES` | `3` |
| `K_VALUE` | `5` |
| `PI_FREQ_MHZ` | `200.0` |
| `OUTPUT_DIR` | `str(_EXP_DIR / 'visuals_K5')` |
| `SAVE_DATA` | `True` |
| `SAVE_PLOTS` | `True` |
| `USE_CUDA` | `True` |
| `SHARED_TEMP` | `1.5` |

## Classes

- **`VAEInference`** — Inference engine for exp043 multifreq PI heatmap VAE.

## Functions

- **`load_experiment_config(path: Path)`** — Parse config.yaml: JSON object with optional full-line ``#`` comments.
- **`_default_data_dir()`**
- **`_norm_stats_path()`**
- **`launch_interactive_viewer(plot_paths)`**
- **`main()`**

## Imports

- [[csv_to_occupancy]]
- [[experiment_paths]]
- [[experiments.exp043.codes.vae_poe_freq]]
- [[heatmap_gmax_norm]]
- [[heatmap_z_clip]]
- [[repo_paths]]

## Imported by

- [[diagnose_sweep_vs_dataset]]

## External dependencies

`concurrent`, `libs`, `matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`
