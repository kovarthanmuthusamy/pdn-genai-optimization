---
title: inference_vae
type: code
path: experiments/exp057_structured_graph/codes/inference_vae.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 458
tags: [code, exp057_structured_graph, runnable]
---

# inference_vae

> Inference script for Multi-Input VAE — exp052.

**Source:** `experiments/exp057_structured_graph/codes/inference_vae.py` · 458 lines
**Experiment:** [[exp057_structured_graph]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp057_structured_graph/codes/inference_vae.py`

## Purpose

```text
Inference script for Multi-Input VAE — exp052.

Reads data_dir / background from this experiment's config.yaml (datasets/data_multifreq_train_norm_robust).
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_EXP_DIR` | `Path(__file__).resolve().parents[1]` |
| `_CONFIG_PATH` | `_EXP_DIR / 'config.yaml'` |
| `CHECKPOINT_PATH` | `str(_EXP_DIR / 'checkpoints/last_model.pt')` |
| `LATENT_STATS_PATH` | `str(_EXP_DIR / 'metrics/latent_stats.json')` |
| `MODEL_LATENT_DIM` | `48` |
| `NUM_SAMPLES` | `3` |
| `K_VALUE` | `5` |
| `PI_FREQ_MHZ` | `200.0` |
| `OUTPUT_DIR` | `str(_EXP_DIR / 'visuals_K5')` |
| `SAVE_DATA` | `True` |
| `SAVE_PLOTS` | `True` |
| `USE_CUDA` | `True` |
| `SHARED_TEMP` | `1.5` |

## Classes

- **`VAEInference`** — Inference engine for exp055 multifreq PI heatmap VAE.

## Functions

- **`load_experiment_config(path: Path)`** — Parse config.yaml: JSON object with optional full-line ``#`` comments.
- **`_default_data_dir()`**
- **`_norm_stats_path()`**
- **`launch_interactive_viewer(plot_paths)`**
- **`main()`**

## Imports

- [[csv_to_occupancy]]
- [[experiment_paths]]
- [[experiments.exp057_structured_graph.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## External dependencies

`concurrent`, `libs`, `matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`
