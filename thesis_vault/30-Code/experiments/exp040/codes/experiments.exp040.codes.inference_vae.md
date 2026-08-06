---
title: inference_vae
type: code
path: experiments/exp040/codes/inference_vae.py
group: experiments/exp040/codes
experiment: exp040
loc: 132
tags: [code, exp040]
---

# inference_vae

> Inference for exp040 FactorizedFreqVAE.

**Source:** `experiments/exp040/codes/inference_vae.py` · 132 lines
**Experiment:** [[exp040]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_EXP_DIR` | `Path(__file__).resolve().parents[1]` |
| `_CONFIG_PATH` | `_EXP_DIR / 'config.yaml'` |
| `CHECKPOINT_PATH` | `f'{exp}/checkpoints/last_model.pt'` |
| `LATENT_STATS_PATH` | `f'{exp}/metrics/latent_stats.json'` |
| `MODEL_LATENT_DIM` | `42` |
| `SHARED_TEMP` | `1.5` |

## Classes

- **`VAEInference`** — Load FactorizedFreqVAE and run layout / anchor_blend inference.

## Functions

- **`load_experiment_config(path: Path)`**
- **`_default_data_dir()`**
- **`_norm_stats_path()`**

## Imports

- [[repo_paths]]
- [[vae_factorized_freq]]

## Imported by

- [[experiments.exp040.codes.eval_cross_freq]]
- [[inspect_sweep_artifacts]]

## External dependencies

`numpy`, `repo_paths`, `torch`
