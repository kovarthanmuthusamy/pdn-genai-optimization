---
title: find_feasible
type: code
path: pipelines/latent/find_feasible.py
group: pipelines/latent
loc: 433
tags: [code, pipelines, runnable]
---

# find_feasible

> Find feasible configs — Monte Carlo sampling and filtering via VAE surrogate.

**Source:** `pipelines/latent/find_feasible.py` · 433 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/latent/find_feasible.py`

## Purpose

```text
Find feasible configs — Monte Carlo sampling and filtering via VAE surrogate.

Run:
    python pipelines/latent/find_feasible.py
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `'experiments/exp029_heat_private/checkpoints/checkpoint_epoch_400.pt'` |
| `K_LIST` | `[1, 2, 3, 4, 5]` |
| `NUM_SAMPLES_PER_K` | `2000` |
| `SAVE_TOP_N` | `20` |
| `ENSURE_UNIQUE_OCC_TOPK` | `True` |
| `SAMPLE_MODE` | `'agg_posterior'` |
| `SHARED_TEMP` | `1.0` |
| `LOSS_SPACE` | `'log'` |
| `BOUNDARY_MARGIN` | `0.0` |
| `NORMALIZATION_STATS_PATH` | `'datasets/data_norm/normalization_stats.json'` |
| `TARGET_IMPEDANCE_PATH` | `'configs/target_impedance.npy'` |
| `OUTPUT_ROOT` | `'data/latent_runs'` |
| `DEVICE` | `'cuda' if torch.cuda.is_available() else 'cpu'` |
| `DTYPE` | `torch.float32` |

## Classes

- **`NormStats`**

## Functions

- **`_project_root()`**
- **`_load_json(path: Path)`**
- **`_load_norm_stats(stats_path: Path)`**
- **`_blend_impedance_norm_log(imp_norm: torch.Tensor)`**
- **`_denorm_impedance_log(imp_log_norm: torch.Tensor, stats: NormStats)`**
- **`_latent_prior_params(*, latent_dim: int, device: torch.device, K: int, latent_stats: dict | None, per_K_latent_stats: dict | None)`**
- **`_sample_z(*, num_samples: int, latent_dim: int, device: torch.device, K: int, latent_stats: dict | None, per_K_latent_stats: dict | None, shared_temp: float)`**
- **`_topk_occupancy(occ_prob: torch.Tensor, K: int)`**
- **`_decode_occupancy_prob(*, model: torch.nn.Module, z: torch.Tensor, K: int, device: torch.device)`** — Decode occupancy probabilities (B,52) without decoding heatmap.
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`experiments`, `numpy`, `repo_paths`, `torch`
