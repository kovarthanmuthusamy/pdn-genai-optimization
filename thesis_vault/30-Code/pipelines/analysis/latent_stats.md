---
title: latent_stats
type: code
path: pipelines/analysis/latent_stats.py
group: pipelines/analysis
loc: 83
tags: [code, pipelines, runnable]
---

# latent_stats

> Per-modality latent statistics from a trained VAE checkpoint.

**Source:** `pipelines/analysis/latent_stats.py` · 83 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/latent_stats.py`

## Purpose

```text
Per-modality latent statistics from a trained VAE checkpoint.

Run: python pipelines/analysis/latent_stats.py
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `'experiments/exp021/checkpoints/last_model.pt'` |
| `DATA_ROOT` | `'datasets/data_norm'` |
| `DEVICE` | `'cuda'` |
| `_MODALITIES` | `('heatmap', 'impedance', 'shared')` |

## Functions

- **`compute_latent_stats(checkpoint_path: str, data_root: str=DATA_ROOT, device: str=DEVICE)`**

## Imports

- [[repo_paths]]

## External dependencies

`experiments`, `numpy`, `repo_paths`, `source`, `torch`
