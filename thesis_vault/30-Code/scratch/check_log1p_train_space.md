---
title: check_log1p_train_space
type: code
path: scratch/check_log1p_train_space.py
group: scratch
loc: 56
tags: [code, scratch]
---

# check_log1p_train_space

> Verify log1p train-space roundtrip + skew improvement.

**Source:** `scratch/check_log1p_train_space.py` · 56 lines

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((p for p in _ROOT.parents if (p / 'src_vae').is_dir()), _ROOT.parents[1])` |
| `DATA` | `REPO_ROOT / 'datasets/data_multifreq_gmax'` |

## Imports

- [[experiments.exp043.codes.train_vae_simple]]
- [[gmax_training_patch]]
- [[heatmap_gmax_norm]]

## External dependencies

`experiments`, `numpy`, `src_vae`, `torch`
