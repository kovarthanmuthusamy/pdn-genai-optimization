---
title: exp059_common
type: code
path: experiments/exp059_capacity_freq/codes/exp059_common.py
group: experiments/exp059_capacity_freq/codes
experiment: exp059_capacity_freq
loc: 56
tags: [code, exp059_capacity_freq, uncommitted]
---

# exp059_common

> exp059 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp059_capacity_freq/codes/exp059_common.py` · 56 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp059_capacity_freq]]

## Constants

| Name | Value |
|------|-------|
| `REPO_ROOT` | `Path(__file__).resolve().parents[3]` |
| `EXP_DIR` | `Path(__file__).resolve().parents[1]` |

## Functions

- **`load_yaml_config(path: Path | None=None)`**
- **`_cfg_get(cfg: dict[str, Any] | Any, key: str, default: Any=None)`**
- **`vae_model_kwargs(cfg: dict[str, Any] | Any)`**

## Imported by

- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]
