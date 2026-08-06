---
title: exp054_common
type: code
path: experiments/exp054_K_30/codes/exp054_common.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 52
tags: [code, exp054_K_30]
---

# exp054_common

> exp054 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp054_K_30/codes/exp054_common.py` · 52 lines
**Experiment:** [[exp054_K_30]]

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

- [[experiments.exp054_K_30.codes.train_vae_simple]]
