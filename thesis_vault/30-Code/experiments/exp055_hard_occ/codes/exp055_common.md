---
title: exp055_common
type: code
path: experiments/exp055_hard_occ/codes/exp055_common.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 52
tags: [code, exp055_hard_occ]
---

# exp055_common

> exp055 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp055_hard_occ/codes/exp055_common.py` · 52 lines
**Experiment:** [[exp055_hard_occ]]

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

- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
