---
title: exp060_common
type: code
path: experiments/exp060_multitype_occ/codes/exp060_common.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 57
tags: [code, exp060_multitype_occ, uncommitted]
---

# exp060_common

> exp060 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp060_multitype_occ/codes/exp060_common.py` · 57 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

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

- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
