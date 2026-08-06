---
title: exp058_common
type: code
path: experiments/exp058_asymmetric_kl/codes/exp058_common.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 56
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# exp058_common

> exp058 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp058_asymmetric_kl/codes/exp058_common.py` · 56 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

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

- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
