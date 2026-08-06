---
title: exp056_common
type: code
path: experiments/exp056_graph_vae/codes/exp056_common.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 55
tags: [code, exp056_graph_vae]
---

# exp056_common

> exp056 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp056_graph_vae/codes/exp056_common.py` · 55 lines
**Experiment:** [[exp056_graph_vae]]

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

- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
