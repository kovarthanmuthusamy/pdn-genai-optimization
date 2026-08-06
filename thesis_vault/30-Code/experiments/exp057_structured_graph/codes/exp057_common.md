---
title: exp057_common
type: code
path: experiments/exp057_structured_graph/codes/exp057_common.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 56
tags: [code, exp057_structured_graph]
---

# exp057_common

> exp057 shared paths, yaml config, and VAE constructor kwargs.

**Source:** `experiments/exp057_structured_graph/codes/exp057_common.py` · 56 lines
**Experiment:** [[exp057_structured_graph]]

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

- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
