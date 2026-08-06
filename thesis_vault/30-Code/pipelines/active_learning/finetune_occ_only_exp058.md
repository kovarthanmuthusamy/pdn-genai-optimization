---
title: finetune_occ_only_exp058
type: code
path: pipelines/active_learning/finetune_occ_only_exp058.py
group: pipelines/active_learning
loc: 123
tags: [code, pipelines, runnable, uncommitted]
---

# finetune_occ_only_exp058

> Pre-AL occupancy-only warm-up for exp058 (no overlay).

**Source:** `pipelines/active_learning/finetune_occ_only_exp058.py` · 123 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/active_learning/finetune_occ_only_exp058.py`

## Purpose

```text
Pre-AL occupancy-only warm-up for exp058 (no overlay).

Trains decode-from-layout-only to match AL candidate scoring, then run AL:

    python pipelines/active_learning/finetune_occ_only_exp058.py
    python pipelines/active_learning/run.py   # CONFIG_PATH=exp058.json

Edit CONFIG below.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `CONFIG_PATH` | `'active_learning_pi/config/exp058.json'` |
| `OCC_ONLY_CONFIG` | `'experiments/exp058_asymmetric_kl/config_occ_only_finetune.yaml'` |
| `EXTRA_EPOCHS` | `30` |

## Functions

- **`_load_yaml_like(path: Path)`**
- **`_write_yaml_like(path: Path, data: dict)`**
- **`main()`**

## Imports

- [[config]]
- [[finetune_run]]
- [[repo_paths]]

## External dependencies

`active_learning_pi`, `repo_paths`
