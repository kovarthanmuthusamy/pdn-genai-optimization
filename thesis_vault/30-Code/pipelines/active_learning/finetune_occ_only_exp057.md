---
title: finetune_occ_only_exp057
type: code
path: pipelines/active_learning/finetune_occ_only_exp057.py
group: pipelines/active_learning
loc: 123
tags: [code, pipelines, runnable, uncommitted]
---

# finetune_occ_only_exp057

> Pre-AL occupancy-only warm-up for exp057 (no overlay).

**Source:** `pipelines/active_learning/finetune_occ_only_exp057.py` · 123 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/active_learning/finetune_occ_only_exp057.py`

## Purpose

```text
Pre-AL occupancy-only warm-up for exp057 (no overlay).

Trains decode-from-layout-only to match AL candidate scoring, then run AL:

    python pipelines/active_learning/finetune_occ_only_exp057.py
    python pipelines/active_learning/run.py   # COMMAND=propose or full

Edit CONFIG below.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `CONFIG_PATH` | `'active_learning_pi/config/exp057.json'` |
| `OCC_ONLY_CONFIG` | `'experiments/exp057_structured_graph/config_occ_only_finetune.yaml'` |
| `EXTRA_EPOCHS` | `100` |

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
