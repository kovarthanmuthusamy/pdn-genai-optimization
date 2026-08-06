---
title: finetune_exp057
type: code
path: pipelines/active_learning/finetune_exp057.py
group: pipelines/active_learning
loc: 90
tags: [code, pipelines, runnable]
---

# finetune_exp057

> Fine-tune exp057 after active-learning (Option B: heatmap-only overlay).

**Source:** `pipelines/active_learning/finetune_exp057.py` · 90 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/active_learning/finetune_exp057.py`

## Purpose

```text
Fine-tune exp057 after active-learning (Option B: heatmap-only overlay).

Run:
    python pipelines/active_learning/finetune_exp057.py

Edit CONFIG below. Usually invoked automatically by ``run.py`` COMMAND=full.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `CONFIG_PATH` | `'active_learning_pi/config/exp057.json'` |
| `SKIP_OVERLAY_BUILD` | `False` |
| `TRAIN_ONLY` | `False` |

## Functions

- **`main()`**

## Imports

- [[build_overlay]]
- [[config]]
- [[finetune_run]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`active_learning_pi`, `repo_paths`, `src_vae`
