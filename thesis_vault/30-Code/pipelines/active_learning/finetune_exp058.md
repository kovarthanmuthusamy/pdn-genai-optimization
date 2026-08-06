---
title: finetune_exp058
type: code
path: pipelines/active_learning/finetune_exp058.py
group: pipelines/active_learning
loc: 95
tags: [code, pipelines, runnable, uncommitted]
---

# finetune_exp058

> Fine-tune exp058 after active-learning (asymmetric: near AL path + distill).

**Source:** `pipelines/active_learning/finetune_exp058.py` · 95 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/active_learning/finetune_exp058.py`

## Purpose

```text
Fine-tune exp058 after active-learning (asymmetric: near AL path + distill).

Run:
    python pipelines/active_learning/finetune_exp058.py

Edit CONFIG below. Usually invoked by ``run.py`` COMMAND=full with exp058.json.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `CONFIG_PATH` | `'active_learning_pi/config/exp058.json'` |
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
