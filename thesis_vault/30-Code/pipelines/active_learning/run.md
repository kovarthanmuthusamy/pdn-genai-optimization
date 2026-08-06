---
title: run
type: code
path: pipelines/active_learning/run.py
group: pipelines/active_learning
loc: 82
tags: [code, pipelines, runnable]
---

# run

> Active-learning PI pipeline (exp059 capacity/freq by default).

**Source:** `pipelines/active_learning/run.py` · 82 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/active_learning/run.py`

## Purpose

```text
Active-learning PI pipeline (exp059 capacity/freq by default).

Run:
    python pipelines/active_learning/run.py

Edit CONFIG below, then run the command above.

``full`` = all 8 steps: generate → infer → select → ECAD → ingest → overlay → fine-tune → post-finetune eval.
Use ``active_learning_pi/config/exp057.json`` / ``exp058.json`` only for legacy runs.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `COMMAND` | `'full'` |
| `CONFIG_PATH` | `'active_learning_pi/config/exp059_gp_error.json'` |
| `ITERATION` | `None` |
| `SKIP_SIMULATE` | `False` |
| `SKIP_INGEST` | `False` |
| `PROPOSE_ONLY` | `False` |

## Functions

- **`main()`**

## Imports

- [[pipeline]]
- [[repo_paths]]

## External dependencies

`active_learning_pi`, `repo_paths`
