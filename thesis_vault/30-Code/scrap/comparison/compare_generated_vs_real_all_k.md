---
title: compare_generated_vs_real_all_k
type: code
path: scrap/comparison/compare_generated_vs_real_all_k.py
group: scrap/comparison
loc: 46
tags: [code, scrap, runnable]
---

# compare_generated_vs_real_all_k

> Generated vs real comparison for all K.

**Source:** `scrap/comparison/compare_generated_vs_real_all_k.py` · 46 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/compare_generated_vs_real_all_k.py`

## Purpose

```text
Generated vs real comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and run heatmap+impedance comparison per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real`` for K=1..52 (or a subrange).
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first K that errors
```

## Constants

| Name | Value |
|------|-------|
| `GENERATED_BASE_DIR` | `'scrap/generated_samples_v2'` |
| `FAIL_FAST` | `False` |

## Functions

- **`main()`**

## Imports

- [[batch_over_k]]
- [[compare_generated_vs_real]]
- [[scrap.comparison.__init__]]
