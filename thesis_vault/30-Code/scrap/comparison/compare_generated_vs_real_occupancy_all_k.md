---
title: compare_generated_vs_real_occupancy_all_k
type: code
path: scrap/comparison/compare_generated_vs_real_occupancy_all_k.py
group: scrap/comparison
loc: 46
tags: [code, scrap, runnable]
---

# compare_generated_vs_real_occupancy_all_k

> Occupancy comparison for all K.

**Source:** `scrap/comparison/compare_generated_vs_real_occupancy_all_k.py` · 46 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/compare_generated_vs_real_occupancy_all_k.py`

## Purpose

```text
Occupancy comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and render occupancy checkbox plots per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real_occupancy``.
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first error
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
- [[compare_generated_vs_real_occupancy]]
- [[scrap.comparison.__init__]]
