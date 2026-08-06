---
title: compare_generated_vs_real_occupancy
type: code
path: scrap/comparison/compare_generated_vs_real_occupancy.py
group: scrap/comparison
loc: 249
tags: [code, scrap, runnable]
---

# compare_generated_vs_real_occupancy

> Occupancy checkbox plot for one K.

**Source:** `scrap/comparison/compare_generated_vs_real_occupancy.py` · 249 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/compare_generated_vs_real_occupancy.py`

## Purpose

```text
Occupancy checkbox plot for one K.

Purpose:
    Visualize generated 52-slot occupancy vectors (C1..C52) for one K using top-K or threshold policy.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy.py

Agent notes:
    - What: Renders which decap slots are active per generated sample (bar/checkbox view).
    - Usage: Set ``K_VALUE``, ``BASE_GENERATED_DIR``, ``ACTIVE_POLICY`` → run.
    - Config keys:
        - ``K_VALUE``, ``BASE_GENERATED_DIR`` — which K folder to read
        - ``NUM_SAMPLES`` — rows to plot
        - ``ACTIVE_POLICY`` — ``"topk"`` or ``"threshold"``; ``THRESHOLD`` when threshold mode
        - ``OCCUPANCY_OUT_NAME`` — output PNG name
```

## Constants

| Name | Value |
|------|-------|
| `K_VALUE` | `1` |
| `BASE_GENERATED_DIR` | `Path('scrap/generated_samples')` |
| `OCCUPANCY_OUT_NAME` | `'generated_occupancy.png'` |

## Functions

- **`_project_root()`**
- **`_infer_num_samples(k_dir: Path)`**
- **`_load_occupancy_matrix(k_dir: Path, *, num_samples: int)`** — Load generated occupancy vectors for all samples under a K folder.
- **`_plot_checkboxes(*, occupancy_matrix: np.ndarray, out_path: Path, title_prefix: str, expected_k: int | None=None)`** — Render compact checkbox grids labeled C1..C52.
- **`main()`**

## Imported by

- [[compare_generated_vs_real_occupancy_all_k]]

## External dependencies

`matplotlib`, `numpy`
