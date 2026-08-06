---
title: evaluate_off_anchor
type: code
path: active_learning_pi/al/evaluate_off_anchor.py
group: active_learning_pi/al
loc: 239
tags: [code, active_learning_pi]
---

# evaluate_off_anchor

> Evaluate model predictions vs ECADSTAR labels at off-anchor MHz frequencies.

**Source:** `active_learning_pi/al/evaluate_off_anchor.py` · 239 lines

## Purpose

```text
Evaluate model predictions vs ECADSTAR labels at off-anchor MHz frequencies.

Run:
    python active_learning_pi/al/evaluate_off_anchor.py
```

## Functions

- **`evaluate_labels_vs_predictions(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]], *, off_anchor_mhz: list[float])`** — Compare ingested real heatmaps to model predictions (pred_* fields) at off-anchor MHz.
- **`_rankdata(x: np.ndarray)`** — Average ranks for ties (1-based), matching scipy.stats.rankdata(method='average').
- **`_corr(x: np.ndarray, y: np.ndarray)`**
- **`_spearman(x: np.ndarray, y: np.ndarray)`**
- **`evaluate_acquisition_rank_quality(manifest: list[dict[str, Any]], scored: list[dict[str, Any]], *, off_anchor_mhz: list[float], selected: list[dict[str, Any]] | None=None)`** — Check whether acquisition badness ranks true p99 error on labeled layouts.

## Imports

- [[robust_stats]]

## Imported by

- [[evaluate_cycle]]
- [[pipeline]]

## External dependencies

`numpy`
