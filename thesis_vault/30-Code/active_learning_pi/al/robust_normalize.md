---
title: robust_normalize
type: code
path: active_learning_pi/al/robust_normalize.py
group: active_learning_pi/al
loc: 60
tags: [code, active_learning_pi]
---

# robust_normalize

> Robust per-MHz heatmap normalization for active-learning labels.

**Source:** `active_learning_pi/al/robust_normalize.py` · 60 lines

## Purpose

```text
Robust per-MHz heatmap normalization for active-learning labels.

Matches ``pipelines/normalize/multifreq.py`` rules used by
``datasets/data_multifreq_train_norm_unbounded``.
```

## Functions

- **`_ensure_repo_on_path(groot: Path)`**
- **`load_heatmap_stats(stats_json: Path)`**
- **`is_robust_per_mhz_stats(heatmap_stats: dict[str, Any])`**
- **`normalize_heatmap_raw(raw: np.ndarray, hm_stats: dict[str, Any], *, mhz: float | None=None, groot: Path | None=None)`** — Normalize raw (2,H,W) or (1,H,W) heatmap using training stats.

## Imports

- [[multifreq]]

## Imported by

- [[build_overlay]]
- [[normalize_labels]]

## External dependencies

`numpy`, `pipelines`
