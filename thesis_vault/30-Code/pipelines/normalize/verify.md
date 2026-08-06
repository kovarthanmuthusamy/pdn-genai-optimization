---
title: verify
type: code
path: pipelines/normalize/verify.py
group: pipelines/normalize
loc: 233
tags: [code, pipelines, runnable]
---

# verify

> Interactive normalization verification viewer.

**Source:** `pipelines/normalize/verify.py` · 233 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/normalize/verify.py`

## Purpose

```text
Interactive normalization verification viewer.

Run: python pipelines/normalize/verify.py
```

## Constants

| Name | Value |
|------|-------|
| `NORM_ROOT` | `_repo_root / 'datasets' / 'data_up5_norm'` |
| `OUTPUT_DIR` | `_repo_root / 'temp_visuals/normalization_verification'` |
| `SAMPLE_INDEX` | `0` |
| `SENTINEL` | `stats['background_value']` |

## Functions

- **`denorm_heatmap(z_data)`** — z-score → log(1+x) → raw.  Returns (raw, bg_mask).
- **`denorm_impedance(z_data)`** — z-score → log → raw impedance.
- **`load_sample(idx)`** — Load and prepare all data for sample at index idx.
- **`draw(idx)`** — Redraw the entire figure for sample idx.
- **`on_scroll(event)`** — Scroll up → next sample, scroll down → previous sample.

## Imports

- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`
