---
title: impedance_decade_diversity
type: code
path: pipelines/analysis/impedance_decade_diversity.py
group: pipelines/analysis
loc: 203
tags: [code, pipelines, runnable]
---

# impedance_decade_diversity

> Per-frequency-bin impedance diversity across layouts (231 PI-spectrum bins).

**Source:** `pipelines/analysis/impedance_decade_diversity.py` · 203 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/impedance_decade_diversity.py`

## Purpose

```text
Per-frequency-bin impedance diversity across layouts (231 PI-spectrum bins).

X-axis: ``configs/Frequency_data_hz.npy`` (Hz) — same as PI-spectrum / impedance plots.
Y-axis: magnitude |Z| stored in ``layouts/<design_id>/imp.npy``.

For each of the 231 bins, measures layout-to-layout spread (std, IQR, CV).

Run:
    python pipelines/analysis/impedance_decade_diversity.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `DATA_ROOT` | `repo_path('datasets', 'data_multifreq_train')` |
| `FREQ_PATH` | `repo_path('configs', 'Frequency_data_hz.npy')` |
| `REPORT_JSON` | `repo_path('experiments', 'impedance_freq_diversity.json')` |
| `REPORT_NPZ` | `repo_path('experiments', 'impedance_freq_diversity.npz')` |
| `SEED` | `0` |
| `TOP_N` | `25` |
| `USE_LOG10_MAGNITUDE` | `True` |

## Functions

- **`_load_imp(path: Path)`**
- **`_iter_layout_imps(layouts_dir: Path, *, max_layouts: int | None, seed: int)`**
- **`analyze_per_bin_diversity(imps: np.ndarray, freq_hz: np.ndarray)`** — Diversity at each of the 231 frequency bins across layouts.
- **`print_report(stats: dict, *, n_layouts: int, top_n: int)`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`
