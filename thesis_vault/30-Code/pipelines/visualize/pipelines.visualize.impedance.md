---
title: impedance
type: code
path: pipelines/visualize/impedance.py
group: pipelines/visualize
loc: 61
tags: [code, pipelines, runnable]
---

# impedance

> Impedance profile log-log comparison plotter.

**Source:** `pipelines/visualize/impedance.py` · 61 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/visualize/impedance.py`

## Purpose

```text
Impedance profile log-log comparison plotter.

Run: python pipelines/visualize/impedance.py
```

## Constants

| Name | Value |
|------|-------|
| `FREQ_PATH` | `repo_path('configs', 'Frequency_data_hz.npy')` |
| `TARGET_IMP_PATH` | `repo_path('configs', 'target_impedance.npy')` |
| `OUTPUT_PATH` | `repo_path('temp_visuals', 'generated_vs_target_impedance_npy.png')` |

## Functions

- **`plot_impedance_profile(output_path: str | Path, *, generated_impedance: np.ndarray | None=None, freq_path: Path=FREQ_PATH, target_path: Path=TARGET_IMP_PATH)`**

## Imports

- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`
