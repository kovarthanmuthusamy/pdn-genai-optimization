---
title: compare_generated_vs_real
type: code
path: scrap/comparison/compare_generated_vs_real.py
group: scrap/comparison
loc: 550
tags: [code, scrap, runnable]
---

# compare_generated_vs_real

> Compare generated vs real for one K.

**Source:** `scrap/comparison/compare_generated_vs_real.py` · 550 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/compare_generated_vs_real.py`

## Purpose

```text
Compare generated vs real for one K.

Purpose:
    Plot heatmap and impedance overlays for one K folder (generated ``.npy`` vs ECADStar ``Real/`` exports).

Run:
    python scrap/comparison/compare_generated_vs_real.py

Agent notes:
    - What: Visual QA — generated VAE samples vs ground-truth PI simulation for a single decap budget.
    - Usage: Set ``K_VALUE`` and ``BASE_GENERATED_DIR`` → run. Also see ``comparison/compare.py``.
    - Config keys:
        - ``K_VALUE`` — decap budget folder ``K{n}`` under base dir
        - ``BASE_GENERATED_DIR`` — root containing ``K1/``, ``K2/``, …
        - ``NUM_SAMPLES`` — how many ``data_sample_*`` to plot
        - ``FREQUENCY_PATH``, ``TARGET_IMPEDANCE_PATH``, ``MASK_PATH`` — shared config arrays
        - ``HEATMAP_OUT_NAME``, ``IMPEDANCE_OUT_NAME`` — output PNG filenames
```

## Constants

| Name | Value |
|------|-------|
| `K_VALUE` | `5` |
| `BASE_GENERATED_DIR` | `Path('scrap/generated_samples_v2')` |
| `FREQUENCY_PATH` | `Path('configs/Frequency_data_hz.npy')` |
| `TARGET_IMPEDANCE_PATH` | `Path('configs/target_impedance.npy')` |
| `MASK_PATH` | `Path('configs/binary_mask.npy')` |
| `HEATMAP_OUT_NAME` | `'generated_vs_real_heatmap.png'` |
| `IMPEDANCE_OUT_NAME` | `'generated_vs_real_impedance_profile.png'` |
| `HEATMAP_CMAP` | `'jet'` |
| `HEATMAP_LEVELS` | `22` |
| `HEATMAP_DIFF_TOLERANCE` | `0.25` |
| `HEATMAP_PATTERN_DIFF_TOLERANCE` | `0.05` |
| `MAP_GLOB_PREFERENCE` | `('Z_*.map', '*.map')` |
| `IMPEDANCE_CSV_GLOB_PREFERENCE` | `('*PIPinZ*.csv', '*.csv')` |

## Functions

- **`_project_root()`**
- **`_infer_num_samples(k_dir: Path)`**
- **`_load_generated_heatmap(path: Path)`**
- **`_load_map_file(file_path: Path, *, resolution: int)`** — Load real heatmap from a .map file and interpolate to a square grid.
- **`_choose_real_heatmap_mapfile(heatmap_real_item: Path)`** — Given Heatmap_real_{i}* item, return the concrete .map file to read.
- **`_choose_real_impedance_csv(imp_real_item: Path)`** — Given Imp_Real{i}* item, return the concrete .csv file to read.
- **`_iter_numeric_rows(path: Path)`** — Yield (x, y) pairs from a CSV/TSV/space-delimited file.
- **`_load_real_impedance(csv_path: Path, *, frequency_hz: np.ndarray)`** — Load real impedance in Ohms; interpolate to `frequency_hz` if frequency column exists.
- **`_load_generated_impedance_log(path: Path)`**
- **`_maybe_load(path: Path)`**
- **`_plot_heatmap_comparisons(*, comparisons: list[dict], mask: np.ndarray, out_path: Path)`**
- **`_plot_impedance_comparisons(*, frequency: np.ndarray, target_impedance: np.ndarray, comparisons: list[dict], out_path: Path)`**
- **`main()`**

## Imported by

- [[compare_generated_vs_real_all_k]]

## External dependencies

`matplotlib`, `numpy`, `scipy`
