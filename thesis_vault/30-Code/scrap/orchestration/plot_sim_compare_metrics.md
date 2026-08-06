---
title: plot_sim_compare_metrics
type: code
path: scrap/orchestration/plot_sim_compare_metrics.py
group: scrap/orchestration
loc: 218
tags: [code, scrap]
---

# plot_sim_compare_metrics

> Plot sim_compare_metrics.json from a multifreq sweep run.

**Source:** `scrap/orchestration/plot_sim_compare_metrics.py` · 218 lines

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `DEFAULT_METRICS` | `_REPO / 'experiments/exp050/multifreq_heatmap_sweep_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_…` |

## Functions

- **`load_metrics(path: Path)`**
- **`_pivot(rows: list[dict], field: str)`**
- **`plot_pearson_vs_k(rows: list[dict], out_dir: Path)`**
- **`plot_max_ratio_vs_k(rows: list[dict], out_dir: Path)`**
- **`plot_heatmap(rows: list[dict], field: str, title: str, out_name: str, out_dir: Path, vmin=None, vmax=None)`**
- **`plot_mhz_summary(by_mhz: list[dict], out_dir: Path)`**
- **`plot_by_k_bars(by_k: list[dict], out_dir: Path)`**
- **`main()`**

## External dependencies

`matplotlib`, `numpy`
