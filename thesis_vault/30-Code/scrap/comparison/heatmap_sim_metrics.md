---
title: heatmap_sim_metrics
type: code
path: scrap/comparison/heatmap_sim_metrics.py
group: scrap/comparison
loc: 317
tags: [code, scrap]
---

# heatmap_sim_metrics

> Metrics for generated vs ECADStar-simulated real heatmaps (post-move).

**Source:** `scrap/comparison/heatmap_sim_metrics.py` · 317 lines

## Purpose

```text
Metrics for generated vs ECADStar-simulated real heatmaps (post-move).

Compares ``heatmap_physical.npy`` (generated) against interpolated ``Z_*MHz.map``
from each ``K{n}/Real/`` folder — not dataset val heatmaps.
```

## Constants

| Name | Value |
|------|-------|
| `SIM_METRICS_CSV` | `'sim_compare_metrics.csv'` |
| `SIM_METRICS_JSON` | `'sim_compare_metrics.json'` |
| `SIM_METRICS_MD` | `'sim_compare_metrics.md'` |

## Functions

- **`_peak_location_2d(arr: np.ndarray, mask: np.ndarray)`** — Argmax (row, col) within mask; ties → first occurrence.
- **`_spearman_rho(x: np.ndarray, y: np.ndarray)`**
- **`compute_sim_heatmap_metrics(real: np.ndarray, gen: np.ndarray, mask: np.ndarray, *, mhz: int | float | None=None, k: int | None=None, sample: int | None=None, label: str='', diff_tolerance: float=0.0, pattern_diff_tolerance: float=0.0)`** — Full error metrics between simulated-real and generated heatmaps (Ω, masked FG).
- **`_aggregate_by_key(rows: list[dict[str, Any]], key: str)`** — Mean numeric metrics grouped by ``key`` (e.g. mhz or k).
- **`format_metrics_table(rows: list[dict[str, Any]], *, columns: Iterable[str] | None=None)`**
- **`write_sim_metrics_bundle(rows: list[dict[str, Any]], out_root: Path, *, source: str='ecadstar_simulated_real')`** — Write CSV + JSON + agent-friendly markdown under ``out_root``.
- **`print_metrics_summary(row: dict[str, Any])`** — One-line stdout summary for a single comparison (primary metrics first).

## Imported by

- [[build_comparison_report]]
- [[compare]]
- [[run_multifreq_sweep_pipeline]]

## External dependencies

`numpy`, `scipy`
