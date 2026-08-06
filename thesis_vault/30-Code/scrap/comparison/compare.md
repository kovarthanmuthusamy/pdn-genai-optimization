---
title: compare
type: code
path: scrap/comparison/compare.py
group: scrap/comparison
loc: 1047
tags: [code, scrap, runnable]
---

# compare

> Compare Generated vs Real (Workflow-Aware).

**Source:** `scrap/comparison/compare.py` · 1047 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/compare.py`

## Purpose

```text
Compare Generated vs Real (Workflow-Aware).

Purpose: Plot heatmap, impedance, and/or occupancy comparisons for each K (and optional
    freq_*MHz subfolders) using generated samples vs ECADStar Real/ outputs.
Run: python scrap/comparison/compare.py  (set WORKFLOW in CONFIGURATION first)
Inputs / outputs: BASE_GENERATED_DIR / OUTPUT_ROOT from run_all_k or run_multifreq_heatmap_sweep;
    reads configs/{Frequency_data_hz,target_impedance,binary_mask}.npy;
    writes generated_vs_real_*.png per K folder.
Dependencies: matplotlib, numpy, scipy; imports generation config by WORKFLOW.
Agent notes:
    - Type: CLI script with WORKFLOW switch (run_all_k | multifreq_heatmap_sweep)
    - Key symbols: main, _run_single_k, HEATMAP_OUT_NAME, WORKFLOW, _plot_heatmap_comparisons,
    - Config keys: ``WORKFLOW``, ``K_MIN``, ``_freq_src``, ``_FREQ_LIST``, ``_FREQ_LIST``, ``_FREQ_LIST``, ``FREQ_LABEL``, ``RUN_HEATMAP``, ``RUN_IMPEDANCE``, ``RUN_OCCUPANCY``, ``FAIL_FAST``, ``FREQUENCY_PATH``, ``TARGET_IMPEDANCE_PATH``, ``MASK_PATH``, ``HEATMAP_OUT_NAME``, ``IMPEDANCE_OUT_NAME``, ``OCCUPANCY_OUT_NAME``, ``HEATMAP_CMAP``, ``HEATMAP_LEVELS``, ``HEATMAP_DIFF_TOLERANCE``, ``HEATMAP_PATTERN_DIFF_TOLERANCE``, ``HEATMAP_SHARED_COLOR_SCALE``, ``HEATMAP_VMAX_PERCENTILE_FG``, ``MAP_GLOB_PREFERENCE``, ``IMPEDANCE_CSV_GLOB_PREFERENCE``, ``ACTIVE_POLICY``, ``THRESHOLD``
      _plot_impedance_comparisons, _plot_checkboxes
```

## Constants

| Name | Value |
|------|-------|
| `WORKFLOW` | `'multifreq_heatmap_sweep'` |
| `FREQ_LABEL` | `''` |
| `RUN_HEATMAP` | `True` |
| `RUN_IMPEDANCE` | `WORKFLOW != 'multifreq_heatmap_sweep'` |
| `RUN_OCCUPANCY` | `WORKFLOW != 'multifreq_heatmap_sweep'` |
| `FAIL_FAST` | `False` |
| `FREQUENCY_PATH` | `Path('configs/Frequency_data_hz.npy')` |
| `TARGET_IMPEDANCE_PATH` | `Path('configs/target_impedance.npy')` |
| `MASK_PATH` | `Path('configs/binary_mask.npy')` |
| `HEATMAP_OUT_NAME` | `'generated_vs_real_heatmap.png'` |
| `IMPEDANCE_OUT_NAME` | `'generated_vs_real_impedance_profile.png'` |
| `OCCUPANCY_OUT_NAME` | `'generated_occupancy.png'` |
| `HEATMAP_CMAP` | `'jet'` |
| `HEATMAP_LEVELS` | `22` |
| `HEATMAP_DIFF_TOLERANCE` | `0.0` |
| `HEATMAP_PATTERN_DIFF_TOLERANCE` | `0.05` |
| `HEATMAP_DIFF_LEVELS` | `25` |
| `HEATMAP_DIFF_COLORBAR_TICKS` | `25` |
| `HEATMAP_DIFF_CMAP` | `'hot'` |
| `HEATMAP_VMAX_MODE` | `'percentile'` |
| `HEATMAP_SHARED_COLOR_SCALE` | `False` |
| `HEATMAP_REAL_VMAX_MATCH_GENERATED` | `False` |
| `HEATMAP_VMAX_PERCENTILE_FG` | `99.9` |
| `HEATMAP_VMAX_STATS_PERCENTILE` | `99.9` |
| `HEATMAP_COLORBAR_TICKS` | `6` |
| `HEATMAP_PLOT_DPI` | `130` |
| `HEATMAP_INTERPOLATION` | `'bilinear'` |
| `MAP_GLOB_PREFERENCE` | `('Z_*.map', '*.map')` |
| `IMPEDANCE_CSV_GLOB_PREFERENCE` | `('*PIPinZ*.csv', '*.csv')` |
| `ACTIVE_POLICY` | `'topk'` |
| `THRESHOLD` | `0.5` |
| `WRITE_SIM_METRICS` | `True` |
| `METRICS_ONLY` | `False` |

## Functions

- **`_base_dir_for_freq(mhz: int | None)`**
- **`_project_root()`**
- **`_load_heatmap_stats()`**
- **`_resolve_heatmap_vmax_ohm()`** — Shared colorbar vmax in Ω from normalization stats.
- **`_resolve_global_max_ohm()`** — Backward-compatible alias.
- **`_colorbar_ticks(vmin: float, vmax: float, *, n: int | None=None)`**
- **`_infer_num_samples(k_dir: Path)`**
- **`_load_generated_heatmap(path: Path)`**
- **`_load_map_file(file_path: Path, *, resolution: int)`** — Load a real heatmap from a .map file and interpolate to a square grid.
- **`_choose_real_heatmap_mapfile(item: Path)`**
- **`_parse_pi_number(name: str)`**
- **`_real_impedance_candidates(real_dir: Path, sample_i: int)`** — Imp_Real{i}* after rename, or fallback to PI-* folders still in Real/.
- **`_choose_real_impedance_csv(item: Path)`**
- **`_iter_numeric_rows(path: Path)`**
- **`_load_real_impedance(csv_path: Path, *, frequency_hz: np.ndarray)`**
- **`_load_generated_impedance_log(path: Path)`**
- **`_maybe_load(path: Path)`**
- **`_panel_title(ax: Axes, headline: str, detail: str='', *, fontsize: float=11.0)`** — Panel title; optional second line only when detail is provided.
- **`_max_diff_caption(max_diff_ohm: float)`** — Signed peak diff: real_max − gen_max (− → gen higher, + → gen lower).
- **`_plot_heatmap_comparisons(*, comparisons: list[dict], mask: np.ndarray, out_path: Path)`**
- **`_plot_impedance_comparisons(*, frequency: np.ndarray, target_impedance: np.ndarray, comparisons: list[dict], out_path: Path)`**
- **`_load_occupancy_matrix(k_dir: Path, *, num_samples: int)`**
- **`_plot_checkboxes(*, occupancy_matrix: np.ndarray, out_path: Path, title_prefix: str, expected_k: int | None=None)`**
- **`_collect_sim_metrics_single_k(k: int, *, repo_root: Path, mhz: int | None=None)`** — Load gen vs simulated-real heatmaps and return metric rows (no plots).
- **`_run_single_k(k: int, *, repo_root: Path, mhz: int | None=None)`**
- **`_iter_freqs()`**
- **`_refresh_multifreq_config()`** — Re-read K and OUTPUT_ROOT from sweep module (pipeline may have updated them).
- **`_iter_k_values()`**
- **`_metrics_output_root(repo_root: Path)`**
- **`run_sim_metrics_only()`** — Evaluate generated vs ECADStar-simulated real heatmaps; write CSV/JSON/MD only.
- **`main()`**

## Imports

- [[heatmap_sim_metrics]]
- [[repo_paths]]
- [[run_all_k]]
- [[run_multifreq_heatmap_sweep]]

## Imported by

- [[build_comparison_report]]
- [[multifreq_move_and_compare]]
- [[scrap_pipeline]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `scipy`
