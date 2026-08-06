---
title: move_and_compare
type: code
path: scrap/comparison/move_and_compare.py
group: scrap/comparison
loc: 1315
tags: [code, scrap, runnable]
---

# move_and_compare

> Move PI Outputs and Compare (run_all_k).

**Source:** `scrap/comparison/move_and_compare.py` · 1315 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/comparison/move_and_compare.py`

## Purpose

```text
Move PI Outputs and Compare (run_all_k).

Run: python scrap/comparison/move_and_compare.py
```

## Constants

| Name | Value |
|------|-------|
| `BASE_GENERATED_DIR` | `_OUTPUT_ROOT` |
| `SOURCE_EMC_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `MOVE` | `True` |
| `OVERWRITE` | `False` |
| `DRY_RUN` | `False` |
| `CLEAN_DEST_BEFORE_PASTE` | `True` |
| `SEARCH_RECURSIVE` | `False` |
| `LIMIT_TO_EXPECTED_PI` | `True` |
| `RENAME_PI_TO_MATCH_SAMPLES` | `True` |
| `RUN_HEATMAP` | `True` |
| `RUN_IMPEDANCE` | `True` |
| `RUN_OCCUPANCY` | `True` |
| `FAIL_FAST` | `False` |
| `FREQUENCY_PATH` | `Path('configs/Frequency_data_hz.npy')` |
| `TARGET_IMPEDANCE_PATH` | `Path('configs/target_impedance.npy')` |
| `MASK_PATH` | `Path('configs/binary_mask.npy')` |
| `HEATMAP_OUT_NAME` | `'generated_vs_real_heatmap.png'` |
| `IMPEDANCE_OUT_NAME` | `'generated_vs_real_impedance_profile.png'` |
| `OCCUPANCY_OUT_NAME` | `'generated_occupancy.png'` |
| `HEATMAP_CMAP` | `'jet'` |
| `HEATMAP_LEVELS` | `22` |
| `HEATMAP_DIFF_TOLERANCE` | `0.25` |
| `HEATMAP_PATTERN_DIFF_TOLERANCE` | `0.05` |
| `HEATMAP_VMAX_PERCENTILE_FG` | `99.9` |
| `MAP_GLOB_PREFERENCE` | `('Z_*.map', '*.map')` |
| `IMPEDANCE_CSV_GLOB_PREFERENCE` | `('*PIPinZ*.csv', '*.csv')` |
| `ACTIVE_POLICY` | `'topk'` |
| `THRESHOLD` | `0.5` |
| `REPORT_ONLY` | `False` |
| `_FREQ_LABEL` | `''` |
| `PI_NAME_REGEX` | `'^PI-\\d+(?:\\..+)?$'` |
| `_CSS` | `'\n* { box-sizing: border-box; margin: 0; padding: 0; }\nhtml, body { height: 100%; overf…` |
| `_JS` | `"\nfunction showTab(idx) {\n document.querySelectorAll('.tab-btn').forEach((b, i) => b.cl…` |
| `_ICON_HEATMAP` | `'<svg viewBox="0 0 24 24"><path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-…` |
| `_ICON_IMPEDANCE` | `'<svg viewBox="0 0 24 24"><polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/></svg>'` |
| `_ICON_OCCUPANCY` | `'<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="…` |

## Functions

- **`_base_dir_for_freq(mhz: int | None)`**
- **`_parse_pi_number(name: str)`**
- **`_suffix(name: str)`**
- **`_infer_num_samples(k_dir: Path)`**
- **`_pis_per_sample()`** — run_all_k always produces 2 PI outputs per sample (heatmap + impedance).
- **`_single_pi_output_kind()`**
- **`_resolve_source_dir(path_str: str)`**
- **`_count_pi_names(directory: Path, *, recursive: bool)`**
- **`resolve_pi_source_dir(path_str: str, *, recursive: bool=True)`**
- **`_iter_candidates(source_dir: Path)`**
- **`_unique_dest_path(dest_dir: Path, name: str)`**
- **`_clean_dest(dest_dir: Path)`**
- **`_do_move_item(item: Path, target: Path)`**
- **`_rename_pi_outputs_with_offset(dest_dir: Path, *, num_samples: int, sample_offset: int)`**
- **`_fallback_rename_pi_to_impedance(dest_dir: Path, num_samples: int)`**
- **`move_pi_outputs_for_k_range(k_min: int, k_max: int, *, source_emc_dir: str=SOURCE_EMC_DIR, base_generated_dir: str | Path=BASE_GENERATED_DIR)`** — Split PI-* outputs across K{k_min}..K{k_max} for a single-frequency run.
- **`move_pi_outputs_multi_freq(*, source_emc_dir: str=SOURCE_EMC_DIR, freq_list: list[int], k_min: int=K_MIN, k_max: int=K_MAX, num_samples: int=_RUN_NUM_SAMPLES, base_generated_dir: str | Path=BASE_GENERATED_DIR)`** — Move PI-* outputs when the PEB was generated with multiple PI frequencies.
- **`_load_generated_heatmap(path: Path)`**
- **`_load_map_file(file_path: Path, *, resolution: int)`**
- **`_choose_real_heatmap_mapfile(item: Path)`**
- **`_real_impedance_candidates(real_dir: Path, sample_i: int)`**
- **`_choose_real_impedance_csv(item: Path)`**
- **`_iter_numeric_rows(path: Path)`**
- **`_load_real_impedance(csv_path: Path, *, frequency_hz: np.ndarray)`**
- **`_load_generated_impedance_log(path: Path)`**
- **`_maybe_load(path: Path)`**
- **`_plot_heatmap_comparisons(*, comparisons: list[dict], mask: np.ndarray, out_path: Path)`**
- **`_plot_impedance_comparisons(*, frequency: np.ndarray, target_impedance: np.ndarray, comparisons: list[dict], out_path: Path)`**
- **`_load_occupancy_matrix(k_dir: Path, *, num_samples: int)`**
- **`_plot_checkboxes(*, occupancy_matrix: np.ndarray, out_path: Path, title_prefix: str, expected_k: int | None=None)`**
- **`_run_compare_single_k(k: int, *, base_dir: Path, repo_root: Path)`**
- **`_run_compare_for_freq(mhz: int | None, *, repo_root: Path)`**
- **`_img_tag(path: Path, alt: str)`**
- **`_copy_report_to_dest(out_path: Path, dest_str: str | None)`**
- **`_build_panel(panel_id: str, tab_idx: int, k_range: range, img_name: str, base_dir: Path, freq_scan: list[tuple[str, str | None]], all_tabs: list[tuple[str, str, str, str]], *, active: bool=False)`**
- **`_write_markdown_summary(*, out_md: Path, base_dir: Path, freq_scan: list[tuple[str, str | None]], k_range: range, title: str, html_name: str)`**
- **`render_comparison_report(*, base_dir: Path, output_html: Path, output_md: Path | None, freq_scan: list[tuple[str, str | None]], k_min: int, k_max: int, tabs: list[tuple[str, str, str, str]], title: str, subtitle: str='', report_copy_dest: str | None=None)`**
- **`build_report(*, report_copy_dest: str | None=None)`** — Build the HTML comparison report for the current run_all_k configuration.
- **`main()`**

## Imports

- [[repo_paths]]
- [[run_all_k]]

## External dependencies

`base64`, `bisect`, `matplotlib`, `numpy`, `repo_paths`, `scipy`
