---
title: multifreq_move_and_compare
type: code
path: scrap/orchestration/multifreq_move_and_compare.py
group: scrap/orchestration
loc: 623
tags: [code, scrap, runnable]
---

# multifreq_move_and_compare

> Multifreq Move, Compare, and Report.

**Source:** `scrap/orchestration/multifreq_move_and_compare.py` · 623 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/orchestration/multifreq_move_and_compare.py`

## Purpose

```text
Multifreq Move, Compare, and Report.

Run: python scrap/multifreq_move_and_compare.py
```

## Constants

| Name | Value |
|------|-------|
| `SOURCE_EMC_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `MOVE` | `True` |
| `CLEAN_DEST_BEFORE_PASTE` | `True` |
| `PI_NAME_REGEX` | `re.compile('^PI-(\\d+)(?:\\..+)?$')` |
| `COPY_PEB_ONLY` | `False` |
| `SKIP_MOVE` | `False` |
| `SKIP_COMPARE` | `False` |
| `SKIP_REPORT` | `False` |
| `METRICS_ONLY` | `False` |
| `RUN_MODE` | `'heatmap'` |
| `RUN_HEATMAP_COMPARE` | `True` |
| `RUN_IMPEDANCE_COMPARE` | `False` |
| `SALVAGE_MISPLACED` | `False` |

## Functions

- **`set_compare_overrides(**kwargs: object)`** — Push heatmap compare settings into compare.py before plotting.
- **`_apply_compare_overrides()`**
- **`_resolve_source_dir(path_str: str)`**
- **`_parse_pi_number(name: str)`**
- **`_suffix(name: str)`**
- **`_extract_map_mhz(folder: Path)`**
- **`_pis_per_sample()`**
- **`_impedance_run()`**
- **`_real_target_stem(i: int)`** — Renamed Real/ item stem compare.py expects for the active run mode.
- **`_k_real_dir(base_generated_dir: Path, mhz: int, k: int)`** — Generated sample Real/ parent for one (MHz, K) slot.
- **`_build_pi_slots(freq_list: list[int], *, k_values: list[int], num_samples: int)`** — Return (target_mhz, k, local_sample_i, global_pi_index) in PEB order.
- **`_rename_heatmap_outputs(dest_dir: Path, *, num_samples: int, sample_offset: int)`** — Rename PI-* in dest_dir for the active run mode (1 PI/sample).
- **`move_sweep_pi_outputs(*, source_emc_dir: str, freq_list: list[int], base_generated_dir: Path, k_values: list[int] | None=None, num_samples: int=NUM_SAMPLES)`** — Move PI-* from ECADStar into freq_*MHz/K{k}/Real/ for this sweep only.
- **`salvage_pi_by_global_index(*, freq_list: list[int], base_generated_dir: Path, k_values: list[int] | None=None, num_samples: int=NUM_SAMPLES)`** — Re-route ``PI-N`` folders using global batch index (fixes wrong K×freq slot maps).
- **`salvage_sweep_heatmap_outputs(*, freq_list: list[int], base_generated_dir: Path, k_values: list[int] | None=None, num_samples: int=NUM_SAMPLES)`** — Re-group misplaced Heatmap_real_* / PI-* by MHz from .map files.
- **`copy_sweep_peb()`** — Copy the multifreq sweep PEB to ``PEB_COPY_DEST`` (Windows PEB folder).
- **`_live_sweep_config()`** — Read K/OUTPUT_ROOT from sweep module after pipeline config apply.
- **`run_move(freq_list: list[int] | None=None)`** — Route PI-* outputs from ECADStar into per-frequency Real/ folders.
- **`run_sim_metrics_only()`** — Simulated-real vs generated metrics only (no comparison PNGs).
- **`run_compare()`** — Generated vs real heatmap and/or impedance plots for each frequency and K.
- **`run_report()`** — Build separate heatmap / impedance comparison HTML + markdown under OUTPUT_ROOT.
- **`main()`**

## Imports

- [[build_comparison_report]]
- [[compare]]
- [[peb_copy]]
- [[repo_paths]]
- [[run_multifreq_heatmap_sweep]]
- [[scrap.comparison.__init__]]

## Imported by

- [[run_multifreq_sweep_pipeline]]

## External dependencies

`repo_paths`
