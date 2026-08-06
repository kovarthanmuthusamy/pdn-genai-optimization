---
title: move_pi_to_real
type: code
path: scrap/orchestration/move_pi_to_real.py
group: scrap/orchestration
loc: 984
tags: [code, scrap, runnable]
---

# move_pi_to_real

> Move ECADStar PI Outputs to Real/ Folders.

**Source:** `scrap/orchestration/move_pi_to_real.py` · 984 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/orchestration/move_pi_to_real.py`

## Purpose

```text
Move ECADStar PI Outputs to Real/ Folders.

Run: python scrap/move_pi_to_real.py  (or import move_pi_outputs_for_k / move_pi_outputs_for_k_range)
```

## Constants

| Name | Value |
|------|-------|
| `WORKFLOW` | `'run_all_k'` |
| `SOURCE_EMC_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `K_VALUE` | `10` |
| `PROCESS_K_RANGE` | `True` |
| `MOVE` | `True` |
| `OVERWRITE` | `False` |
| `DRY_RUN` | `False` |
| `CLEAN_DEST_BEFORE_PASTE` | `True` |
| `SEARCH_RECURSIVE` | `False` |
| `LIMIT_TO_EXPECTED_PI` | `True` |
| `RENAME_PI_TO_MATCH_SAMPLES` | `True` |
| `PI_NAME_REGEX` | `'^PI-\\d+(?:\\..+)?$'` |

## Functions

- **`_base_dir_for_freq(mhz: int | None)`** — Return the effective base directory for a given PI frequency.
- **`_parse_pi_number(name: str)`**
- **`_suffix(name: str)`** — Return extension suffix (including dot) if present, else empty string.
- **`_pis_per_sample()`** — 1 for single PI group per sample; 2 for distribution + spectrum.
- **`_single_pi_output_kind()`** — Rename target for one PI per sample: Heatmap_real_* or Imp_Real*.
- **`_infer_num_samples(generated_k_dir: Path)`** — Infer N from consecutive data_sample_0..data_sample_{N-1} folders.
- **`_rename_pi_outputs(dest_dir: Path, num_samples: int)`** — Rename PI outputs in dest_dir to match sample ordering.
- **`_rename_pi_outputs_with_offset(dest_dir: Path, *, num_samples: int, sample_offset: int)`** — Rename PI outputs when PI numbering is global across a combined PEB.
- **`_count_pi_names(directory: Path, *, recursive: bool)`**
- **`resolve_pi_source_dir(path_str: str, *, recursive: bool=True)`** — Pick the directory that actually contains PI-* outputs (top-level or subfolder).
- **`_resolve_source_dir(path_str: str)`** — Resolve a potentially Windows-style path to a usable Path on this machine.
- **`_iter_candidates(source_dir: Path)`**
- **`_unique_dest_path(dest_dir: Path, name: str)`** — Create a non-colliding destination path by appending _<n> if needed.
- **`move_pi_outputs_for_k(k_value: int, *, source_emc_dir: str=SOURCE_EMC_DIR, base_generated_dir: str | Path=BASE_GENERATED_DIR)`** — Move/copy PI-* items into `.../K{k_value}/Real/` and rename them per sample.
- **`move_pi_outputs_for_k_list(k_values: list[int], *, source_emc_dir: str=SOURCE_EMC_DIR, base_generated_dir: str | Path=BASE_GENERATED_DIR, pis_per_sample: int | None=None, pi_output_kind: str | None=None)`** — Split PI-* outputs across listed K folders (combined-PEB order = sorted K list).
- **`_move_pi_outputs_for_k_list_impl(k_values: list[int], *, source_emc_dir: str, base_generated_dir: str | Path)`**
- **`ensure_real_impedance_renamed(dest_dir: Path, num_samples: int)`** — Ensure PI-Spectrum outputs under Real/ are named Imp_Real{i}* (for compare.py).
- **`_fallback_rename_pi_to_impedance(dest_dir: Path, num_samples: int)`** — If offset-based rename fails, map sorted PI-* → Imp_Real0.. for single-sample K folders.
- **`move_pi_outputs_for_k_range(k_min: int, k_max: int, *, source_emc_dir: str=SOURCE_EMC_DIR, base_generated_dir: str | Path=BASE_GENERATED_DIR)`** — Split PI-* outputs across K{k_min}..K{k_max} assuming one combined-PEB run.
- **`main()`**
- **`_extract_map_mhz(folder: Path)`** — Read simulated MHz from ECADStar Z_*MHz.map under a PI or Heatmap_real folder.
- **`_salvage_item_priority(name: str)`**
- **`salvage_heatmap_real_by_map_mhz(*, freq_list: list[int], k_value: int, num_samples: int, base_generated_dir: str | Path=BASE_GENERATED_DIR)`** — Re-group misplaced Heatmap_real_* / PI-* folders by MHz read from .map files.
- **`move_pi_outputs_multi_freq(*, source_emc_dir: str=SOURCE_EMC_DIR, freq_list: list[int] | None=None, k_min: int=K_MIN, k_max: int=K_MAX, num_samples: int=_RUN_NUM_SAMPLES, base_generated_dir: str | Path=BASE_GENERATED_DIR, pis_per_sample: int | None=None, pi_output_kind: str | None=None)`** — Move PI-* outputs when PEB was generated with multiple PI frequencies.
- **`_move_pi_outputs_multi_freq_impl(*, source_emc_dir: str, freq_list: list[int], k_min: int, k_max: int, num_samples: int, base_generated_dir: str | Path)`**

## Imports

- [[repo_paths]]
- [[run_all_k]]
- [[run_multifreq_heatmap_sweep]]

## Imported by

- [[scrap_pipeline]]

## External dependencies

`bisect`, `repo_paths`
