---
title: multifreq
type: code
path: pipelines/normalize/multifreq.py
group: pipelines/normalize
loc: 965
tags: [code, pipelines, runnable]
---

# multifreq

> Multifreq dataset normalization pipeline.

**Source:** `pipelines/normalize/multifreq.py` · 965 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/normalize/multifreq.py`

## Purpose

```text
Multifreq dataset normalization pipeline.

Purpose:
    Read raw multifreq dataset; apply log/global-max normalization; write ``data_multifreq_norm``
    with ``normalization_stats.json`` and updated ``dataset_meta.json``.

Run:
    python pipelines/normalize/multifreq.py

Agent notes:
    - What: Produces VAE-ready normalized tensors from ``pipelines/data/processing_multifreq.py`` output.
    - Usage: Set ``DATA_DIR`` (input) and ``OUTPUT_DIR`` → run. ``APPEND=True`` adds new samples only.
    - Config keys:
        - ``DATA_DIR`` — raw multifreq root
        - ``OUTPUT_DIR`` — normalized output; ``None`` defaults beside input
        - ``APPEND`` — incremental normalize vs full rebuild
    - Key symbols: ``normalize_multifreq``, ``prepare_output_dir``
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `OVERWRITE_OUTPUT` | `os.getenv('DATA_OVERWRITE', '1').strip().lower() not in ('0', 'false', 'no')` |
| `DATA_DIR` | `_REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `USE_GLOBAL_MAX_HEATMAP` | `False` |
| `USE_ROBUST_PER_MHZ` | `os.getenv('NORM_ROBUST_PER_MHZ', '1').strip().lower() in ('1', 'true', 'yes')` |
| `USE_UNBOUNDED_Z` | `os.getenv('NORM_UNBOUNDED_Z', '0').strip().lower() in ('1', 'true', 'yes')` |
| `ROBUST_IQR_EPS` | `0.0001` |
| `ROBUST_CLIP_PERCENTILE_LOWER` | `0.5` |
| `ROBUST_CLIP_PERCENTILE_UPPER` | `99.5` |
| `GMAX_PERCENTILE` | `99.5` |
| `BG_OHM` | `0.05` |
| `CLIP_MAX` | `1.02` |
| `APPEND` | `False` |
| `REPAIR_DATASET` | `True` |
| `MANIFEST_LAYOUTS_ONLY` | `True` |
| `PRUNE_ORPHAN_LAYOUTS` | `True` |

## Functions

- **`_k_from_occ(data_root: Path, design_id: str)`**
- **`load_design_k(data_root: Path)`** — design_id → decap count K (from shared occupancy).
- **`_allowed_design_ids(data_root: Path, max_k: int)`**
- **`_filter_heatmap_paths(heatmap_dir: Path, manifest_index: dict[str, str], allowed_design_ids: set[str])`**
- **`_filter_manifest_rows(rows: list[dict], allowed_design_ids: set[str])`**
- **`_print_k_filter_summary(design_k: dict[str, int], allowed_design_ids: set[str], *, max_k: int, manifest_rows_before: int, manifest_rows_after: int)`**
- **`_maybe_repair_dataset(data_root: Path)`**
- **`_is_gmax_heatmap_stats(heatmap_stats: dict)`**
- **`_output_dir()`**
- **`_raw_heatmap_plane_mask(data: np.ndarray)`**
- **`prepare_output_dir(output_root: Path, *, overwrite: bool=OVERWRITE_OUTPUT)`**
- **`calculate_heatmap_stats_gmax(heatmap_dir: Path, *, global_max_ohm: float | None=None, gmax_percentile: float=GMAX_PERCENTILE, bg_ohm: float=BG_OHM, clip_max: float=CLIP_MAX, heatmap_files: list[Path] | None=None)`** — Scan raw physical Ω heatmaps; return global-max stats.
- **`_load_stem_to_mhz(data_root: Path)`** — Map heatmap stem → freq_mhz from manifest.csv.
- **`_is_robust_per_mhz_stats(heatmap_stats: dict)`**
- **`_is_unbounded_heatmap_stats(heatmap_stats: dict)`**
- **`_mhz_key(mhz: float)`**
- **`calculate_heatmap_stats_robust_per_mhz(heatmap_dir: Path, data_root: Path, *, percentile_lower: float=ROBUST_CLIP_PERCENTILE_LOWER, percentile_upper: float=ROBUST_CLIP_PERCENTILE_UPPER, iqr_eps: float=ROBUST_IQR_EPS, heatmap_files: list[Path] | None=None)`** — Robust median/IQR stats per training-anchor MHz.
- **`_normalize_raw_heatmap_to_array(data: np.ndarray, heatmap_stats: dict, *, mhz: float | None=None)`** — Normalize one raw (2,H,W) heatmap to saved (1,H,W) tensor.
- **`calculate_heatmap_stats(heatmap_dir: Path, percentile_lower=0.1, percentile_upper=99.9, heatmap_files: list[Path] | None=None)`**
- **`normalize_heatmaps(heatmap_dir: Path, output_heatmap_dir: Path, heatmap_stats: dict, *, stem_mhz: dict[str, float] | None=None, heatmap_files: list[Path] | None=None)`**
- **`calculate_impedance_stats(data_root: Path, *, allowed_design_ids: set[str] | None=None)`**
- **`normalize_layout_impedance(data_root: Path, output_root: Path, imp_stats: dict, *, allowed_design_ids: set[str] | None=None)`**
- **`copy_layout_occupancy(data_root: Path, output_root: Path, *, allowed_design_ids: set[str] | None=None)`**
- **`validate_pifreq(pifreq_dir: Path, expected_mhz=None)`**
- **`copy_pifreq(pifreq_dir: Path, output_pifreq_dir: Path, *, allowed_stems: set[str] | None=None)`**
- **`_missing_heatmap_names(data_root: Path, output_root: Path)`**
- **`normalize_append(data_root: Path, output_root: Path)`** — Normalize new heatmap rows; copy/normalize new layouts from raw layouts/.
- **`run_full_pipeline(data_root: Path, output_root: Path)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[heatmap_gmax_norm]]
- [[multifreq_anchors]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## Imported by

- [[build_train_norm_unbounded]]
- [[robust_normalize]]

## External dependencies

`libs`, `numpy`, `repo_paths`, `src_vae`, `tqdm`
