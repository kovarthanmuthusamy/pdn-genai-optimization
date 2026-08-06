---
title: append_legacy_19k_multifreq
type: code
path: pipelines/heatmaps/append_legacy_19k_multifreq.py
group: pipelines/heatmaps
loc: 493
tags: [code, pipelines, runnable]
---

# append_legacy_19k_multifreq

> Append or replace legacy 19k layout heatmaps from Dataset_19k into data_multifreq_train.

**Source:** `pipelines/heatmaps/append_legacy_19k_multifreq.py` · 493 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/append_legacy_19k_multifreq.py`

## Purpose

```text
Append or replace legacy 19k layout heatmaps from Dataset_19k into data_multifreq_train.

Maps ``all_combinations.csv`` row *i* → ``PI-(i+1)`` under ``heatmap_{MHz}MHz/``.
Uses ``decap_index_map.csv`` for ``design_id`` / ``decap_index``.

Run:
    python pipelines/heatmaps/append_legacy_19k_multifreq.py              # dry-run
    python pipelines/heatmaps/append_legacy_19k_multifreq.py --execute
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `_PEB_19K` | `repo_path('data', 'heatmaps', 'peb_with_19k')` |
| `DECAP_CSV` | `_PEB_19K / 'all_combinations.csv'` |
| `INDEX_MAP` | `_PEB_19K / 'decap_index_map.csv'` |
| `RAW_ROOT_WIN` | `'C:\\Users\\muthusamy\\Desktop\\Dataset_19k'` |
| `OUTPUT_ROOT` | `repo_path('datasets', 'data_multifreq_train')` |
| `APPEND_TAG` | `'legacy_19k_restore'` |
| `SOURCE_FOLDER` | `'layout'` |
| `REPLACE_EXISTING` | `True` |
| `SKIP_IMPEDANCE_IF_MISSING` | `True` |
| `MANIFEST_FLUSH_EVERY` | `5000` |
| `INDEX_THREADS` | `int(os.getenv('INDEX_THREADS', '16'))` |
| `NUM_WORKERS` | `int(os.getenv('NUM_WORKERS', _default_workers))` |
| `RAW_ROOT` | `resolve_windows_path(RAW_ROOT_WIN)` |
| `OUTPUT_ROOT` | `Path(OUTPUT_ROOT)` |

## Functions

- **`load_19k_index_map(path: Path)`**
- **`_heatmap_raw_dir(raw_root: Path, mhz: float)`**
- **`_map_path_for(pi_dir: Path, mhz: float)`**
- **`_index_hm_dir(args: tuple[float, Path])`**
- **`_load_hm_pi(raw_root: Path, mhz_list: list[float], *, threads: int)`**
- **`_existing_manifest_index(output_root: Path)`**
- **`build_19k_tasks(index_entries: list[dict], mhz_list: list[float], hm_pi: dict[float, dict[int, Path]], freq_hz: dict[str, float], existing: dict[tuple[str, str], dict], *, append_tag: str, replace_existing: bool)`**
- **`_process_sample(task: tuple)`**
- **`_append_manifest_rows(output_root: Path, rows: list[dict])`**
- **`_update_manifest_rows(output_root: Path, updated: dict[str, dict])`** — Update existing manifest rows by sample_name (in-place rewrite).
- **`append_legacy_19k_dataset(*, decap_csv: Path, index_map: Path, raw_root: Path, output_root: Path, mhz_list: list[float], num_workers: int=NUM_WORKERS, dry_run: bool=False, replace_existing: bool=REPLACE_EXISTING, append_tag: str=APPEND_TAG)`**
- **`main()`**

## Imports

- [[combinations]]
- [[dataset_meta]]
- [[libs.data_creation.heatmap]]
- [[multifreq_anchors]]
- [[multifreq_layout_store]]
- [[pipelines.data.__init__]]
- [[pipelines.dataset_sim.ecadstar]]
- [[processing_multifreq]]
- [[repo_paths]]

## External dependencies

`concurrent`, `libs`, `multiprocessing`, `numpy`, `repo_paths`, `src_vae`
