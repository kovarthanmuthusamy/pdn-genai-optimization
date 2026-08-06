---
title: dataset_meta
type: code
path: libs/dataset_meta.py
group: libs
loc: 279
tags: [code, libs]
---

# dataset_meta

> Write ``dataset_meta.json`` summarizing an on-disk training dataset.

**Source:** `libs/dataset_meta.py` · 279 lines

## Constants

| Name | Value |
|------|-------|
| `META_FILENAME` | `'dataset_meta.json'` |
| `SCHEMA_VERSION` | `1` |
| `RAW_MULTIFREQ_TRAIN` | `'datasets/data_multifreq_train'` |

## Functions

- **`_dir_size_bytes(path: Path)`**
- **`_dataset_size_bytes(root: Path, subdirs: list[str])`** — Sum known data subdirs plus loose files at dataset root.
- **`_mb(n_bytes: int)`**
- **`_read_manifest_rows(manifest: Path)`**
- **`_subdir_sizes_mb(root: Path, names: list[str])`**
- **`_freq_summary_from_manifest(rows: list[dict[str, str]])`**
- **`_count_npy(dir_path: Path)`**
- **`_count_layout_dirs(layouts: Path)`**
- **`build_dataset_meta(data_root: Path | str, *, stage: str, source_script: str, extra: dict[str, Any] | None=None)`** — Build metadata dict for ``data_root`` (raw multifreq, legacy single-freq, or normalized).
- **`write_dataset_meta(data_root: Path | str, *, stage: str, source_script: str, extra: dict[str, Any] | None=None, filename: str=META_FILENAME)`** — Write ``dataset_meta.json`` under ``data_root``; return path.
- **`read_pi_frequencies_mhz(data_root: Path | str, *, fallback_raw: bool=True)`** — Read anchor MHz from ``dataset_meta.json`` (preferred) or ``manifest.csv``.
- **`_pi_frequencies_from_root(root: Path)`**

## Imports

- [[repo_paths]]

## Imported by

- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_peb_batch_raw]]
- [[append_restore_49k_legacy_multifreq]]
- [[apply_stats]]
- [[clean_dataset_symlinks]]
- [[copy_multifreq_k_subset]]
- [[dedupe_mhz_manifest]]
- [[extract_subset]]
- [[multifreq]]
- [[multifreq_anchors]]
- [[processing_eval]]
- [[processing_multifreq]]
- [[remove_mhz_from_train]]
- [[run_append_locked]]
- [[run_append_worker]]
- [[smoke_dataset_meta]]

## External dependencies

`numpy`, `repo_paths`
