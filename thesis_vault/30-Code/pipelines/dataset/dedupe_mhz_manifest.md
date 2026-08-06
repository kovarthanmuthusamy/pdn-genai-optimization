---
title: dedupe_mhz_manifest
type: code
path: pipelines/dataset/dedupe_mhz_manifest.py
group: pipelines/dataset
loc: 140
tags: [code, pipelines]
---

# dedupe_mhz_manifest

> Remove duplicate manifest rows for one MHz (keep newest sample_N per design_id).

**Source:** `pipelines/dataset/dedupe_mhz_manifest.py` · 140 lines

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `DATA_DIR` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `FREQ_MHZ` | `470.0` |
| `EXECUTE` | `True` |

## Functions

- **`_sample_index(name: str)`**
- **`_matches_mhz(row: dict[str, str], freq_mhz: float)`**
- **`dedupe_mhz(data_dir: Path, freq_mhz: float, *, execute: bool)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`libs`, `repo_paths`, `src_vae`
