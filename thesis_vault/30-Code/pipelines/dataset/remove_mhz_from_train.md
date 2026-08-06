---
title: remove_mhz_from_train
type: code
path: pipelines/dataset/remove_mhz_from_train.py
group: pipelines/dataset
loc: 137
tags: [code, pipelines]
---

# remove_mhz_from_train

> Remove manifest rows + heatmap/PI_freq files for given MHz values in data_multifreq_train.

**Source:** `pipelines/dataset/remove_mhz_from_train.py` · 137 lines

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `DATA_DIR` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `FREQ_MHZ_LIST` | `[470.0]` |
| `EXECUTE` | `True` |

## Functions

- **`_matches_mhz(row: dict[str, str], freq_mhz: float)`**
- **`remove_freqs(data_dir: Path, freq_list: list[float], *, execute: bool)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`libs`, `repo_paths`, `src_vae`
