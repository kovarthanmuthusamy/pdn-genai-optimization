---
title: verify_peb_gmax_match
type: code
path: scratch/verify_peb_gmax_match.py
group: scratch
loc: 342
tags: [code, scratch]
---

# verify_peb_gmax_match

> TEMPORARY one-off: verify PEB-batch append + gmax dataset alignment.

**Source:** `scratch/verify_peb_gmax_match.py` · 342 lines

## Purpose

```text
TEMPORARY one-off: verify PEB-batch append + gmax dataset alignment.

Delete when done. Does not modify any pipeline scripts.

Run:
    python scratch/verify_peb_gmax_match.py
    python scratch/verify_peb_gmax_match.py --quick      # counts + occ map only (~30s)
    python scratch/verify_peb_gmax_match.py --raw-spot  # also check 5 Raw PI-N folders (slow on /mnt/c)
```

## Constants

| Name | Value |
|------|-------|
| `TRAIN_DIR` | `ROOT / 'datasets' / 'data_multifreq_train'` |
| `GMAX_DIR` | `ROOT / 'datasets' / 'data_multifreq_gmax'` |
| `REF_DIR` | `ROOT / 'datasets' / 'data_multifreq_norm_z_score'` |
| `MAP_CSV` | `ROOT / 'data' / 'heatmaps' / 'decap_index_map.csv'` |
| `DECAP_CSV` | `ROOT / 'data' / 'heatmaps' / 'all_combinations.csv'` |
| `RAW_DIR` | `Path('/mnt/c/Users/muthusamy/Desktop/Raw')` |
| `NEW_MHZ` | `['350MHz', '370MHz', '390MHz', '420MHz', '450MHz']` |
| `EXPECTED_LAYOUTS` | `19499` |

## Functions

- **`load_manifest(path: Path)`**
- **`load_peb_map(path: Path)`** — batch_pi (1-based) -> {peb_row, design_id, original_decap_index}
- **`section(title: str)`**
- **`check_decap_map_occ(train_dir: Path, decap_csv: Path, map_csv: Path)`**
- **`check_peb_batch_manifest(train_dir: Path, map_csv: Path)`**
- **`check_ref_subset(train_dir: Path, ref_dir: Path)`**
- **`check_train_vs_gmax(train_dir: Path, gmax_dir: Path, *, spot_n: int=20)`**
- **`check_raw_batch_spot(raw_dir: Path, train_dir: Path, map_csv: Path, n: int=5)`**
- **`main()`**

## Imports

- [[heatmap_gmax_norm]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`numpy`, `pandas`, `repo_paths`, `src_vae`
