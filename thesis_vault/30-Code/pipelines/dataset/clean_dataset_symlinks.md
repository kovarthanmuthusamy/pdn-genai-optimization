---
title: clean_dataset_symlinks
type: code
path: pipelines/dataset/clean_dataset_symlinks.py
group: pipelines/dataset
loc: 170
tags: [code, pipelines]
---

# clean_dataset_symlinks

> Remove broken symlinks and materialize valid ones as real files (no symlinks in datasets).

**Source:** `pipelines/dataset/clean_dataset_symlinks.py` · 170 lines

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `PRIMARY_RAW` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `PRIMARY_NORM` | `REPO_ROOT / 'datasets' / 'data_multifreq_train_norm_unbounded'` |
| `LEGACY_NORM` | `REPO_ROOT / 'datasets' / 'data_multifreq_train_norm_robust'` |
| `BROKEN_STAGING` | `REPO_ROOT / 'datasets' / 'data_multifreq_train_expanded'` |
| `OLD_NORM` | `REPO_ROOT / 'data_multi_norm_robust'` |

## Functions

- **`_is_broken(path: Path)`**
- **`_materialize(path: Path, *, dry_run: bool)`** — Replace symlink with a copy of its target. Returns action label.
- **`clean_tree(root: Path, *, dry_run: bool)`** — Walk bottom-up so directory symlinks are handled after contents.
- **`remove_staging_expanded(*, dry_run: bool)`**
- **`repair_raw(*, dry_run: bool)`**
- **`count_symlinks(root: Path)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`libs`, `repo_paths`, `src_vae`
