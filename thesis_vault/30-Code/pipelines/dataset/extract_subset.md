---
title: extract_subset
type: code
path: pipelines/dataset/extract_subset.py
group: pipelines/dataset
loc: 380
tags: [code, pipelines, runnable]
---

# extract_subset

> Extract a layout subset from full multifreq into a smaller folder.

**Source:** `pipelines/dataset/extract_subset.py` · 380 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/dataset/extract_subset.py`

## Purpose

```text
Extract a layout subset from full multifreq into a smaller folder.

Purpose:
    Match layouts present in a reference dataset, then copy or symlink all multifreq rows
    (every MHz anchor) for those layouts from source to destination.

Run:
    python pipelines/dataset/extract_subset.py

Agent notes:
    - What: Build a train/eval split folder by layout ID (not by manifest row filter).
    - Usage: Defaults to dry-run. Set ``EXECUTE=True`` to write ``DST``. Use ``OVERWRITE`` or ``RESUME``, not both.
    - Config keys:
        - ``SRC`` — full multifreq source (e.g. ``datasets/data_multifreq``)
        - ``REF`` — reference with ``layouts/`` to match (e.g. ``data_multifreq_norm``)
        - ``DST`` — output folder to create
        - ``EXECUTE`` — ``False`` = preview only; ``True`` = copy/symlink
        - ``HARD_COPY`` — copy files instead of symlinks
        - ``OVERWRITE`` — wipe ``DST`` before extract
        - ``RESUME`` — fill missing files only
    - Key symbol: ``extract_subset``
```

## Constants

| Name | Value |
|------|-------|
| `SRC` | `_ROOT / 'datasets' / 'data_multifreq'` |
| `REF` | `_ROOT / 'datasets' / 'data_multifreq_norm_z_score'` |
| `REF_MATCH` | `'manifest'` |
| `DST` | `_ROOT / 'datasets' / 'data_multifreq_train'` |
| `EXECUTE` | `False` |
| `HARD_COPY` | `False` |
| `OVERWRITE` | `True` |
| `RESUME` | `False` |
| `MANIFEST_FIELDS` | `['sample_name', 'design_id', 'freq_label', 'freq_mhz', 'freq_hz', 'source_folder', 'pi_nu…` |

## Functions

- **`_layout_ids_from_ref(ref: Path, match: str=REF_MATCH)`** — Reference layout IDs for subset extraction.
- **`_load_manifest_rows(path: Path)`**
- **`_link_or_copy(src: Path, dst: Path, *, hard_copy: bool)`**
- **`_link_or_copy_layout_dir(src_sub: Path, dst_sub: Path, *, hard_copy: bool, resume: bool)`**
- **`_prepare_dst(dst: Path, *, overwrite: bool, resume: bool)`** — Return action taken when destination already exists.
- **`extract_subset(*, src: Path, ref: Path, dst: Path, execute: bool, hard_copy: bool, overwrite: bool=False, resume: bool=False, ref_match: str | None=None)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`libs`, `numpy`, `repo_paths`, `src_vae`, `tqdm`
