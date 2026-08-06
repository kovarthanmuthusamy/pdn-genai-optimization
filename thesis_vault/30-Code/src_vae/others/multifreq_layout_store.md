---
title: multifreq_layout_store
type: code
path: src_vae/others/multifreq_layout_store.py
group: src_vae/others
loc: 477
tags: [code, src_vae]
---

# multifreq_layout_store

> Layout-centric on-disk layout for multifreq PI datasets.

**Source:** `src_vae/others/multifreq_layout_store.py` · 477 lines

## Purpose

```text
Layout-centric on-disk layout for multifreq PI datasets.

Purpose: Path helpers, manifest indexing, validation, cache invalidation, and migration
    from legacy per-sample Imp/Occ_map to shared layout folders.
Run: Imported by ``dataloader.py`` and ``datasets/*.py`` maintenance scripts.
Inputs / outputs: Dataset root with ``manifest.csv``, ``layouts/{design_id}/``,
    ``heatmap/``, ``PI_freq/``; returns paths and design_id ↔ sample stem maps.
Dependencies: ``numpy``, ``csv``, ``shutil`` (stdlib).
Agent notes:
    - Type: library module (import-only).
    - Key symbols: ``has_layout_store``, ``load_manifest_index``, ``validate_multifreq_dataset``,
    - Config keys: none (library — import only)
      ``invalidate_training_caches``, ``migrate_to_layout_store``, ``layout_imp_path``, ``layout_occ_path``.
```

## Constants

| Name | Value |
|------|-------|
| `MANIFEST_NAME` | `'manifest.csv'` |
| `LAYOUTS_DIR_NAME` | `'layouts'` |

## Functions

- **`manifest_path(data_dir: Path)`**
- **`layouts_root(data_dir: Path)`**
- **`layout_dir(data_dir: Path, design_id: str)`**
- **`layout_imp_path(data_dir: Path, design_id: str)`**
- **`layout_occ_path(data_dir: Path, design_id: str)`**
- **`has_layout_store(data_dir: Path)`**
- **`load_manifest_index(data_dir: Path)`** — Map heatmap stem (``sample_1``) → ``design_id``.
- **`load_manifest_rows(data_dir: Path)`** — All manifest rows (may include duplicate ``sample_name``).
- **`manifest_design_ids(data_dir: Path)`** — Unique ``design_id`` values referenced in ``manifest.csv``.
- **`manifest_layout_keys(data_dir: Path)`** — Map ``(pi_number, decap_index)`` → ``design_id`` from manifest.
- **`manifest_pi_to_design_id(data_dir: Path)`** — Map ``pi_number`` → ``design_id`` when unique in manifest (fallback for append).
- **`resolve_manifest_design_id(sample: dict, by_pi_decap: dict[tuple[int, int], str], by_pi: dict[int, str])`** — Match a raw sample to an existing manifest ``design_id``.
- **`iter_layout_imp_paths(data_dir: Path)`** — All ``layouts/*/imp.npy`` under ``data_dir``.
- **`iter_layout_occ_paths(data_dir: Path)`**
- **`iter_manifest_layout_imp_paths(data_dir: Path)`** — ``layouts/{design_id}/imp.npy`` for manifest ``design_id`` values only.
- **`prune_orphan_layouts(data_dir: Path, *, allowed_design_ids: set[str] | None=None, dry_run: bool=False)`** — Remove ``layouts/{design_id}/`` folders not referenced in manifest.
- **`filter_dataset_to_design_ids(data_dir: Path, allowed_design_ids: set[str], *, dry_run: bool=False)`** — Keep only rows/files for ``allowed_design_ids`` (manifest, heatmap, PI_freq, layouts).
- **`_heatmap_stems(data_dir: Path)`**
- **`_remove_broken_npy_links(directory: Path)`** — Unlink *.npy symlinks whose target is missing.
- **`repair_multifreq_dataset(data_dir: Path, *, prune_orphan_heatmaps: bool=True, prune_orphan_manifest: bool=True, prune_orphan_pifreq: bool=True, dedupe_manifest: bool=True, dry_run: bool=False)`** — Align ``heatmap/``, ``PI_freq/``, and ``manifest.csv`` (in-place).
- **`validate_multifreq_dataset(data_dir: Path)`** — Raise if heatmap/manifest/layouts/PI_freq are not present for multifreq training.
- **`invalidate_training_caches(data_dir: Path)`**
- **`migrate_to_layout_store(data_dir: Path, *, prune_legacy: bool=False, dry_run: bool=False)`** — Build ``layouts/{design_id}/`` from manifest + per-sample Imp/Occ_map.

## Imports

- [[repo_paths]]

## Imported by

- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_peb_batch_raw]]
- [[append_restore_49k_legacy_multifreq]]
- [[build_overlay]]
- [[clean_dataset_symlinks]]
- [[dataloader]]
- [[dedupe_mhz_manifest]]
- [[delete_combinations_append]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp043.codes.dataloader_multifreq]]
- [[experiments.exp054_K_30.codes.dataloader_base]]
- [[experiments.exp055_hard_occ.codes.dataloader_base]]
- [[experiments.exp056_graph_vae.codes.dataloader_base]]
- [[experiments.exp057_structured_graph.codes.dataloader_base]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.dataloader_base]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.dataloader_base]]
- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.dataloader_base]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[extract_subset]]
- [[finetune_exp057]]
- [[finetune_exp058]]
- [[multifreq]]
- [[pipeline]]
- [[processing_multifreq]]
- [[refresh_multifreq_train_meta]]
- [[remove_mhz_from_train]]
- [[subsample_inverse_k]]
- [[verify_layout_store]]
- [[verify_peb_gmax_match]]

## External dependencies

`numpy`, `repo_paths`
