---
title: multifreq_anchors
type: code
path: src_vae/others/multifreq_anchors.py
group: src_vae/others
loc: 135
tags: [code, src_vae]
---

# multifreq_anchors

> Shared PI frequency anchor MHz list and label helpers.

**Source:** `src_vae/others/multifreq_anchors.py` · 135 lines

## Purpose

```text
Shared PI frequency anchor MHz list and label helpers.

Anchor MHz are read from ``dataset_meta.json`` (``pi_frequencies_mhz``) on the
training/normalize dataset dir, with fallback to raw ``datasets/data_multifreq_train``.
``configs/multifreq_anchors.yaml`` is legacy fallback only.

Run: Imported by dataset build, normalization, and training scripts.
```

## Functions

- **`_project_root()`**
- **`_yaml_anchors(path: Path)`**
- **`_resolve_dataset_dir(dataset_dir: str | Path | None)`**
- **`load_anchors_mhz(dataset_dir: str | Path | None=None, *, path: Path | None=None)`** — Load sorted anchor MHz from dataset JSON (preferred) or legacy YAML.
- **`mhz_to_label(mhz: float)`**
- **`mhz_to_heatmap_subdir(mhz: float)`**
- **`anchors_to_freq_labels(anchors: Sequence[float] | None=None, *, dataset_dir: str | Path | None=None)`**
- **`anchors_to_freq_hz(anchors: Sequence[float] | None=None, *, dataset_dir: str | Path | None=None)`**
- **`nearest_anchor_mhz(mhz: float, anchors: Sequence[float] | None=None, *, dataset_dir: str | Path | None=None)`**
- **`anchor_bin_index(mhz: float, anchors: Sequence[float] | None=None, tol_mhz: float=0.5, *, dataset_dir: str | Path | None=None)`** — Index into anchors for exact (within tol) or nearest anchor.

## Imports

- [[dataset_meta]]
- [[repo_paths]]

## Imported by

- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_peb_batch_raw]]
- [[append_restore_49k_legacy_multifreq]]
- [[build_overlay]]
- [[exp041_eval_common]]
- [[exp042_eval_common]]
- [[exp043_eval_common]]
- [[exp044_eval_common]]
- [[exp045_eval_common]]
- [[exp046_eval_common]]
- [[exp047_eval_common]]
- [[exp048_eval_common]]
- [[exp049_eval_common]]
- [[exp050_eval_common]]
- [[exp051_eval_common]]
- [[exp052_eval_common]]
- [[exp053_eval_common]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp039_improved_heatmap.codes.codes.dataloader_multifreq]]
- [[experiments.exp039_improved_heatmap.codes.exp039_eval_common]]
- [[experiments.exp040.codes.codes.dataloader_multifreq]]
- [[experiments.exp040.codes.exp039_eval_common]]
- [[experiments.exp041.codes.codes.dataloader_multifreq]]
- [[experiments.exp043.codes.dataloader_multifreq]]
- [[experiments.exp054_K_30.codes.dataloader_base]]
- [[experiments.exp055_hard_occ.codes.dataloader_base]]
- [[experiments.exp056_graph_vae.codes.dataloader_base]]
- [[experiments.exp057_structured_graph.codes.dataloader_base]]
- [[experiments.exp058_asymmetric_kl.codes.dataloader_base]]
- [[experiments.exp059_capacity_freq.codes.dataloader_base]]
- [[experiments.exp060_multitype_occ.codes.dataloader_base]]
- [[multifreq]]
- [[norm_stats]]
- [[processing_multifreq]]
- [[run_combinations_sim_pipeline]]
- [[run_multitype_sim_pipeline]]

## External dependencies

`libs`, `repo_paths`, `yaml`
