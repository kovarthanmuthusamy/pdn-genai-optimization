---
title: normalize_labels
type: code
path: active_learning_pi/al/normalize_labels.py
group: active_learning_pi/al
loc: 284
tags: [code, active_learning_pi]
---

# normalize_labels

> Package and normalize ingested ECADSTAR labels for VAE fine-tuning.

**Source:** `active_learning_pi/al/normalize_labels.py` · 284 lines

## Purpose

```text
Package and normalize ingested ECADSTAR labels for VAE fine-tuning.

Run:
    python active_learning_pi/al/normalize_labels.py
```

## Constants

| Name | Value |
|------|-------|
| `EXPECTED_IMP_LEN` | `231` |

## Functions

- **`_load_training_stats(stats_json: Path)`**
- **`denormalize_impedance_from_model(z: np.ndarray, imp_stats: dict)`** — Invert log z-score used in pipelines/normalize/multifreq.py → raw |Z| (231,).
- **`normalize_heatmap_raw(raw: np.ndarray, hm_stats: dict, *, mhz: float | None=None, groot: Path | None=None)`** — Normalize raw heatmap using training stats (robust per-MHz or legacy log-z).
- **`normalize_impedance_raw(raw: np.ndarray, imp_stats: dict)`** — Same as pipelines/normalize/multifreq.py — raw (231,) → (1, 231) log z-score.
- **`_resolve_impedance_csv(sample_dir: Path)`**
- **`package_raw_dataset(labels_dir: Path, raw_root: Path, manifest: list[dict[str, Any]], groot: Path, imp_stats: dict, *, heatmap_only_labels: bool=False, hm_stats: dict | None=None)`** — Layout under raw_root/ (matches Data_Creation multifreq before Normalization.py):
- **`normalize_raw_tree(raw_root: Path, norm_root: Path, stats_json: Path, *, overwrite: bool=False, heatmap_only_labels: bool=False, groot: Path | None=None)`** — Apply training stats (pipelines/normalize/multifreq.py rules) to raw_root → norm_root.
- **`normalize_iteration_labels(cfg: dict, iteration_dir: Path, groot: Path)`** — Package labels/ → raw_dataset/ → dataset_norm/ using training normalization stats.

## Imports

- [[libs.data_creation.impedance]]
- [[robust_normalize]]

## Imported by

- [[build_overlay]]
- [[pipeline]]

## External dependencies

`libs`, `numpy`
