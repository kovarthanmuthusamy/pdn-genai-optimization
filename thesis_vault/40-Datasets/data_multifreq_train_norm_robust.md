---
title: data_multifreq_train_norm_robust
type: dataset
path: datasets/data_multifreq_train_norm_robust
tags: [dataset]
---

# data_multifreq_train_norm_robust

**Path:** `datasets/data_multifreq_train_norm_robust/` — present on disk (untracked by git)

## dataset_meta.json

| Key | Value |
|-----|-------|
| `schema_version` | `1` |
| `stage` | `normalized` |
| `source_script` | `pipelines/data/refresh_multifreq_train_meta.py` |
| `generated_at_utc` | `2026-06-29T15:03:12.554068+00:00` |
| `dataset_root` | `/home/ubuntu/genai_pdn/datasets/data_multifreq_train_norm_robust` |
| `storage` | `layout_store` |
| `counts` | `{"manifest_rows": 471979, "unique_layouts": 29499, "heatmap_files": 471979, "pi_` |
| `size` | `{"total_bytes": 7914475267, "total_mb": 7547.83, "by_subdirectory_mb": {"heatmap` |
| `pi_frequencies_mhz` | `[10.0, 63.0, 80.0, 200.0, 230.0, 250.0, 270.0, 300.0, 330.0, 350.0, 370.0, 390.0` |
| `samples_per_mhz` | `{"10": 29499, "63": 29499, "80": 29494, "200": 29499, "230": 29499, "250": 29499` |
| `samples_per_freq_label` | `{"10MHz": 29499, "200MHz": 29499, "230MHz": 29499, "250MHz": 29499, "270MHz": 29` |
| `normalization` | `{"stats_file": "normalization_stats.json", "has_heatmap_stats": true}` |
| `mode` | `full` |
| `source_dataset` | `/home/ubuntu/genai_pdn/datasets/data_multifreq_train` |
| `heatmap_norm` | `robust_log1p_per_mhz` |
| `notes` | `["Lightweight metadata touch \u2014 counts from manifest.csv only"]` |

## Referenced by

- `experiments/exp051_new_datas_appended/config.yaml`
