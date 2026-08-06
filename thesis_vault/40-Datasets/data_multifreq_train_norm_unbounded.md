---
title: data_multifreq_train_norm_unbounded
type: dataset
path: datasets/data_multifreq_train_norm_unbounded
tags: [dataset]
---

# data_multifreq_train_norm_unbounded

**Path:** `datasets/data_multifreq_train_norm_unbounded/` — present on disk (untracked by git)

## dataset_meta.json

| Key | Value |
|-----|-------|
| `schema_version` | `1` |
| `stage` | `normalized` |
| `source_script` | `pipelines/normalize/multifreq.py` |
| `generated_at_utc` | `2026-07-06T13:39:03.768907+00:00` |
| `dataset_root` | `/home/ubuntu/genai_pdn/datasets/data_multifreq_train_norm_unbounded` |
| `storage` | `layout_store` |
| `counts` | `{"manifest_rows": 582997, "unique_layouts": 24979, "heatmap_files": 582997, "pi_` |
| `size` | `{"total_bytes": 9798022468, "total_mb": 9344.12, "by_subdirectory_mb": {"heatmap` |
| `pi_frequencies_mhz` | `[10.0, 63.0, 80.0, 100.0, 120.0, 150.0, 170.0, 180.0, 200.0, 230.0, 250.0, 270.0` |
| `samples_per_mhz` | `{"10": 24979, "63": 24979, "80": 8480, "100": 24979, "120": 24979, "150": 24979,` |
| `samples_per_freq_label` | `{"100MHz": 24979, "10MHz": 24979, "120MHz": 24979, "150MHz": 24979, "170MHz": 24` |
| `normalization` | `{"stats_file": "normalization_stats.json", "has_heatmap_stats": true}` |
| `mode` | `full` |
| `source_dataset` | `/home/ubuntu/genai_pdn/datasets/data_multifreq_train` |
| `heatmap_norm` | `robust_log1p_per_mhz_unbounded` |
| `max_k_filter` | `30` |

## Referenced by

- `active_learning_pi/config/exp057.json`
- `active_learning_pi/config/exp057_gp_error.json`
- `active_learning_pi/config/exp057_random.json`
- `active_learning_pi/config/exp058.json`
- `active_learning_pi/config/exp059.json`
- `active_learning_pi/config/exp059_gp_error.json`
- `active_learning_pi/config/exp059_random.json`
- `experiments/exp052_unbounded_pearson/config.yaml`
- `experiments/exp053_peak_log1p_losses/config.yaml`
- `experiments/exp054_K_30/config.yaml`
- `experiments/exp055_hard_occ/config.yaml`
- `experiments/exp056_graph_vae/config.yaml`
- `experiments/exp057_structured_graph/config.yaml`
- `experiments/exp058_asymmetric_kl/config.yaml`
- `experiments/exp059_capacity_freq/config.yaml`
- `experiments/exp060_multitype_occ/config.yaml`
