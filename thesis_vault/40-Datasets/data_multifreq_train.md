---
title: data_multifreq_train
type: dataset
path: datasets/data_multifreq_train
tags: [dataset]
---

# data_multifreq_train

**Path:** `datasets/data_multifreq_train/` — present on disk (untracked by git)

## dataset_meta.json

| Key | Value |
|-----|-------|
| `schema_version` | `1` |
| `stage` | `raw` |
| `source_script` | `pipelines/data/append_merged_combinations_multifreq.py` |
| `generated_at_utc` | `2026-08-05T15:24:43.778950+00:00` |
| `dataset_root` | `/home/ubuntu/genai_pdn/datasets/data_multifreq_train` |
| `storage` | `layout_store` |
| `counts` | `{"manifest_rows": 688477, "unique_layouts": 29499, "heatmap_files": 688477, "pi_` |
| `size` | `{"total_bytes": 45410966762, "total_mb": 43307.27, "by_subdirectory_mb": {"heatm` |
| `pi_frequencies_mhz` | `[10.0, 63.0, 80.0, 100.0, 120.0, 150.0, 170.0, 180.0, 200.0, 230.0, 250.0, 270.0` |
| `samples_per_mhz` | `{"10": 29499, "63": 29499, "80": 10000, "100": 29499, "120": 29499, "150": 29499` |
| `samples_per_freq_label` | `{"100MHz": 29499, "10MHz": 29499, "120MHz": 29499, "150MHz": 29499, "170MHz": 29` |

## Referenced by

- `configs/multifreq_anchors.yaml`
- `experiments/impedance_decade_diversity.json`
- `experiments/impedance_freq_diversity.json`
