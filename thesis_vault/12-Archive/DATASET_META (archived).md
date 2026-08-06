---
title: DATASET_META (archived)
type: archive
source: docs/_archive/DATASET_META.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# `dataset_meta.json`

Each data-processing or normalization run writes a small JSON summary at the dataset root.

## Location

`<dataset_root>/dataset_meta.json` — e.g. `datasets/data_multifreq/dataset_meta.json`.

## Written by

| Script | Stage |
|--------|--------|
| `pipelines/data/processing_multifreq.py` | `raw` |
| `pipelines/data/processing_single.py` | `raw` |
| `pipelines/data/processing_eval.py` | `raw` |
| `pipelines/normalize/multifreq.py` | `normalized` |
| `pipelines/normalize/apply_stats.py` | `normalized` |

Implementation: `libs/dataset_meta.py` (`write_dataset_meta`, `build_dataset_meta`).

## Example fields

```json
{
  "schema_version": 1,
  "stage": "raw",
  "source_script": "pipelines/data/processing_multifreq.py",
  "generated_at_utc": "2026-06-15T12:00:00+00:00",
  "dataset_root": "/path/to/datasets/data_multifreq",
  "storage": "layout_store",
  "counts": {
    "manifest_rows": 12000,
    "unique_layouts": 500,
    "heatmap_files": 12000,
    "pi_freq_files": 12000,
    "layout_directories": 500
  },
  "size": {
    "total_bytes": 524288000,
    "total_mb": 500.0,
    "by_subdirectory_mb": {
      "heatmap": 320.5,
      "layouts": 12.1,
      "PI_freq": 8.2
    }
  },
  "pi_frequencies_mhz": [10, 80, 130, 180, 230, 280, 330, 380, 430, 480, 530, 580, 600],
  "samples_per_mhz": {
    "10": 500,
    "80": 500
  },
  "samples_per_freq_label": {}
}
```

Normalized datasets may also include a `normalization` block when `normalization_stats.json` is present.

## PI frequency sources

1. `manifest.csv` columns `freq_mhz` / `freq_label` (preferred for multifreq layout-store datasets).
2. `normalization_stats.json` → `PI_freq.anchor_mhz` (normalized sets).
3. Fallback: first values from `PI_freq/sample_*.npy` files (Hz → MHz).

## Regenerating without a full rebuild

```bash
cd /path/to/gan
python3 -c "
from pathlib import Path
from libs.dataset_meta import write_dataset_meta
write_dataset_meta('datasets/data_multifreq', stage='raw', source_script='manual')
"
```
