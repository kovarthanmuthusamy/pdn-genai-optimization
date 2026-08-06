---
title: ANCHORS_FROM_DATASET_META (archived)
type: archive
source: docs/_archive/ANCHORS_FROM_DATASET_META.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Anchor MHz from dataset_meta.json

### 📝 Summary of Changes

- Anchors are loaded from **`dataset_meta.json`** → `pi_frequencies_mhz` (or `manifest.csv`), not from hand-edited YAML.
- Raw pool `datasets/data_multifreq_train/dataset_meta.json` is the source of truth after appends.
- Normalized training dirs fall back to raw meta when local meta is missing.
- `configs/multifreq_anchors.yaml` is **legacy fallback only**.

### 🚀 Implementation Details

**Resolution order** (`load_anchors_mhz(dataset_dir)`):

1. `{dataset_dir}/dataset_meta.json`
2. `{dataset_dir}/manifest.csv` (`freq_mhz` column)
3. `datasets/data_multifreq_train/dataset_meta.json`
4. YAML → defaults

**After appending data:** refresh raw meta so `pi_frequencies_mhz` matches manifest:

```python
from libs.dataset_meta import write_dataset_meta
write_dataset_meta("datasets/data_multifreq_train", stage="raw", source_script="append")
```

Or re-run your append pipeline (it should call `write_dataset_meta`). Then re-normalize if per-MHz stats changed.

Training rebuilds `multifreq_meta.json` automatically when `anchor_mhz` list changes.

### 🛠️ Verification

```bash
.venv/bin/python -c "from src_vae.others.multifreq_anchors import load_anchors_mhz; print(load_anchors_mhz())"
# → 16 MHz from raw dataset_meta.json
```
