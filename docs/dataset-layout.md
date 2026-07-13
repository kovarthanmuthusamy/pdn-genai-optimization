# Dataset layout (genai_pdn)

## Primary datasets (use these)

| Path | Role |
|------|------|
| `datasets/data_multifreq_train` | **Raw** — `heatmap/`, `PI_freq/`, `layouts/`, `manifest.csv` |
| `datasets/data_multifreq_train_norm_unbounded` | **Normalized for exp053** — run `build_train_norm_unbounded.py` |

## Optional / legacy

| Path | Role |
|------|------|
| `datasets/data_multifreq_train_norm_robust` | Bounded robust norm (exp052-era) |
| `data_multi_norm_robust` | Older backup at repo root |

## Removed

- `datasets/data_multifreq_train_expanded` — broken symlink staging (deleted)
- `datasets/data_multifreq` — deleted source of old layout symlinks

## Layout store

Real files only (no symlinks): `layouts/{design_id}/imp.npy`, `occ.npy`

- Legacy 19,499: `pipelines/heatmaps/fill_legacy_layouts_19k_imp.py`
- Combinations 10,000: merged append

## Maintenance

```bash
python pipelines/dataset/clean_dataset_symlinks.py --execute
python pipelines/dataset/clean_train_metadata.py --execute
python pipelines/normalize/build_train_norm_unbounded.py
```

### Raw root files (keep only)

| File | Purpose |
|------|---------|
| `manifest.csv` | Sample index (required) |
| `dataset_meta.json` | Anchor MHz + counts |

Removed by cleanup: `append_*`, `*_progress.json`, `append_batches/`, `manifest.csv.bak_*`
