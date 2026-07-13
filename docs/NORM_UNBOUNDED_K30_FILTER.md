# Normalization K≤30 filter (`build_train_norm_unbounded.py`)

### 📝 Summary of Changes

- Added **`MAX_K`** config to `pipelines/normalize/multifreq.py` to drop layouts with decap count **K > MAX_K** during full rebuild and append.
- Set **`norm.MAX_K = 30`** in `pipelines/normalize/build_train_norm_unbounded.py`.
- Filter applies consistently to: robust heatmap stats, heatmap writes, impedance stats/normalize, occupancy copy, PI_freq copy, and **filtered `manifest.csv`**.
- `normalization_stats.json` and `dataset_meta` record `max_k_filter: 30`.

### 🚀 Implementation Details

- **K** is computed per `design_id` from `layouts/{id}/occ.npy` (`(occ > 0.5).sum()`), same as training dataloader / `subsample_inverse_k.py`.
- Manifest rows whose `design_id` exceeds `MAX_K` are excluded before stats and output.
- Stats (median/IQR per MHz, impedance log z-score) are computed **only** on the kept subset so normalization matches the written dataset.
- Output path unchanged: `datasets/data_multifreq_train_norm_unbounded`.

To change the cap, edit `MAX_K` in `build_train_norm_unbounded.py` (or set `norm.MAX_K` when calling `multifreq.main()`). Use `MAX_K = None` for no filter.

### 🛠️ Verification & Execution Results

Rebuild completed successfully:

```text
K filter (MAX_K=30): 24,979 layouts kept, 4,520 layouts dropped
manifest rows: 655,894 → 558,466 (97,428 removed)
dropped K range: 31..50
```

Post-build check on `data_multifreq_train_norm_unbounded`:

- **558,466** manifest rows, **24,979** layouts
- **max K = 30**, **0** rows with K > 30
- `normalization_stats.json` contains `"max_k_filter": 30`

Log: `logs/build_train_norm_unbounded_k30.log`
