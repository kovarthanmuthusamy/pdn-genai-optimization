# Full robust normalization — data_multifreq_train

### 📝 Summary of Changes

- Set `OUTPUT_DIR` → `datasets/data_multifreq_train_norm_robust`
- `APPEND=False` — full rebuild (not incremental)
- `USE_ROBUST_PER_MHZ=True` — median/IQR per anchor MHz (default robust mode)
- Restored **160,000** combinations manifest rows after repair truncated the CSV
- Fixed `repair_multifreq_dataset()` to drop stray `None` CSV keys before rewrite
- Started: `python pipelines/normalize/multifreq.py`

### 🚀 Implementation Details

**Input:** `datasets/data_multifreq_train` (471,979 manifest rows after orphan prune)

**Output:** `datasets/data_multifreq_train_norm_robust`

**Pipeline steps:**
1. Repair dataset (remove ~116k orphan heatmaps not in manifest)
2. Compute robust per-MHz stats from all 471,979 raw heatmaps
3. Normalize heatmaps, impedance, copy layouts + PI_freq
4. Write `normalization_stats.json`, `manifest.csv`, `dataset_meta.json`

**Monitor:**
```bash
tail -f logs/normalize_train_robust.log
```

### 🛠️ Verification & Execution Results

Run in progress. Expect several hours for ~472k heatmaps.

When complete, verify:
```bash
ls datasets/data_multifreq_train_norm_robust/
wc -l datasets/data_multifreq_train_norm_robust/manifest.csv
cat datasets/data_multifreq_train_norm_robust/dataset_meta.json
```
