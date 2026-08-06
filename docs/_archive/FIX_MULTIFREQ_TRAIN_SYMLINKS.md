# Fix data_multifreq_train broken symlinks

### 📝 Summary of Changes

- Added `pipelines/data/fix_multifreq_train_symlinks.py`
- Repointed **428,968** broken symlinks (`heatmap` + `PI_freq`) from `/home/ubuntu/gan/datasets/data_multifreq/` to `datasets/data_multifreq/` in this repo

### 🚀 Implementation Details

Original `data_multifreq_train` heatmaps/PI_freq used symlinks to an old path under `/home/ubuntu/gan/`. The real files live in `datasets/data_multifreq/`. The script:

1. Scans `datasets/data_multifreq_train/{heatmap,PI_freq}`
2. Finds broken symlinks with the old `gan` prefix
3. Replaces them with symlinks to the matching file under `datasets/data_multifreq/`
4. Leaves real files (combinations append data) unchanged

Re-run anytime:
```bash
python pipelines/data/fix_multifreq_train_symlinks.py --execute
```

### 🛠️ Verification & Execution Results

```
fixed=428,968  missing_source=0  real_files=748,696 (unchanged)
```

Spot-check: `sample_100.npy` and other original manifest rows now resolve to valid heatmap files.
