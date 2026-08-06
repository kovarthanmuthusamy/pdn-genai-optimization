# Adding PI-Distribution anchor frequencies to the training set

**Recommended path (least effort):** extend `Data_processing_multifreq.py` + `Normalization.py --append`.  
The dataloader reads anchors from `configs/multifreq_anchors.yaml` automatically.

## 1. Edit anchor list

Update `configs/multifreq_anchors.yaml` with your **11** MHz values (7 original + 4 new).  
Raw ECADStar folders must exist as `heatmap_<MHz>/` under your Raw root (e.g. `heatmap_80MHz`).

## 2. Append raw dataset rows

```bash
# Only the four new MHz (example)
python pipelines/data/Data_processing_multifreq.py --append --freqs-mhz 80 150 250 330

# Or process all anchors in YAML (slower; use for full rebuild)
# python pipelines/data/Data_processing_multifreq.py
```

Output: new `sample_*.npy` under `datasets/data_multifreq/` (existing samples kept).

## 3. Append normalization (frozen stats)

```bash
python scripts/Normalization.py --append
```

Uses existing `datasets/data_multifreq_norm/normalization_stats.json` so old and new rows stay comparable.  
Deletes stale `multifreq_meta.json` so training rebuilds metadata.

## 4. Train

Point `data_dir` at `datasets/data_multifreq_norm` (unchanged).  
`ANCHOR_MHZ` in training is loaded from the same YAML.

## Full rebuild (optional)

If you changed anchors globally or want new heatmap stats:

```bash
python pipelines/data/Data_processing_multifreq.py
python scripts/Normalization.py
```
