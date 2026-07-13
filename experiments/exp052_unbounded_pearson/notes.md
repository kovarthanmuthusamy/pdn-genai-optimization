# exp052 — unbounded Pearson+grad heatmaps

Fork of exp051 with:

1. **Dataset:** `datasets/data_multifreq_train_norm_unbounded` (`robust_log1p_per_mhz_unbounded`)
2. **Loss:** FG Pearson (global) + Sobel grad (local) + phys top-k blob — no Huber FG, no p99/dynrange
3. **450 epochs** fresh training

Build dataset first:
```bash
python pipelines/normalize/build_train_norm_unbounded.py
```

Train:
```bash
.venv/bin/python -m experiments.exp052_unbounded_pearson.codes.train_vae_simple
```
