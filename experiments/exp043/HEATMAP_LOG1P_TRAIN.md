# log1p(phys/gmax) train space

On-disk heatmaps stay **linear** `phys/gmax` in `data_multifreq_gmax`. Training and decode use **log1p remap** on the fly:

```
t = log1p(n × gmax) / log1p(gmax)     # train space ∈ [0, 1]
n = expm1(t × log1p(gmax)) / gmax     # back to disk linear
phys = expm1(t × log1p(gmax))         # direct to Ω
```

## Why

Linear `/gmax` crushes low MHz values (skew ≈ 4.8; 41% of FG in norm [0.02, 0.05)). Log1p spreads the low end (skew ≈ 0.7; p90/p10 ≈ 5.8× vs 14.5×).

## Config

```json
"heatmap_train_space": "log1p_gmax",
"heatmap_fg_threshold": null
```

`null` → dataset `fg_norm_threshold` (0.000919 norm ≈ 0.05 Ω), mapped into train space automatically.

## Code touchpoints

| File | Role |
|------|------|
| `src_vae/others/heatmap_gmax_norm.py` | `disk_to_train_space`, `train_to_disk_space`, phys loss |
| `gmax_training_patch.py` | Batch prep, `apply_gmax_config` clip/threshold in train space |
| `eval_cross_freq_gmax.py` | Off-anchor MSE in train space |
| `inference_vae.py` | `heatmap_train_to_disk()` for saved `.npy` |

## Fresh restart required

Incompatible with linear-space checkpoints. Decoder `_hm_out_scale` uses train-space `clip_max` (~0.998 for log1p).

```bash
.venv/bin/python scratch/check_log1p_train_space.py
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```

## Sweep / compare

Generated `.npy` saved via `heatmap_train_to_disk()` are linear norm (compatible with `compare.py` and Real/).
