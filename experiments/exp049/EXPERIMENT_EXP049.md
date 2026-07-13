# exp049 — Unbounded Robust Per-MHz Log Z-Score

## Summary

**exp049** builds on exp048 with **unbounded foreground z** — robust median/IQR per anchor MHz **without** hard clip at normalize, load, or denorm. Preserves peak spatial structure for grad/peak_loc/pearson losses while keeping per-MHz scale.

| Item | exp048 | exp049 |
|------|--------|--------|
| Dataset | `data_multi_norm_robust` | `data_multi_norm_unbounded` |
| `norm_mode` | `robust_log1p_per_mhz` | `robust_log1p_per_mhz_unbounded` |
| Foreground clip | p99.5 hard clip | **none** |
| `heatmap_z_clip_*` | widest bin | **null** |
| Phys p99 loss | under-predict only | under + **overshoot** (`over_weight=0.75`) |

## Build dataset

```bash
cd /home/ubuntu/gan
NORM_ROBUST_PER_MHZ=1 NORM_UNBOUNDED_Z=1 .venv/bin/python pipelines/normalize/multifreq.py
```

Output: `data_multi_norm_unbounded/normalization_stats.json` with `"unbounded": true` and `clip_min/max` kept as **QC metadata only** (p0.5/p99.5 reference).

## Train

```bash
cd /home/ubuntu/gan
.venv/bin/python -m experiments.exp049.codes.train_vae_simple
```

Config: `experiments/exp049/config.yaml`

## Central norm (`src_vae/others/norm_stats.py`)

- `HeatmapNormStats.is_unbounded()` → no clip in denorm/dataloader
- `physical_ceiling_ohm_soft(mhz)` → p99.5 reference for sweep QC flags
- `physical_ceiling_ohm(mhz)` → uses `z_max` when unbounded (data tail)

## Why this helps spatial pattern

Clipped targets flatten peak regions → zero grad/lap signal, ambiguous peak_loc. Unbounded z keeps relative ordering in the hotspot so layout/encode paths can learn **where** peaks are, not just “raise to ceiling”.

## Files changed

- `pipelines/normalize/multifreq.py` — `USE_UNBOUNDED_Z`, `NORM_UNBOUNDED_Z=1`
- `src_vae/others/norm_stats.py` — unbounded mode
- `src_vae/others/dataloader.py` — skip clip on load
- `experiments/exp049/` — training, inference, eval
- Sweep scripts → exp049 + `data_multi_norm_unbounded`

## Notes

- exp048/exp047 checkpoints **not** compatible — train fresh.
- Rare extreme outliers (80+ Ω) remain in z-space; fg-masked losses + overshoot penalty mitigate blow-up.
- Sweep `near_clip` uses **soft p99.5 reference**, not a hard training bound.
