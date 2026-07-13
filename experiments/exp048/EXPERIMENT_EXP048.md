# exp048 — Robust Per-MHz Scaling + Centralized Denorm

## Summary

**exp048** continues exp047’s layout→heatmap training (distillation, spatial losses, high-freq bias) but switches heatmap normalization to **robust median/IQR per training-anchor MHz** and routes all scale/denorm through a single module.

| Item | exp047 | exp048 |
|------|--------|--------|
| Dataset | `data_multi_norm` | `data_multi_norm_robust` |
| Heatmap norm | global log1p z-score | `robust_log1p_per_mhz` (median/IQR per anchor) |
| Denorm API | inline `log_mean`/`log_std` | `src_vae/others/norm_stats.py` |
| Recon clip in loss | `heatmap_clip_recon: true` | `heatmap_clip_recon: false` |
| Resume | from epoch 400 | **fresh** (`resume_checkpoint: null`) |

## Central norm module

**File:** `src_vae/others/norm_stats.py`

- `load_norm_stats(data_dir)` → `NormStatsBundle` (heatmap + impedance)
- `HeatmapNormStats.norm_to_physical(..., mhz=..., pi_norm=...)` — per-sample MHz-aware denorm
- `HeatmapNormStats.bin_stats(mhz)` — clip bounds, background, median/IQR for one anchor
- `HeatmapNormStats.physical_ceiling_ohm(mhz)` — Ω ceiling at clip max

**Legacy wrapper:** `src_vae/others/heatmap_z_clip.py` delegates to `norm_stats` when `hm_stats=` is passed.

## Build robust dataset

```bash
cd /home/ubuntu/gan
NORM_ROBUST_PER_MHZ=1 python pipelines/normalize/multifreq.py
```

Output: `data_multi_norm_robust/normalization_stats.json` with:

```json
"Heatmap": {
  "norm_mode": "robust_log1p_per_mhz",
  "by_mhz": { "10.0": { "median", "iqr", "clip_min", "clip_max", ... }, ... }
}
```

## Train exp048

```bash
cd /home/ubuntu/gan
python -m experiments.exp048.codes.train_vae_simple
```

Config: `experiments/exp048/config.yaml`

## Scripts wired to centralized stats

| Script | Change |
|--------|--------|
| `experiments/exp048/codes/inference_vae.py` | `load_norm_stats()`, `denorm_heatmap_physical(mhz=...)` |
| `experiments/exp048/codes/eval_real_data_sweep.py` | per-batch denorm via `pi_norm` |
| `experiments/exp048/codes/run_epoch_encode.py` | phys p99 loss uses `_norm_stats` + `pi_freq` |
| `experiments/exp048/codes/train_vae_simple.py` | `_on_stats_loaded` attaches `NormStatsBundle` |
| `src_vae/others/dataloader.py` | per-sample clip from `PI_freq` when robust |
| `scrap/generation/run_multifreq_heatmap_sweep.py` | exp048 + `data_multi_norm_robust` |
| `scrap/orchestration/run_multifreq_sweep_pipeline.py` | exp048 + robust data dir |
| `scrap/generation/sweep_qc_eval.py` | per-MHz denorm + ceiling |

## Sweep / QC

After checkpoint exists:

```bash
python scrap/generation/run_multifreq_heatmap_sweep.py
python scrap/orchestration/run_multifreq_sweep_pipeline.py
```

## Expected benefit

- High MHz (330–400) get their own scale → less global z-score saturation at clip ceiling
- Denorm at inference uses the **decode MHz**, not global mean/std
- Disabling recon clip reduces training-time artificial ceiling on peaks

## Notes

- Impedance stays log z-score (unchanged).
- Widest clip across MHz bins is used for config `heatmap_z_clip_*` during training; per-sample clipping happens in the dataloader.
- exp047 checkpoints are **not** compatible with exp048 data/stats — train fresh.
