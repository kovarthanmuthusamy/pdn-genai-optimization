# Sweep QC Report

- **Experiment:** `experiments/exp053_peak_log1p_losses`
- **Checkpoint:** `experiments/exp053_peak_log1p_losses/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 17 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.12 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.22 | 0.07 | 0.14 | yes |
| 70 | no | 2.23 | 0.60 | 1.27 | no |
| 120 | yes | 4.81 | 1.18 | 2.59 | no |
| 270 | yes | 20.56 | 3.00 | 9.71 | no |
| 400 | yes | 10.63 | 3.06 | 6.13 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 1 | 0.972 | 0.1001 | 0.007 | 0.28 | 0.20 | 0.72 |
| 120 | 1 | 0.996 | 0.0031 | 0.032 | 4.74 | 4.82 | 1.02 |
| 270 | 1 | 0.951 | 0.1648 | 1.107 | 31.75 | 33.43 | 1.05 |
| 400 | 1 | 0.941 | 0.0660 | 0.260 | 7.77 | 8.87 | 1.14 |

## Flags

- [WARN] 10 MHz: gen_max=0.22Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.972, max_ratio=0.72
- [OK]   120 MHz anchor: pearson_r=0.996, max_ratio=1.02
- [OK]   270 MHz anchor: pearson_r=0.951, max_ratio=1.05
- [OK]   400 MHz anchor: pearson_r=0.941, max_ratio=1.14

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp053_peak_log1p_losses
checkpoint: experiments/exp053_peak_log1p_losses/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 17  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.12

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.22 |    0.07 |    0.14 |     0.21 | yes
    70 | no     |    2.23 |    0.60 |    1.27 |     2.75 | no
   120 | yes    |    4.81 |    1.18 |    2.59 |    11.78 | no
   270 | yes    |   20.56 |    3.00 |    9.71 |    28.51 | no
   400 | yes    |   10.63 |    3.06 |    6.13 |    16.91 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 1 | 0.972 | 0.1001 | 0.007 | 0.28 | 0.20 | 0.72
 120 | 1 | 0.996 | 0.0031 | 0.032 | 4.74 | 4.82 | 1.02
 270 | 1 | 0.951 | 0.1648 | 1.107 | 31.75 | 33.43 | 1.05
 400 | 1 | 0.941 | 0.0660 | 0.260 | 7.77 | 8.87 | 1.14

Flags:
[WARN] 10 MHz: gen_max=0.22Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.972, max_ratio=0.72
[OK]   120 MHz anchor: pearson_r=0.996, max_ratio=1.02
[OK]   270 MHz anchor: pearson_r=0.951, max_ratio=1.05
[OK]   400 MHz anchor: pearson_r=0.941, max_ratio=1.14
=== END SWEEP QC ===
```
