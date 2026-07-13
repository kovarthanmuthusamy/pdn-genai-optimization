# Sweep QC Report

- **Experiment:** `experiments/exp054_K_30`
- **Checkpoint:** `experiments/exp054_K_30/checkpoints/checkpoint_epoch_625.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 17 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.29 | 0.07 | 0.14 | yes |
| 70 | no | 1.81 | 0.55 | 1.02 | no |
| 120 | yes | 4.24 | 1.14 | 2.27 | no |
| 270 | yes | 25.93 | 4.72 | 14.64 | no |
| 400 | yes | 10.30 | 2.65 | 4.64 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 1 | 0.987 | 0.0671 | 0.004 | 0.31 | 0.30 | 0.94 |
| 120 | 1 | 0.994 | 0.0030 | 0.032 | 4.28 | 4.33 | 1.01 |
| 270 | 1 | 0.906 | 0.2212 | 1.557 | 16.53 | 31.10 | 1.88 |
| 400 | 1 | 0.875 | 0.2111 | 0.554 | 8.95 | 10.48 | 1.17 |

## Flags

- [WARN] 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.987, max_ratio=0.94
- [OK]   120 MHz anchor: pearson_r=0.994, max_ratio=1.01
- [WARN] 270 MHz anchor: gen_max/real_max=1.88 — magnitude overshoot
- [OK]   400 MHz anchor: pearson_r=0.875, max_ratio=1.17

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp054_K_30
checkpoint: experiments/exp054_K_30/checkpoints/checkpoint_epoch_625.pt
mode: layout_qc  layout_source: val
K: 17  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 21.28

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.29 |    0.07 |    0.14 |     0.21 | yes
    70 | no     |    1.81 |    0.55 |    1.02 |     2.91 | no
   120 | yes    |    4.24 |    1.14 |    2.27 |    12.39 | no
   270 | yes    |   25.93 |    4.72 |   14.64 |    30.06 | no
   400 | yes    |   10.30 |    2.65 |    4.64 |    19.29 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 1 | 0.987 | 0.0671 | 0.004 | 0.31 | 0.30 | 0.94
 120 | 1 | 0.994 | 0.0030 | 0.032 | 4.28 | 4.33 | 1.01
 270 | 1 | 0.906 | 0.2212 | 1.557 | 16.53 | 31.10 | 1.88
 400 | 1 | 0.875 | 0.2111 | 0.554 | 8.95 | 10.48 | 1.17

Flags:
[WARN] 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.987, max_ratio=0.94
[OK]   120 MHz anchor: pearson_r=0.994, max_ratio=1.01
[WARN] 270 MHz anchor: gen_max/real_max=1.88 — magnitude overshoot
[OK]   400 MHz anchor: pearson_r=0.875, max_ratio=1.17
=== END SWEEP QC ===
```
