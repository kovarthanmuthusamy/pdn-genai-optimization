# Sweep QC Report

- **Experiment:** `experiments/exp050`
- **Checkpoint:** `experiments/exp050/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 20 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.24 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.30 | 0.07 | 0.13 | yes |
| 70 | no | 1.73 | 0.45 | 0.83 | no |
| 120 | no | 2.26 | 0.65 | 1.16 | no |
| 270 | yes | 10.81 | 2.22 | 4.49 | no |
| 400 | yes | 12.15 | 2.41 | 4.68 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 1 | 0.962 | 0.0449 | 0.006 | 0.31 | 0.31 | 0.99 |
| 270 | 1 | 0.633 | 0.3543 | 1.082 | 8.55 | 34.57 | 4.04 |
| 400 | 1 | 0.760 | 1.1513 | 3.940 | 38.07 | 24.26 | 0.64 |

## Flags

- [WARN] 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.962, max_ratio=0.99
- [WARN] 270 MHz anchor: pearson_r=0.633 (<0.75) — weak spatial pattern
- [WARN] 270 MHz anchor: gen_max/real_max=4.04 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp050
checkpoint: experiments/exp050/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 20  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.24

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.30 |    0.07 |    0.13 |     0.21 | yes
    70 | no     |    1.73 |    0.45 |    0.83 |     2.84 | no
   120 | no     |    2.26 |    0.65 |    1.16 |    11.15 | no
   270 | yes    |   10.81 |    2.22 |    4.49 |    28.42 | no
   400 | yes    |   12.15 |    2.41 |    4.68 |    19.74 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 1 | 0.962 | 0.0449 | 0.006 | 0.31 | 0.31 | 0.99
 270 | 1 | 0.633 | 0.3543 | 1.082 | 8.55 | 34.57 | 4.04
 400 | 1 | 0.760 | 1.1513 | 3.940 | 38.07 | 24.26 | 0.64

Flags:
[WARN] 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.962, max_ratio=0.99
[WARN] 270 MHz anchor: pearson_r=0.633 (<0.75) — weak spatial pattern
[WARN] 270 MHz anchor: gen_max/real_max=4.04 — magnitude overshoot
=== END SWEEP QC ===
```
