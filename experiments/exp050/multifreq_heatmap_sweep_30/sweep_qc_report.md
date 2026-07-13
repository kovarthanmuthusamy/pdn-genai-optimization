# Sweep QC Report

- **Experiment:** `experiments/exp050`
- **Checkpoint:** `experiments/exp050/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 30 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.24 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.23 | 0.06 | 0.10 | yes |
| 70 | no | 1.45 | 0.40 | 0.66 | no |
| 120 | no | 1.87 | 0.53 | 0.88 | no |
| 270 | yes | 7.25 | 1.85 | 3.19 | no |
| 400 | yes | 9.23 | 2.99 | 5.37 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 2 | 0.920 | 0.0496 | 0.007 | 0.25 | 0.24 | 0.98 |
| 270 | 2 | 0.970 | 0.0123 | 0.196 | 10.75 | 10.06 | 0.94 |
| 400 | 2 | 0.589 | 0.5292 | 1.369 | 12.74 | 19.91 | 1.56 |

## Flags

- [WARN] 10 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.920, max_ratio=0.98
- [OK]   270 MHz anchor: pearson_r=0.970, max_ratio=0.94
- [WARN] 400 MHz anchor: pearson_r=0.589 (<0.75) — weak spatial pattern
- [WARN] 400 MHz anchor: gen_max/real_max=1.56 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp050
checkpoint: experiments/exp050/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 30  samples: 2  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.24

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.23 |    0.06 |    0.10 |     0.21 | yes
    70 | no     |    1.45 |    0.40 |    0.66 |     2.84 | no
   120 | no     |    1.87 |    0.53 |    0.88 |    11.15 | no
   270 | yes    |    7.25 |    1.85 |    3.19 |    28.42 | no
   400 | yes    |    9.23 |    2.99 |    5.37 |    19.74 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 2 | 0.920 | 0.0496 | 0.007 | 0.25 | 0.24 | 0.98
 270 | 2 | 0.970 | 0.0123 | 0.196 | 10.75 | 10.06 | 0.94
 400 | 2 | 0.589 | 0.5292 | 1.369 | 12.74 | 19.91 | 1.56

Flags:
[WARN] 10 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.920, max_ratio=0.98
[OK]   270 MHz anchor: pearson_r=0.970, max_ratio=0.94
[WARN] 400 MHz anchor: pearson_r=0.589 (<0.75) — weak spatial pattern
[WARN] 400 MHz anchor: gen_max/real_max=1.56 — magnitude overshoot
=== END SWEEP QC ===
```
