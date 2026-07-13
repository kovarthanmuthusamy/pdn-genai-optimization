# Sweep QC Report

- **Experiment:** `experiments/exp048`
- **Checkpoint:** `experiments/exp048/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 30 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.24 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.21 | 0.06 | 0.10 | yes |
| 63 | yes | 1.84 | 0.45 | 0.72 | no |
| 300 | yes | 11.60 | 2.29 | 3.65 | no |
| 330 | yes | 16.19 | 2.78 | 4.43 | no |
| 400 | yes | 19.74 | 2.93 | 5.21 | yes |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 2 | 0.952 | 0.0578 | 0.008 | 0.21 | 0.21 | 1.00 |
| 63 | 2 | 0.978 | 0.0381 | 0.059 | 1.68 | 1.88 | 1.12 |
| 300 | 2 | 0.980 | 0.1046 | 0.873 | 10.87 | 21.74 | 2.00 |
| 330 | 2 | 0.987 | 0.0559 | 0.697 | 20.41 | 23.89 | 1.17 |
| 400 | 2 | 0.862 | 0.4193 | 1.202 | 12.74 | 19.74 | 1.55 |

## Flags

- [WARN] 10 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] 400 MHz: gen_max=19.74Ω (≥90% per-MHz clip ceiling 19.74Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.952, max_ratio=1.00
- [OK]   63 MHz anchor: pearson_r=0.978, max_ratio=1.12
- [WARN] 300 MHz anchor: gen_max/real_max=2.00 — magnitude overshoot
- [OK]   330 MHz anchor: pearson_r=0.987, max_ratio=1.17
- [WARN] 400 MHz anchor: gen_max/real_max=1.55 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp048
checkpoint: experiments/exp048/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 30  samples: 2  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.24

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.21 |    0.06 |    0.10 |     0.21 | yes
    63 | yes    |    1.84 |    0.45 |    0.72 |     2.84 | no
   300 | yes    |   11.60 |    2.29 |    3.65 |    28.20 | no
   330 | yes    |   16.19 |    2.78 |    4.43 |    23.89 | no
   400 | yes    |   19.74 |    2.93 |    5.21 |    19.74 | yes

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 2 | 0.952 | 0.0578 | 0.008 | 0.21 | 0.21 | 1.00
  63 | 2 | 0.978 | 0.0381 | 0.059 | 1.68 | 1.88 | 1.12
 300 | 2 | 0.980 | 0.1046 | 0.873 | 10.87 | 21.74 | 2.00
 330 | 2 | 0.987 | 0.0559 | 0.697 | 20.41 | 23.89 | 1.17
 400 | 2 | 0.862 | 0.4193 | 1.202 | 12.74 | 19.74 | 1.55

Flags:
[WARN] 10 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] 400 MHz: gen_max=19.74Ω (≥90% per-MHz clip ceiling 19.74Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.952, max_ratio=1.00
[OK]   63 MHz anchor: pearson_r=0.978, max_ratio=1.12
[WARN] 300 MHz anchor: gen_max/real_max=2.00 — magnitude overshoot
[OK]   330 MHz anchor: pearson_r=0.987, max_ratio=1.17
[WARN] 400 MHz anchor: gen_max/real_max=1.55 — magnitude overshoot
=== END SWEEP QC ===
```
