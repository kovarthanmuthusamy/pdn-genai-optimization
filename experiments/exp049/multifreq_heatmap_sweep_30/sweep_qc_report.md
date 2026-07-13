# Sweep QC Report

- **Experiment:** `experiments/exp049`
- **Checkpoint:** `experiments/exp049/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 30 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.24 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.28 | 0.07 | 0.11 | yes |
| 70 | no | 2.06 | 0.47 | 0.76 | no |
| 120 | no | 2.46 | 0.62 | 0.99 | no |
| 270 | yes | 10.37 | 2.12 | 3.69 | no |
| 400 | yes | 66.70 | 3.77 | 6.47 | yes |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 2 | 0.890 | 0.1223 | 0.011 | 0.25 | 0.28 | 1.15 |
| 270 | 2 | 0.960 | 0.0181 | 0.223 | 10.75 | 20.19 | 1.88 |
| 400 | 2 | 0.539 | 0.7667 | 1.844 | 12.74 | 60.06 | 4.71 |

## Flags

- [WARN] 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] 400 MHz: gen_max=66.70Ω (≥90% per-MHz clip ceiling 19.74Ω) — magnitude saturation
- [OK]   10 MHz anchor: pearson_r=0.890, max_ratio=1.15
- [WARN] 270 MHz anchor: gen_max/real_max=1.88 — magnitude overshoot
- [WARN] 400 MHz anchor: pearson_r=0.539 (<0.75) — weak spatial pattern
- [WARN] 400 MHz anchor: gen_max/real_max=4.71 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp049
checkpoint: experiments/exp049/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 30  samples: 2  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.24

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.28 |    0.07 |    0.11 |     0.21 | yes
    70 | no     |    2.06 |    0.47 |    0.76 |     2.84 | no
   120 | no     |    2.46 |    0.62 |    0.99 |    11.15 | no
   270 | yes    |   10.37 |    2.12 |    3.69 |    28.42 | no
   400 | yes    |   66.70 |    3.77 |    6.47 |    19.74 | yes

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 2 | 0.890 | 0.1223 | 0.011 | 0.25 | 0.28 | 1.15
 270 | 2 | 0.960 | 0.0181 | 0.223 | 10.75 | 20.19 | 1.88
 400 | 2 | 0.539 | 0.7667 | 1.844 | 12.74 | 60.06 | 4.71

Flags:
[WARN] 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] 400 MHz: gen_max=66.70Ω (≥90% per-MHz clip ceiling 19.74Ω) — magnitude saturation
[OK]   10 MHz anchor: pearson_r=0.890, max_ratio=1.15
[WARN] 270 MHz anchor: gen_max/real_max=1.88 — magnitude overshoot
[WARN] 400 MHz anchor: pearson_r=0.539 (<0.75) — weak spatial pattern
[WARN] 400 MHz anchor: gen_max/real_max=4.71 — magnitude overshoot
=== END SWEEP QC ===
```
