# Sweep QC Report

- **Experiment:** `experiments/exp052_unbounded_pearson`
- **Checkpoint:** `experiments/exp052_unbounded_pearson/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 20 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.12 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.18 | 0.06 | 0.10 | no |
| 70 | no | 1.59 | 0.48 | 0.83 | no |
| 120 | no | 2.18 | 0.67 | 1.16 | no |
| 270 | yes | 11.49 | 2.43 | 5.15 | no |
| 400 | yes | 9.68 | 2.94 | 6.30 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 1 | 0.978 | 0.1223 | 0.009 | 0.31 | 0.22 | 0.70 |
| 270 | 1 | 0.815 | 0.4451 | 1.708 | 14.15 | 25.81 | 1.82 |
| 400 | 1 | 0.954 | 0.0970 | 0.567 | 22.92 | 21.50 | 0.94 |

## Flags

- [OK]   10 MHz anchor: pearson_r=0.978, max_ratio=0.70
- [WARN] 270 MHz anchor: gen_max/real_max=1.82 — magnitude overshoot
- [OK]   400 MHz anchor: pearson_r=0.954, max_ratio=0.94

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp052_unbounded_pearson
checkpoint: experiments/exp052_unbounded_pearson/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 20  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.12

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.18 |    0.06 |    0.10 |     0.21 | no
    70 | no     |    1.59 |    0.48 |    0.83 |     2.75 | no
   120 | no     |    2.18 |    0.67 |    1.16 |    10.61 | no
   270 | yes    |   11.49 |    2.43 |    5.15 |    28.51 | no
   400 | yes    |    9.68 |    2.94 |    6.30 |    20.34 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 1 | 0.978 | 0.1223 | 0.009 | 0.31 | 0.22 | 0.70
 270 | 1 | 0.815 | 0.4451 | 1.708 | 14.15 | 25.81 | 1.82
 400 | 1 | 0.954 | 0.0970 | 0.567 | 22.92 | 21.50 | 0.94

Flags:
[OK]   10 MHz anchor: pearson_r=0.978, max_ratio=0.70
[WARN] 270 MHz anchor: gen_max/real_max=1.82 — magnitude overshoot
[OK]   400 MHz anchor: pearson_r=0.954, max_ratio=0.94
=== END SWEEP QC ===
```
