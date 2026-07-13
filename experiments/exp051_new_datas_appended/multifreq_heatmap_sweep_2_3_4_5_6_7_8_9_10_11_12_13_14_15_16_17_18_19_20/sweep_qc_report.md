# Sweep QC Report

- **Experiment:** `experiments/exp051_new_datas_appended`
- **Checkpoint:** `experiments/exp051_new_datas_appended/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 20 | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 20.12 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.17 | 0.06 | 0.10 | no |
| 70 | no | 1.62 | 0.49 | 0.85 | no |
| 120 | no | 2.22 | 0.72 | 1.31 | no |
| 270 | yes | 12.09 | 2.85 | 6.17 | no |
| 400 | yes | 9.91 | 3.07 | 6.13 | no |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 1 | 0.941 | 0.2507 | 0.018 | 0.21 | 0.21 | 1.00 |
| 270 | 1 | 0.742 | 1.9257 | 5.829 | 14.15 | 28.51 | 2.02 |
| 400 | 1 | 0.803 | 0.6785 | 1.587 | 19.73 | 17.92 | 0.91 |

## Flags

- [OK]   10 MHz anchor: pearson_r=0.941, max_ratio=1.00
- [WARN] 270 MHz anchor: pearson_r=0.742 (<0.75) — weak spatial pattern
- [WARN] 270 MHz anchor: gen_max/real_max=2.02 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp051_new_datas_appended
checkpoint: experiments/exp051_new_datas_appended/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 20  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 20.12

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
    10 | yes    |    0.17 |    0.06 |    0.10 |     0.21 | no
    70 | no     |    1.62 |    0.49 |    0.85 |     2.75 | no
   120 | no     |    2.22 |    0.72 |    1.31 |    10.61 | no
   270 | yes    |   12.09 |    2.85 |    6.17 |    28.51 | no
   400 | yes    |    9.91 |    3.07 |    6.13 |    19.73 | no

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 1 | 0.941 | 0.2507 | 0.018 | 0.21 | 0.21 | 1.00
 270 | 1 | 0.742 | 1.9257 | 5.829 | 14.15 | 28.51 | 2.02
 400 | 1 | 0.803 | 0.6785 | 1.587 | 19.73 | 17.92 | 0.91

Flags:
[OK]   10 MHz anchor: pearson_r=0.941, max_ratio=1.00
[WARN] 270 MHz anchor: pearson_r=0.742 (<0.75) — weak spatial pattern
[WARN] 270 MHz anchor: gen_max/real_max=2.02 — magnitude overshoot
=== END SWEEP QC ===
```
