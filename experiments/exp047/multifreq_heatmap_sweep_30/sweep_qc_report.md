# Sweep QC Report

- **Experiment:** `experiments/exp047`
- **Checkpoint:** `experiments/exp047/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 30 | **PI ref:** 200.0 MHz
- **Clip ceiling (approx):** 33.75 Ω

## Per-MHz generation

| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|-----|------------------|-----------|---------|-----|------------|
| 10 | yes | 0.53 | 0.16 | 0.24 | no |
| 63 | yes | 2.04 | 0.51 | 0.77 | no |
| 300 | yes | 9.75 | 1.98 | 3.10 | no |
| 330 | yes | 18.94 | 2.77 | 4.48 | no |
| 400 | yes | 32.41 | 2.99 | 5.80 | yes |

## Layout vs real (val, training anchors)

| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|-----|---|-----------|--------|----------|----------|---------|-----------|
| 10 | 2 | 0.932 | 0.0198 | 0.097 | 0.25 | 0.51 | 2.10 |
| 63 | 2 | 0.986 | 0.0281 | 0.144 | 1.68 | 2.07 | 1.23 |
| 300 | 2 | 0.988 | 0.0328 | 0.316 | 10.87 | 21.73 | 2.00 |
| 330 | 2 | 0.980 | 0.0623 | 0.686 | 23.02 | 33.75 | 1.47 |
| 400 | 2 | 0.881 | 0.3991 | 1.304 | 12.74 | 33.75 | 2.65 |

## Flags

- [WARN] 400 MHz: gen_max=32.41Ω (≥90% clip ceiling 33.75Ω) — magnitude saturation
- [WARN] 10 MHz anchor: gen_max/real_max=2.10 — magnitude overshoot
- [OK]   63 MHz anchor: pearson_r=0.986, max_ratio=1.23
- [WARN] 300 MHz anchor: gen_max/real_max=2.00 — magnitude overshoot
- [WARN] 400 MHz anchor: gen_max/real_max=2.65 — magnitude overshoot

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp047
checkpoint: experiments/exp047/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 30  samples: 2  pi_ref: 200.0 MHz
z_clip_ceiling_ohm: 33.75

Per-MHz generation (layout decode, no GT):
MHz | anchor | gen_max | mean_fg | p95 | near_clip
    10 | yes    |    0.53 |    0.16 |    0.24 | no
    63 | yes    |    2.04 |    0.51 |    0.77 | no
   300 | yes    |    9.75 |    1.98 |    3.10 | no
   330 | yes    |   18.94 |    2.77 |    4.48 | no
   400 | yes    |   32.41 |    2.99 |    5.80 | yes

Layout vs real val heatmap (training anchors only):
MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  10 | 2 | 0.932 | 0.0198 | 0.097 | 0.25 | 0.51 | 2.10
  63 | 2 | 0.986 | 0.0281 | 0.144 | 1.68 | 2.07 | 1.23
 300 | 2 | 0.988 | 0.0328 | 0.316 | 10.87 | 21.73 | 2.00
 330 | 2 | 0.980 | 0.0623 | 0.686 | 23.02 | 33.75 | 1.47
 400 | 2 | 0.881 | 0.3991 | 1.304 | 12.74 | 33.75 | 2.65

Flags:
[WARN] 400 MHz: gen_max=32.41Ω (≥90% clip ceiling 33.75Ω) — magnitude saturation
[WARN] 10 MHz anchor: gen_max/real_max=2.10 — magnitude overshoot
[OK]   63 MHz anchor: pearson_r=0.986, max_ratio=1.23
[WARN] 300 MHz anchor: gen_max/real_max=2.00 — magnitude overshoot
[WARN] 400 MHz anchor: gen_max/real_max=2.65 — magnitude overshoot
=== END SWEEP QC ===
```
