# Sweep QC Report

- **Experiment:** `experiments/exp057_structured_graph`
- **Checkpoint:** `experiments/exp057_structured_graph/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 3–10 (8 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 3 | 10 | yes | 0.35 | 0.09 | 0.18 | yes |
| 3 | 63 | yes | 3.19 | 1.34 | 2.10 | yes |
| 3 | 150 | yes | 8.69 | 2.68 | 6.17 | no |
| 3 | 270 | yes | 36.32 | 7.78 | 22.97 | yes |
| 3 | 450 | yes | 10.65 | 2.47 | 5.34 | no |
| 3 | 500 | yes | 16.71 | 4.12 | 8.98 | yes |
| 4 | 10 | yes | 0.32 | 0.08 | 0.17 | yes |
| 4 | 63 | yes | 2.38 | 0.88 | 1.48 | no |
| 4 | 150 | yes | 7.46 | 2.02 | 4.72 | no |
| 4 | 270 | yes | 54.27 | 9.52 | 36.92 | yes |
| 4 | 450 | yes | 13.57 | 3.27 | 7.39 | no |
| 4 | 500 | yes | 19.51 | 4.86 | 11.43 | yes |
| 5 | 10 | yes | 0.34 | 0.09 | 0.17 | yes |
| 5 | 63 | yes | 2.59 | 0.88 | 1.56 | no |
| 5 | 150 | yes | 6.13 | 1.50 | 3.45 | no |
| 5 | 270 | yes | 22.14 | 3.84 | 11.27 | no |
| 5 | 450 | yes | 10.41 | 2.44 | 5.74 | no |
| 5 | 500 | yes | 16.05 | 4.38 | 10.30 | yes |
| 6 | 10 | yes | 0.32 | 0.08 | 0.16 | yes |
| 6 | 63 | yes | 2.19 | 0.68 | 1.23 | no |
| 6 | 150 | yes | 5.14 | 1.26 | 2.82 | no |
| 6 | 270 | yes | 18.70 | 3.08 | 9.17 | no |
| 6 | 450 | yes | 10.92 | 2.57 | 5.52 | no |
| 6 | 500 | yes | 15.51 | 3.76 | 7.84 | yes |
| 7 | 10 | yes | 0.34 | 0.07 | 0.15 | yes |
| 7 | 63 | yes | 2.32 | 0.65 | 1.16 | no |
| 7 | 150 | yes | 4.33 | 0.91 | 2.00 | no |
| 7 | 270 | yes | 17.50 | 3.42 | 9.71 | no |
| 7 | 450 | yes | 11.50 | 2.17 | 5.03 | no |
| 7 | 500 | yes | 14.64 | 2.81 | 6.33 | no |
| 8 | 10 | yes | 0.34 | 0.08 | 0.16 | yes |
| 8 | 63 | yes | 2.17 | 0.62 | 1.15 | no |
| 8 | 150 | yes | 4.79 | 0.99 | 2.30 | no |
| 8 | 270 | yes | 18.24 | 2.63 | 8.25 | no |
| 8 | 450 | yes | 10.40 | 2.14 | 4.90 | no |
| 8 | 500 | yes | 13.83 | 2.87 | 6.34 | no |
| 9 | 10 | yes | 0.31 | 0.08 | 0.16 | yes |
| 9 | 63 | yes | 2.12 | 0.62 | 1.13 | no |
| 9 | 150 | yes | 4.56 | 0.94 | 2.30 | no |
| 9 | 270 | yes | 25.38 | 4.54 | 13.42 | no |
| 9 | 450 | yes | 8.84 | 2.00 | 4.73 | no |
| 9 | 500 | yes | 12.85 | 2.98 | 6.98 | no |
| 10 | 10 | yes | 0.33 | 0.08 | 0.15 | yes |
| 10 | 63 | yes | 2.14 | 0.57 | 1.08 | no |
| 10 | 150 | yes | 6.47 | 1.70 | 3.88 | no |
| 10 | 270 | yes | 22.07 | 4.67 | 16.64 | no |
| 10 | 450 | yes | 10.80 | 2.70 | 5.17 | no |
| 10 | 500 | yes | 12.75 | 3.16 | 6.58 | no |

## Layout vs real (val, training anchors)

| K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|---|-----|---|-----------|--------|----------|----------|---------|-----------|
| 3 | 10 | 1 | 0.984 | 0.0818 | 0.005 | 0.36 | 0.35 | 0.99 |
| 3 | 63 | 1 | 0.998 | 0.0405 | 0.024 | 2.91 | 2.98 | 1.03 |
| 3 | 150 | 1 | 0.995 | 0.0088 | 0.064 | 4.81 | 4.93 | 1.03 |
| 3 | 270 | 1 | 0.962 | 0.2307 | 2.509 | 33.68 | 23.33 | 0.69 |
| 3 | 450 | 1 | 0.993 | 0.0098 | 0.166 | 10.84 | 12.37 | 1.14 |
| 3 | 500 | 1 | 0.935 | 0.1798 | 1.341 | 18.84 | 12.86 | 0.68 |
| 4 | 10 | 1 | 0.975 | 0.1006 | 0.007 | 0.34 | 0.33 | 0.97 |
| 4 | 63 | 1 | 0.997 | 0.0406 | 0.020 | 2.47 | 2.49 | 1.01 |
| 4 | 150 | 1 | 0.971 | 0.0350 | 0.092 | 4.42 | 4.35 | 0.99 |
| 4 | 270 | 1 | 0.968 | 0.0658 | 0.723 | 16.91 | 14.45 | 0.85 |
| 4 | 450 | 1 | 0.992 | 0.0084 | 0.119 | 10.58 | 10.69 | 1.01 |
| 4 | 500 | 1 | 0.986 | 0.0139 | 0.221 | 11.51 | 10.99 | 0.96 |
| 5 | 10 | 1 | 0.992 | 0.0709 | 0.005 | 0.36 | 0.35 | 0.96 |
| 5 | 63 | 1 | 0.997 | 0.0529 | 0.052 | 2.66 | 2.61 | 0.98 |
| 5 | 150 | 1 | 0.965 | 0.0354 | 0.111 | 4.14 | 3.78 | 0.91 |
| 5 | 270 | 1 | 0.990 | 0.0088 | 0.150 | 10.76 | 11.14 | 1.04 |
| 5 | 450 | 1 | 0.995 | 0.0051 | 0.086 | 9.24 | 10.66 | 1.15 |
| 5 | 500 | 1 | 0.989 | 0.0219 | 0.487 | 19.84 | 16.19 | 0.82 |
| 6 | 10 | 1 | 0.984 | 0.0768 | 0.006 | 0.34 | 0.34 | 1.00 |
| 6 | 63 | 1 | 0.996 | 0.0442 | 0.032 | 2.51 | 2.50 | 1.00 |
| 6 | 150 | 1 | 0.931 | 1.9073 | 3.122 | 9.59 | 18.31 | 1.91 |
| 6 | 270 | 1 | 0.948 | 0.3343 | 2.999 | 34.37 | 19.22 | 0.56 |
| 6 | 450 | 1 | 0.973 | 0.0207 | 0.171 | 8.95 | 9.75 | 1.09 |
| 6 | 500 | 1 | 0.925 | 0.1693 | 1.464 | 16.75 | 23.43 | 1.40 |
| 7 | 10 | 1 | 0.990 | 0.0659 | 0.003 | 0.33 | 0.32 | 0.99 |
| 7 | 63 | 1 | 0.995 | 0.0376 | 0.008 | 2.35 | 2.33 | 0.99 |
| 7 | 150 | 1 | 0.973 | 0.1226 | 0.736 | 13.61 | 13.26 | 0.97 |
| 7 | 270 | 1 | 0.992 | 0.0216 | 0.477 | 21.00 | 22.95 | 1.09 |
| 7 | 450 | 1 | 0.982 | 0.0139 | 0.133 | 8.57 | 9.10 | 1.06 |
| 7 | 500 | 1 | 0.957 | 0.0468 | 0.479 | 13.78 | 11.03 | 0.80 |
| 8 | 10 | 1 | 0.997 | 0.0585 | 0.002 | 0.35 | 0.34 | 0.97 |
| 8 | 63 | 1 | 0.996 | 0.0385 | 0.011 | 2.31 | 2.16 | 0.94 |
| 8 | 150 | 1 | 0.984 | 0.3961 | 1.329 | 11.81 | 17.15 | 1.45 |
| 8 | 270 | 1 | 0.948 | 0.1443 | 1.234 | 18.45 | 11.50 | 0.62 |
| 8 | 450 | 1 | 0.982 | 0.0200 | 0.150 | 9.21 | 9.55 | 1.04 |
| 8 | 500 | 1 | 0.972 | 0.0549 | 0.787 | 25.12 | 24.45 | 0.97 |
| 9 | 10 | 1 | 0.988 | 0.0721 | 0.004 | 0.34 | 0.33 | 0.96 |
| 9 | 63 | 1 | 0.996 | 0.0386 | 0.013 | 2.32 | 2.24 | 0.97 |
| 9 | 150 | 1 | 0.998 | 0.0041 | 0.074 | 8.80 | 8.54 | 0.97 |
| 9 | 270 | 1 | 0.993 | 0.0087 | 0.171 | 16.31 | 15.47 | 0.95 |
| 9 | 450 | 1 | 0.936 | 0.0684 | 0.382 | 7.42 | 8.23 | 1.11 |
| 9 | 500 | 1 | 0.968 | 0.0332 | 0.385 | 13.97 | 14.16 | 1.01 |
| 10 | 10 | 1 | 0.979 | 0.0806 | 0.005 | 0.33 | 0.34 | 1.01 |
| 10 | 63 | 1 | 0.994 | 0.0377 | 0.009 | 2.15 | 2.09 | 0.97 |
| 10 | 150 | 1 | 0.968 | 0.2431 | 0.684 | 6.92 | 7.94 | 1.15 |
| 10 | 270 | 1 | 0.924 | 0.1237 | 1.226 | 19.43 | 15.17 | 0.78 |
| 10 | 450 | 1 | 0.969 | 0.1090 | 1.143 | 23.38 | 16.46 | 0.70 |
| 10 | 500 | 1 | 0.978 | 0.0236 | 0.398 | 17.88 | 18.63 | 1.04 |

## Flags

- [WARN] K=3 10 MHz: gen_max=0.35Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=3 63 MHz: gen_max=3.19Ω (≥90% per-MHz clip ceiling 2.91Ω) — magnitude saturation
- [WARN] K=3 270 MHz: gen_max=36.32Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
- [WARN] K=3 500 MHz: gen_max=16.71Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
- [WARN] K=4 10 MHz: gen_max=0.32Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=4 270 MHz: gen_max=54.27Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
- [WARN] K=4 500 MHz: gen_max=19.51Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
- [WARN] K=5 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=5 500 MHz: gen_max=16.05Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
- [WARN] K=6 10 MHz: gen_max=0.32Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=6 500 MHz: gen_max=15.51Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
- [WARN] K=7 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=8 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=10 10 MHz: gen_max=0.33Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   K=3 10 MHz anchor: pearson_r=0.984, max_ratio=0.99
- [OK]   K=3 63 MHz anchor: pearson_r=0.998, max_ratio=1.03
- [OK]   K=3 150 MHz anchor: pearson_r=0.995, max_ratio=1.03
- [OK]   K=3 270 MHz anchor: pearson_r=0.962, max_ratio=0.69
- [OK]   K=3 450 MHz anchor: pearson_r=0.993, max_ratio=1.14
- [OK]   K=3 500 MHz anchor: pearson_r=0.935, max_ratio=0.68
- [OK]   K=4 10 MHz anchor: pearson_r=0.975, max_ratio=0.97
- [OK]   K=4 63 MHz anchor: pearson_r=0.997, max_ratio=1.01
- [OK]   K=4 150 MHz anchor: pearson_r=0.971, max_ratio=0.99
- [OK]   K=4 270 MHz anchor: pearson_r=0.968, max_ratio=0.85
- [OK]   K=4 450 MHz anchor: pearson_r=0.992, max_ratio=1.01
- [OK]   K=4 500 MHz anchor: pearson_r=0.986, max_ratio=0.96
- [OK]   K=5 10 MHz anchor: pearson_r=0.992, max_ratio=0.96
- [OK]   K=5 63 MHz anchor: pearson_r=0.997, max_ratio=0.98
- [OK]   K=5 150 MHz anchor: pearson_r=0.965, max_ratio=0.91
- [OK]   K=5 270 MHz anchor: pearson_r=0.990, max_ratio=1.04
- [OK]   K=5 450 MHz anchor: pearson_r=0.995, max_ratio=1.15
- [OK]   K=5 500 MHz anchor: pearson_r=0.989, max_ratio=0.82
- [OK]   K=6 10 MHz anchor: pearson_r=0.984, max_ratio=1.00
- [OK]   K=6 63 MHz anchor: pearson_r=0.996, max_ratio=1.00
- [WARN] K=6 150 MHz anchor: gen_max/real_max=1.91 — magnitude overshoot
- [OK]   K=6 270 MHz anchor: pearson_r=0.948, max_ratio=0.56
- [OK]   K=6 450 MHz anchor: pearson_r=0.973, max_ratio=1.09
- [OK]   K=7 10 MHz anchor: pearson_r=0.990, max_ratio=0.99
- [OK]   K=7 63 MHz anchor: pearson_r=0.995, max_ratio=0.99
- [OK]   K=7 150 MHz anchor: pearson_r=0.973, max_ratio=0.97
- [OK]   K=7 270 MHz anchor: pearson_r=0.992, max_ratio=1.09
- [OK]   K=7 450 MHz anchor: pearson_r=0.982, max_ratio=1.06
- [OK]   K=7 500 MHz anchor: pearson_r=0.957, max_ratio=0.80
- [OK]   K=8 10 MHz anchor: pearson_r=0.997, max_ratio=0.97
- [OK]   K=8 63 MHz anchor: pearson_r=0.996, max_ratio=0.94
- [OK]   K=8 270 MHz anchor: pearson_r=0.948, max_ratio=0.62
- [OK]   K=8 450 MHz anchor: pearson_r=0.982, max_ratio=1.04
- [OK]   K=8 500 MHz anchor: pearson_r=0.972, max_ratio=0.97
- [OK]   K=9 10 MHz anchor: pearson_r=0.988, max_ratio=0.96
- [OK]   K=9 63 MHz anchor: pearson_r=0.996, max_ratio=0.97
- [OK]   K=9 150 MHz anchor: pearson_r=0.998, max_ratio=0.97
- [OK]   K=9 270 MHz anchor: pearson_r=0.993, max_ratio=0.95
- [OK]   K=9 450 MHz anchor: pearson_r=0.936, max_ratio=1.11
- [OK]   K=9 500 MHz anchor: pearson_r=0.968, max_ratio=1.01
- [OK]   K=10 10 MHz anchor: pearson_r=0.979, max_ratio=1.01
- [OK]   K=10 63 MHz anchor: pearson_r=0.994, max_ratio=0.97
- [OK]   K=10 150 MHz anchor: pearson_r=0.968, max_ratio=1.15
- [OK]   K=10 270 MHz anchor: pearson_r=0.924, max_ratio=0.78
- [OK]   K=10 450 MHz anchor: pearson_r=0.969, max_ratio=0.70
- [OK]   K=10 500 MHz anchor: pearson_r=0.978, max_ratio=1.04

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp057_structured_graph
checkpoint: experiments/exp057_structured_graph/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 3–10 (8 values)  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 21.28

Per-MHz generation (layout decode, no GT):
  K | MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
  3 |     10 | yes    |    0.35 |    0.09 |    0.18 |     0.21 | yes
  3 |     63 | yes    |    3.19 |    1.34 |    2.10 |     2.91 | yes
  3 |    150 | yes    |    8.69 |    2.68 |    6.17 |    13.03 | no
  3 |    270 | yes    |   36.32 |    7.78 |   22.97 |    30.06 | yes
  3 |    450 | yes    |   10.65 |    2.47 |    5.34 |    15.12 | no
  3 |    500 | yes    |   16.71 |    4.12 |    8.98 |    16.68 | yes
  4 |     10 | yes    |    0.32 |    0.08 |    0.17 |     0.21 | yes
  4 |     63 | yes    |    2.38 |    0.88 |    1.48 |     2.91 | no
  4 |    150 | yes    |    7.46 |    2.02 |    4.72 |    13.03 | no
  4 |    270 | yes    |   54.27 |    9.52 |   36.92 |    30.06 | yes
  4 |    450 | yes    |   13.57 |    3.27 |    7.39 |    15.12 | no
  4 |    500 | yes    |   19.51 |    4.86 |   11.43 |    16.68 | yes
  5 |     10 | yes    |    0.34 |    0.09 |    0.17 |     0.21 | yes
  5 |     63 | yes    |    2.59 |    0.88 |    1.56 |     2.91 | no
  5 |    150 | yes    |    6.13 |    1.50 |    3.45 |    13.03 | no
  5 |    270 | yes    |   22.14 |    3.84 |   11.27 |    30.06 | no
  5 |    450 | yes    |   10.41 |    2.44 |    5.74 |    15.12 | no
  5 |    500 | yes    |   16.05 |    4.38 |   10.30 |    16.68 | yes
  6 |     10 | yes    |    0.32 |    0.08 |    0.16 |     0.21 | yes
  6 |     63 | yes    |    2.19 |    0.68 |    1.23 |     2.91 | no
  6 |    150 | yes    |    5.14 |    1.26 |    2.82 |    13.03 | no
  6 |    270 | yes    |   18.70 |    3.08 |    9.17 |    30.06 | no
  6 |    450 | yes    |   10.92 |    2.57 |    5.52 |    15.12 | no
  6 |    500 | yes    |   15.51 |    3.76 |    7.84 |    16.68 | yes
  7 |     10 | yes    |    0.34 |    0.07 |    0.15 |     0.21 | yes
  7 |     63 | yes    |    2.32 |    0.65 |    1.16 |     2.91 | no
  7 |    150 | yes    |    4.33 |    0.91 |    2.00 |    13.03 | no
  7 |    270 | yes    |   17.50 |    3.42 |    9.71 |    30.06 | no
  7 |    450 | yes    |   11.50 |    2.17 |    5.03 |    15.12 | no
  7 |    500 | yes    |   14.64 |    2.81 |    6.33 |    16.68 | no
  8 |     10 | yes    |    0.34 |    0.08 |    0.16 |     0.21 | yes
  8 |     63 | yes    |    2.17 |    0.62 |    1.15 |     2.91 | no
  8 |    150 | yes    |    4.79 |    0.99 |    2.30 |    13.03 | no
  8 |    270 | yes    |   18.24 |    2.63 |    8.25 |    30.06 | no
  8 |    450 | yes    |   10.40 |    2.14 |    4.90 |    15.12 | no
  8 |    500 | yes    |   13.83 |    2.87 |    6.34 |    16.68 | no
  9 |     10 | yes    |    0.31 |    0.08 |    0.16 |     0.21 | yes
  9 |     63 | yes    |    2.12 |    0.62 |    1.13 |     2.91 | no
  9 |    150 | yes    |    4.56 |    0.94 |    2.30 |    13.03 | no
  9 |    270 | yes    |   25.38 |    4.54 |   13.42 |    30.06 | no
  9 |    450 | yes    |    8.84 |    2.00 |    4.73 |    15.12 | no
  9 |    500 | yes    |   12.85 |    2.98 |    6.98 |    16.68 | no
 10 |     10 | yes    |    0.33 |    0.08 |    0.15 |     0.21 | yes
 10 |     63 | yes    |    2.14 |    0.57 |    1.08 |     2.91 | no
 10 |    150 | yes    |    6.47 |    1.70 |    3.88 |    13.03 | no
 10 |    270 | yes    |   22.07 |    4.67 |   16.64 |    30.06 | no
 10 |    450 | yes    |   10.80 |    2.70 |    5.17 |    15.12 | no
 10 |    500 | yes    |   12.75 |    3.16 |    6.58 |    16.68 | no

Layout vs real val heatmap (training anchors only):
  K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  3 |   10 | 1 | 0.984 | 0.0818 | 0.005 | 0.36 | 0.35 | 0.99
  3 |   63 | 1 | 0.998 | 0.0405 | 0.024 | 2.91 | 2.98 | 1.03
  3 |  150 | 1 | 0.995 | 0.0088 | 0.064 | 4.81 | 4.93 | 1.03
  3 |  270 | 1 | 0.962 | 0.2307 | 2.509 | 33.68 | 23.33 | 0.69
  3 |  450 | 1 | 0.993 | 0.0098 | 0.166 | 10.84 | 12.37 | 1.14
  3 |  500 | 1 | 0.935 | 0.1798 | 1.341 | 18.84 | 12.86 | 0.68
  4 |   10 | 1 | 0.975 | 0.1006 | 0.007 | 0.34 | 0.33 | 0.97
  4 |   63 | 1 | 0.997 | 0.0406 | 0.020 | 2.47 | 2.49 | 1.01
  4 |  150 | 1 | 0.971 | 0.0350 | 0.092 | 4.42 | 4.35 | 0.99
  4 |  270 | 1 | 0.968 | 0.0658 | 0.723 | 16.91 | 14.45 | 0.85
  4 |  450 | 1 | 0.992 | 0.0084 | 0.119 | 10.58 | 10.69 | 1.01
  4 |  500 | 1 | 0.986 | 0.0139 | 0.221 | 11.51 | 10.99 | 0.96
  5 |   10 | 1 | 0.992 | 0.0709 | 0.005 | 0.36 | 0.35 | 0.96
  5 |   63 | 1 | 0.997 | 0.0529 | 0.052 | 2.66 | 2.61 | 0.98
  5 |  150 | 1 | 0.965 | 0.0354 | 0.111 | 4.14 | 3.78 | 0.91
  5 |  270 | 1 | 0.990 | 0.0088 | 0.150 | 10.76 | 11.14 | 1.04
  5 |  450 | 1 | 0.995 | 0.0051 | 0.086 | 9.24 | 10.66 | 1.15
  5 |  500 | 1 | 0.989 | 0.0219 | 0.487 | 19.84 | 16.19 | 0.82
  6 |   10 | 1 | 0.984 | 0.0768 | 0.006 | 0.34 | 0.34 | 1.00
  6 |   63 | 1 | 0.996 | 0.0442 | 0.032 | 2.51 | 2.50 | 1.00
  6 |  150 | 1 | 0.931 | 1.9073 | 3.122 | 9.59 | 18.31 | 1.91
  6 |  270 | 1 | 0.948 | 0.3343 | 2.999 | 34.37 | 19.22 | 0.56
  6 |  450 | 1 | 0.973 | 0.0207 | 0.171 | 8.95 | 9.75 | 1.09
  6 |  500 | 1 | 0.925 | 0.1693 | 1.464 | 16.75 | 23.43 | 1.40
  7 |   10 | 1 | 0.990 | 0.0659 | 0.003 | 0.33 | 0.32 | 0.99
  7 |   63 | 1 | 0.995 | 0.0376 | 0.008 | 2.35 | 2.33 | 0.99
  7 |  150 | 1 | 0.973 | 0.1226 | 0.736 | 13.61 | 13.26 | 0.97
  7 |  270 | 1 | 0.992 | 0.0216 | 0.477 | 21.00 | 22.95 | 1.09
  7 |  450 | 1 | 0.982 | 0.0139 | 0.133 | 8.57 | 9.10 | 1.06
  7 |  500 | 1 | 0.957 | 0.0468 | 0.479 | 13.78 | 11.03 | 0.80
  8 |   10 | 1 | 0.997 | 0.0585 | 0.002 | 0.35 | 0.34 | 0.97
  8 |   63 | 1 | 0.996 | 0.0385 | 0.011 | 2.31 | 2.16 | 0.94
  8 |  150 | 1 | 0.984 | 0.3961 | 1.329 | 11.81 | 17.15 | 1.45
  8 |  270 | 1 | 0.948 | 0.1443 | 1.234 | 18.45 | 11.50 | 0.62
  8 |  450 | 1 | 0.982 | 0.0200 | 0.150 | 9.21 | 9.55 | 1.04
  8 |  500 | 1 | 0.972 | 0.0549 | 0.787 | 25.12 | 24.45 | 0.97
  9 |   10 | 1 | 0.988 | 0.0721 | 0.004 | 0.34 | 0.33 | 0.96
  9 |   63 | 1 | 0.996 | 0.0386 | 0.013 | 2.32 | 2.24 | 0.97
  9 |  150 | 1 | 0.998 | 0.0041 | 0.074 | 8.80 | 8.54 | 0.97
  9 |  270 | 1 | 0.993 | 0.0087 | 0.171 | 16.31 | 15.47 | 0.95
  9 |  450 | 1 | 0.936 | 0.0684 | 0.382 | 7.42 | 8.23 | 1.11
  9 |  500 | 1 | 0.968 | 0.0332 | 0.385 | 13.97 | 14.16 | 1.01
 10 |   10 | 1 | 0.979 | 0.0806 | 0.005 | 0.33 | 0.34 | 1.01
 10 |   63 | 1 | 0.994 | 0.0377 | 0.009 | 2.15 | 2.09 | 0.97
 10 |  150 | 1 | 0.968 | 0.2431 | 0.684 | 6.92 | 7.94 | 1.15
 10 |  270 | 1 | 0.924 | 0.1237 | 1.226 | 19.43 | 15.17 | 0.78
 10 |  450 | 1 | 0.969 | 0.1090 | 1.143 | 23.38 | 16.46 | 0.70
 10 |  500 | 1 | 0.978 | 0.0236 | 0.398 | 17.88 | 18.63 | 1.04

Flags:
[WARN] K=3 10 MHz: gen_max=0.35Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=3 63 MHz: gen_max=3.19Ω (≥90% per-MHz clip ceiling 2.91Ω) — magnitude saturation
[WARN] K=3 270 MHz: gen_max=36.32Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
[WARN] K=3 500 MHz: gen_max=16.71Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
[WARN] K=4 10 MHz: gen_max=0.32Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=4 270 MHz: gen_max=54.27Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
[WARN] K=4 500 MHz: gen_max=19.51Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
[WARN] K=5 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=5 500 MHz: gen_max=16.05Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
[WARN] K=6 10 MHz: gen_max=0.32Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=6 500 MHz: gen_max=15.51Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
[WARN] K=7 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=8 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=10 10 MHz: gen_max=0.33Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   K=3 10 MHz anchor: pearson_r=0.984, max_ratio=0.99
[OK]   K=3 63 MHz anchor: pearson_r=0.998, max_ratio=1.03
[OK]   K=3 150 MHz anchor: pearson_r=0.995, max_ratio=1.03
[OK]   K=3 270 MHz anchor: pearson_r=0.962, max_ratio=0.69
[OK]   K=3 450 MHz anchor: pearson_r=0.993, max_ratio=1.14
[OK]   K=3 500 MHz anchor: pearson_r=0.935, max_ratio=0.68
[OK]   K=4 10 MHz anchor: pearson_r=0.975, max_ratio=0.97
[OK]   K=4 63 MHz anchor: pearson_r=0.997, max_ratio=1.01
[OK]   K=4 150 MHz anchor: pearson_r=0.971, max_ratio=0.99
[OK]   K=4 270 MHz anchor: pearson_r=0.968, max_ratio=0.85
[OK]   K=4 450 MHz anchor: pearson_r=0.992, max_ratio=1.01
[OK]   K=4 500 MHz anchor: pearson_r=0.986, max_ratio=0.96
[OK]   K=5 10 MHz anchor: pearson_r=0.992, max_ratio=0.96
[OK]   K=5 63 MHz anchor: pearson_r=0.997, max_ratio=0.98
[OK]   K=5 150 MHz anchor: pearson_r=0.965, max_ratio=0.91
[OK]   K=5 270 MHz anchor: pearson_r=0.990, max_ratio=1.04
[OK]   K=5 450 MHz anchor: pearson_r=0.995, max_ratio=1.15
[OK]   K=5 500 MHz anchor: pearson_r=0.989, max_ratio=0.82
[OK]   K=6 10 MHz anchor: pearson_r=0.984, max_ratio=1.00
[OK]   K=6 63 MHz anchor: pearson_r=0.996, max_ratio=1.00
[WARN] K=6 150 MHz anchor: gen_max/real_max=1.91 — magnitude overshoot
[OK]   K=6 270 MHz anchor: pearson_r=0.948, max_ratio=0.56
[OK]   K=6 450 MHz anchor: pearson_r=0.973, max_ratio=1.09
[OK]   K=7 10 MHz anchor: pearson_r=0.990, max_ratio=0.99
[OK]   K=7 63 MHz anchor: pearson_r=0.995, max_ratio=0.99
[OK]   K=7 150 MHz anchor: pearson_r=0.973, max_ratio=0.97
[OK]   K=7 270 MHz anchor: pearson_r=0.992, max_ratio=1.09
[OK]   K=7 450 MHz anchor: pearson_r=0.982, max_ratio=1.06
[OK]   K=7 500 MHz anchor: pearson_r=0.957, max_ratio=0.80
[OK]   K=8 10 MHz anchor: pearson_r=0.997, max_ratio=0.97
[OK]   K=8 63 MHz anchor: pearson_r=0.996, max_ratio=0.94
[OK]   K=8 270 MHz anchor: pearson_r=0.948, max_ratio=0.62
[OK]   K=8 450 MHz anchor: pearson_r=0.982, max_ratio=1.04
[OK]   K=8 500 MHz anchor: pearson_r=0.972, max_ratio=0.97
[OK]   K=9 10 MHz anchor: pearson_r=0.988, max_ratio=0.96
[OK]   K=9 63 MHz anchor: pearson_r=0.996, max_ratio=0.97
[OK]   K=9 150 MHz anchor: pearson_r=0.998, max_ratio=0.97
[OK]   K=9 270 MHz anchor: pearson_r=0.993, max_ratio=0.95
[OK]   K=9 450 MHz anchor: pearson_r=0.936, max_ratio=1.11
[OK]   K=9 500 MHz anchor: pearson_r=0.968, max_ratio=1.01
[OK]   K=10 10 MHz anchor: pearson_r=0.979, max_ratio=1.01
[OK]   K=10 63 MHz anchor: pearson_r=0.994, max_ratio=0.97
[OK]   K=10 150 MHz anchor: pearson_r=0.968, max_ratio=1.15
[OK]   K=10 270 MHz anchor: pearson_r=0.924, max_ratio=0.78
[OK]   K=10 450 MHz anchor: pearson_r=0.969, max_ratio=0.70
[OK]   K=10 500 MHz anchor: pearson_r=0.978, max_ratio=1.04
=== END SWEEP QC ===
```
