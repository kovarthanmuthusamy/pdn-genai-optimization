# Sweep QC Report

- **Experiment:** `experiments/exp059_capacity_freq`
- **Checkpoint:** `experiments/exp059_capacity_freq/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 3–10 (8 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 3 | 10 | yes | 0.33 | 0.10 | 0.17 | yes |
| 3 | 80 | yes | 3.39 | 1.89 | 2.72 | no |
| 3 | 150 | yes | 4.94 | 0.98 | 2.38 | no |
| 3 | 270 | yes | 16.28 | 3.17 | 12.64 | no |
| 3 | 450 | yes | 10.00 | 1.97 | 4.49 | no |
| 3 | 550 | no | 14.63 | 3.79 | 7.95 | no |
| 4 | 10 | yes | 0.31 | 0.09 | 0.16 | yes |
| 4 | 80 | yes | 2.86 | 1.14 | 1.83 | no |
| 4 | 150 | yes | 4.12 | 0.72 | 1.85 | no |
| 4 | 270 | yes | 24.47 | 4.18 | 18.27 | no |
| 4 | 450 | yes | 10.03 | 2.57 | 5.67 | no |
| 4 | 550 | no | 10.02 | 2.49 | 5.71 | no |
| 5 | 10 | yes | 0.34 | 0.09 | 0.17 | yes |
| 5 | 80 | yes | 3.40 | 1.20 | 2.05 | no |
| 5 | 150 | yes | 4.21 | 0.74 | 1.81 | no |
| 5 | 270 | yes | 13.26 | 2.17 | 5.92 | no |
| 5 | 450 | yes | 7.42 | 2.10 | 4.38 | no |
| 5 | 550 | no | 13.21 | 3.29 | 6.90 | no |
| 6 | 10 | yes | 0.31 | 0.09 | 0.16 | yes |
| 6 | 80 | yes | 2.81 | 0.93 | 1.59 | no |
| 6 | 150 | yes | 6.20 | 3.49 | 5.60 | no |
| 6 | 270 | yes | 21.90 | 4.91 | 18.93 | no |
| 6 | 450 | yes | 8.53 | 2.05 | 4.45 | no |
| 6 | 550 | no | 9.69 | 2.48 | 4.89 | no |
| 7 | 10 | yes | 0.31 | 0.08 | 0.14 | yes |
| 7 | 80 | yes | 2.76 | 0.85 | 1.47 | no |
| 7 | 150 | yes | 7.01 | 3.25 | 6.44 | no |
| 7 | 270 | yes | 10.51 | 1.66 | 4.47 | no |
| 7 | 450 | yes | 6.42 | 2.36 | 4.57 | no |
| 7 | 550 | no | 13.54 | 2.99 | 6.54 | no |
| 8 | 10 | yes | 0.31 | 0.08 | 0.15 | yes |
| 8 | 80 | yes | 2.71 | 0.83 | 1.45 | no |
| 8 | 150 | yes | 7.48 | 2.56 | 5.66 | no |
| 8 | 270 | yes | 18.86 | 3.08 | 11.26 | no |
| 8 | 450 | yes | 6.89 | 2.05 | 3.92 | no |
| 8 | 550 | no | 14.01 | 3.61 | 7.20 | no |
| 9 | 10 | yes | 0.30 | 0.09 | 0.15 | yes |
| 9 | 80 | yes | 2.50 | 0.82 | 1.43 | no |
| 9 | 150 | yes | 5.30 | 1.90 | 3.38 | no |
| 9 | 270 | yes | 29.96 | 5.69 | 19.57 | yes |
| 9 | 450 | yes | 6.60 | 2.55 | 4.53 | no |
| 9 | 550 | no | 11.78 | 3.52 | 5.95 | no |
| 10 | 10 | yes | 0.30 | 0.08 | 0.15 | yes |
| 10 | 80 | yes | 2.68 | 0.79 | 1.41 | no |
| 10 | 150 | yes | 5.95 | 1.75 | 3.42 | no |
| 10 | 270 | yes | 24.27 | 2.83 | 11.14 | no |
| 10 | 450 | yes | 13.96 | 4.20 | 8.17 | yes |
| 10 | 550 | no | 13.87 | 2.86 | 5.53 | no |

## Layout vs real (val, training anchors)

| K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|---|-----|---|-----------|--------|----------|----------|---------|-----------|
| 3 | 10 | 1 | 0.990 | 0.0790 | 0.007 | 0.36 | 0.34 | 0.94 |
| 3 | 80 | 1 | 0.994 | 0.0848 | 0.165 | 4.62 | 3.37 | 0.73 |
| 3 | 150 | 1 | 0.996 | 0.0077 | 0.055 | 4.81 | 4.48 | 0.93 |
| 3 | 270 | 1 | 0.936 | 0.4342 | 3.184 | 33.68 | 24.62 | 0.73 |
| 3 | 450 | 1 | 0.992 | 0.0142 | 0.154 | 10.84 | 10.05 | 0.93 |
| 4 | 10 | 1 | 0.978 | 0.1319 | 0.011 | 0.34 | 0.31 | 0.91 |
| 4 | 80 | 1 | 0.999 | 0.0470 | 0.057 | 3.21 | 2.96 | 0.92 |
| 4 | 150 | 1 | 0.994 | 0.0111 | 0.051 | 4.42 | 4.20 | 0.95 |
| 4 | 270 | 1 | 0.983 | 0.0431 | 0.596 | 16.91 | 20.39 | 1.21 |
| 4 | 450 | 1 | 0.989 | 0.0181 | 0.147 | 10.58 | 10.14 | 0.96 |
| 5 | 10 | 1 | 0.989 | 0.0834 | 0.006 | 0.36 | 0.33 | 0.92 |
| 5 | 80 | 1 | 0.999 | 0.0438 | 0.053 | 3.73 | 3.41 | 0.91 |
| 5 | 150 | 1 | 0.943 | 0.0544 | 0.135 | 4.14 | 3.96 | 0.96 |
| 5 | 270 | 1 | 0.984 | 0.0382 | 0.443 | 10.76 | 14.22 | 1.32 |
| 5 | 450 | 1 | 0.992 | 0.0198 | 0.165 | 9.24 | 8.77 | 0.95 |
| 6 | 10 | 1 | 0.969 | 0.1367 | 0.011 | 0.34 | 0.30 | 0.88 |
| 6 | 80 | 1 | 0.999 | 0.0421 | 0.041 | 3.46 | 3.14 | 0.91 |
| 6 | 150 | 1 | 0.967 | 0.2792 | 0.765 | 9.59 | 6.29 | 0.66 |
| 6 | 270 | 1 | 0.973 | 0.1663 | 1.887 | 34.37 | 29.16 | 0.85 |
| 6 | 450 | 1 | 0.919 | 0.0600 | 0.284 | 8.95 | 8.69 | 0.97 |
| 7 | 10 | 1 | 0.982 | 0.1070 | 0.009 | 0.33 | 0.31 | 0.94 |
| 7 | 80 | 1 | 0.998 | 0.0453 | 0.049 | 3.04 | 2.75 | 0.91 |
| 7 | 150 | 1 | 0.997 | 0.5978 | 1.653 | 13.61 | 6.91 | 0.51 |
| 7 | 270 | 1 | 0.993 | 0.0474 | 0.801 | 21.00 | 30.52 | 1.45 |
| 7 | 450 | 1 | 0.934 | 0.0591 | 0.294 | 8.57 | 8.27 | 0.96 |
| 8 | 10 | 1 | 0.988 | 0.0892 | 0.008 | 0.35 | 0.33 | 0.94 |
| 8 | 80 | 1 | 0.999 | 0.0451 | 0.049 | 3.02 | 2.62 | 0.87 |
| 8 | 150 | 1 | 0.997 | 0.1128 | 0.542 | 11.81 | 7.52 | 0.64 |
| 8 | 270 | 1 | 0.989 | 0.0444 | 0.849 | 18.45 | 22.46 | 1.22 |
| 8 | 450 | 1 | 0.983 | 0.0189 | 0.144 | 9.21 | 8.10 | 0.88 |
| 9 | 10 | 1 | 0.974 | 0.1237 | 0.010 | 0.34 | 0.32 | 0.93 |
| 9 | 80 | 1 | 0.998 | 0.0441 | 0.043 | 3.03 | 2.78 | 0.92 |
| 9 | 150 | 1 | 0.996 | 0.0205 | 0.174 | 8.80 | 6.84 | 0.78 |
| 9 | 270 | 1 | 0.978 | 0.0488 | 0.730 | 16.31 | 25.17 | 1.54 |
| 9 | 450 | 1 | 0.952 | 0.0403 | 0.279 | 7.42 | 7.08 | 0.95 |
| 10 | 10 | 1 | 0.976 | 0.0966 | 0.008 | 0.33 | 0.30 | 0.91 |
| 10 | 80 | 1 | 0.997 | 0.0455 | 0.046 | 2.98 | 2.64 | 0.89 |
| 10 | 150 | 1 | 0.995 | 0.0584 | 0.276 | 6.92 | 5.17 | 0.75 |
| 10 | 270 | 1 | 0.985 | 0.0573 | 0.747 | 19.43 | 19.45 | 1.00 |
| 10 | 450 | 1 | 0.962 | 0.1637 | 1.510 | 23.38 | 12.50 | 0.53 |

## Flags

- [WARN] K=3 10 MHz: gen_max=0.33Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=4 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=5 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=6 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=7 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=8 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 270 MHz: gen_max=29.96Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
- [WARN] K=10 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=10 450 MHz: gen_max=13.96Ω (≥90% per-MHz clip ceiling 15.12Ω) — magnitude saturation
- [OK]   K=3 10 MHz anchor: pearson_r=0.990, max_ratio=0.94
- [OK]   K=3 80 MHz anchor: pearson_r=0.994, max_ratio=0.73
- [OK]   K=3 150 MHz anchor: pearson_r=0.996, max_ratio=0.93
- [OK]   K=3 270 MHz anchor: pearson_r=0.936, max_ratio=0.73
- [OK]   K=3 450 MHz anchor: pearson_r=0.992, max_ratio=0.93
- [OK]   K=4 10 MHz anchor: pearson_r=0.978, max_ratio=0.91
- [OK]   K=4 80 MHz anchor: pearson_r=0.999, max_ratio=0.92
- [OK]   K=4 150 MHz anchor: pearson_r=0.994, max_ratio=0.95
- [OK]   K=4 270 MHz anchor: pearson_r=0.983, max_ratio=1.21
- [OK]   K=4 450 MHz anchor: pearson_r=0.989, max_ratio=0.96
- [OK]   K=5 10 MHz anchor: pearson_r=0.989, max_ratio=0.92
- [OK]   K=5 80 MHz anchor: pearson_r=0.999, max_ratio=0.91
- [OK]   K=5 150 MHz anchor: pearson_r=0.943, max_ratio=0.96
- [OK]   K=5 450 MHz anchor: pearson_r=0.992, max_ratio=0.95
- [OK]   K=6 10 MHz anchor: pearson_r=0.969, max_ratio=0.88
- [OK]   K=6 80 MHz anchor: pearson_r=0.999, max_ratio=0.91
- [OK]   K=6 150 MHz anchor: pearson_r=0.967, max_ratio=0.66
- [OK]   K=6 270 MHz anchor: pearson_r=0.973, max_ratio=0.85
- [OK]   K=6 450 MHz anchor: pearson_r=0.919, max_ratio=0.97
- [OK]   K=7 10 MHz anchor: pearson_r=0.982, max_ratio=0.94
- [OK]   K=7 80 MHz anchor: pearson_r=0.998, max_ratio=0.91
- [OK]   K=7 150 MHz anchor: pearson_r=0.997, max_ratio=0.51
- [OK]   K=7 450 MHz anchor: pearson_r=0.934, max_ratio=0.96
- [OK]   K=8 10 MHz anchor: pearson_r=0.988, max_ratio=0.94
- [OK]   K=8 80 MHz anchor: pearson_r=0.999, max_ratio=0.87
- [OK]   K=8 150 MHz anchor: pearson_r=0.997, max_ratio=0.64
- [OK]   K=8 270 MHz anchor: pearson_r=0.989, max_ratio=1.22
- [OK]   K=8 450 MHz anchor: pearson_r=0.983, max_ratio=0.88
- [OK]   K=9 10 MHz anchor: pearson_r=0.974, max_ratio=0.93
- [OK]   K=9 80 MHz anchor: pearson_r=0.998, max_ratio=0.92
- [OK]   K=9 150 MHz anchor: pearson_r=0.996, max_ratio=0.78
- [WARN] K=9 270 MHz anchor: gen_max/real_max=1.54 — magnitude overshoot
- [OK]   K=9 450 MHz anchor: pearson_r=0.952, max_ratio=0.95
- [OK]   K=10 10 MHz anchor: pearson_r=0.976, max_ratio=0.91
- [OK]   K=10 80 MHz anchor: pearson_r=0.997, max_ratio=0.89
- [OK]   K=10 150 MHz anchor: pearson_r=0.995, max_ratio=0.75
- [OK]   K=10 270 MHz anchor: pearson_r=0.985, max_ratio=1.00
- [OK]   K=10 450 MHz anchor: pearson_r=0.962, max_ratio=0.53

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp059_capacity_freq
checkpoint: experiments/exp059_capacity_freq/checkpoints/last_model.pt
mode: layout_qc  layout_source: val
K: 3–10 (8 values)  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 21.28

Per-MHz generation (layout decode, no GT):
  K | MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
  3 |     10 | yes    |    0.33 |    0.10 |    0.17 |     0.21 | yes
  3 |     80 | yes    |    3.39 |    1.89 |    2.72 |     9.78 | no
  3 |    150 | yes    |    4.94 |    0.98 |    2.38 |    13.03 | no
  3 |    270 | yes    |   16.28 |    3.17 |   12.64 |    30.06 | no
  3 |    450 | yes    |   10.00 |    1.97 |    4.49 |    15.12 | no
  3 |    550 | no     |   14.63 |    3.79 |    7.95 |    16.68 | no
  4 |     10 | yes    |    0.31 |    0.09 |    0.16 |     0.21 | yes
  4 |     80 | yes    |    2.86 |    1.14 |    1.83 |     9.78 | no
  4 |    150 | yes    |    4.12 |    0.72 |    1.85 |    13.03 | no
  4 |    270 | yes    |   24.47 |    4.18 |   18.27 |    30.06 | no
  4 |    450 | yes    |   10.03 |    2.57 |    5.67 |    15.12 | no
  4 |    550 | no     |   10.02 |    2.49 |    5.71 |    16.68 | no
  5 |     10 | yes    |    0.34 |    0.09 |    0.17 |     0.21 | yes
  5 |     80 | yes    |    3.40 |    1.20 |    2.05 |     9.78 | no
  5 |    150 | yes    |    4.21 |    0.74 |    1.81 |    13.03 | no
  5 |    270 | yes    |   13.26 |    2.17 |    5.92 |    30.06 | no
  5 |    450 | yes    |    7.42 |    2.10 |    4.38 |    15.12 | no
  5 |    550 | no     |   13.21 |    3.29 |    6.90 |    16.68 | no
  6 |     10 | yes    |    0.31 |    0.09 |    0.16 |     0.21 | yes
  6 |     80 | yes    |    2.81 |    0.93 |    1.59 |     9.78 | no
  6 |    150 | yes    |    6.20 |    3.49 |    5.60 |    13.03 | no
  6 |    270 | yes    |   21.90 |    4.91 |   18.93 |    30.06 | no
  6 |    450 | yes    |    8.53 |    2.05 |    4.45 |    15.12 | no
  6 |    550 | no     |    9.69 |    2.48 |    4.89 |    16.68 | no
  7 |     10 | yes    |    0.31 |    0.08 |    0.14 |     0.21 | yes
  7 |     80 | yes    |    2.76 |    0.85 |    1.47 |     9.78 | no
  7 |    150 | yes    |    7.01 |    3.25 |    6.44 |    13.03 | no
  7 |    270 | yes    |   10.51 |    1.66 |    4.47 |    30.06 | no
  7 |    450 | yes    |    6.42 |    2.36 |    4.57 |    15.12 | no
  7 |    550 | no     |   13.54 |    2.99 |    6.54 |    16.68 | no
  8 |     10 | yes    |    0.31 |    0.08 |    0.15 |     0.21 | yes
  8 |     80 | yes    |    2.71 |    0.83 |    1.45 |     9.78 | no
  8 |    150 | yes    |    7.48 |    2.56 |    5.66 |    13.03 | no
  8 |    270 | yes    |   18.86 |    3.08 |   11.26 |    30.06 | no
  8 |    450 | yes    |    6.89 |    2.05 |    3.92 |    15.12 | no
  8 |    550 | no     |   14.01 |    3.61 |    7.20 |    16.68 | no
  9 |     10 | yes    |    0.30 |    0.09 |    0.15 |     0.21 | yes
  9 |     80 | yes    |    2.50 |    0.82 |    1.43 |     9.78 | no
  9 |    150 | yes    |    5.30 |    1.90 |    3.38 |    13.03 | no
  9 |    270 | yes    |   29.96 |    5.69 |   19.57 |    30.06 | yes
  9 |    450 | yes    |    6.60 |    2.55 |    4.53 |    15.12 | no
  9 |    550 | no     |   11.78 |    3.52 |    5.95 |    16.68 | no
 10 |     10 | yes    |    0.30 |    0.08 |    0.15 |     0.21 | yes
 10 |     80 | yes    |    2.68 |    0.79 |    1.41 |     9.78 | no
 10 |    150 | yes    |    5.95 |    1.75 |    3.42 |    13.03 | no
 10 |    270 | yes    |   24.27 |    2.83 |   11.14 |    30.06 | no
 10 |    450 | yes    |   13.96 |    4.20 |    8.17 |    15.12 | yes
 10 |    550 | no     |   13.87 |    2.86 |    5.53 |    16.68 | no

Layout vs real val heatmap (training anchors only):
  K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  3 |   10 | 1 | 0.990 | 0.0790 | 0.007 | 0.36 | 0.34 | 0.94
  3 |   80 | 1 | 0.994 | 0.0848 | 0.165 | 4.62 | 3.37 | 0.73
  3 |  150 | 1 | 0.996 | 0.0077 | 0.055 | 4.81 | 4.48 | 0.93
  3 |  270 | 1 | 0.936 | 0.4342 | 3.184 | 33.68 | 24.62 | 0.73
  3 |  450 | 1 | 0.992 | 0.0142 | 0.154 | 10.84 | 10.05 | 0.93
  4 |   10 | 1 | 0.978 | 0.1319 | 0.011 | 0.34 | 0.31 | 0.91
  4 |   80 | 1 | 0.999 | 0.0470 | 0.057 | 3.21 | 2.96 | 0.92
  4 |  150 | 1 | 0.994 | 0.0111 | 0.051 | 4.42 | 4.20 | 0.95
  4 |  270 | 1 | 0.983 | 0.0431 | 0.596 | 16.91 | 20.39 | 1.21
  4 |  450 | 1 | 0.989 | 0.0181 | 0.147 | 10.58 | 10.14 | 0.96
  5 |   10 | 1 | 0.989 | 0.0834 | 0.006 | 0.36 | 0.33 | 0.92
  5 |   80 | 1 | 0.999 | 0.0438 | 0.053 | 3.73 | 3.41 | 0.91
  5 |  150 | 1 | 0.943 | 0.0544 | 0.135 | 4.14 | 3.96 | 0.96
  5 |  270 | 1 | 0.984 | 0.0382 | 0.443 | 10.76 | 14.22 | 1.32
  5 |  450 | 1 | 0.992 | 0.0198 | 0.165 | 9.24 | 8.77 | 0.95
  6 |   10 | 1 | 0.969 | 0.1367 | 0.011 | 0.34 | 0.30 | 0.88
  6 |   80 | 1 | 0.999 | 0.0421 | 0.041 | 3.46 | 3.14 | 0.91
  6 |  150 | 1 | 0.967 | 0.2792 | 0.765 | 9.59 | 6.29 | 0.66
  6 |  270 | 1 | 0.973 | 0.1663 | 1.887 | 34.37 | 29.16 | 0.85
  6 |  450 | 1 | 0.919 | 0.0600 | 0.284 | 8.95 | 8.69 | 0.97
  7 |   10 | 1 | 0.982 | 0.1070 | 0.009 | 0.33 | 0.31 | 0.94
  7 |   80 | 1 | 0.998 | 0.0453 | 0.049 | 3.04 | 2.75 | 0.91
  7 |  150 | 1 | 0.997 | 0.5978 | 1.653 | 13.61 | 6.91 | 0.51
  7 |  270 | 1 | 0.993 | 0.0474 | 0.801 | 21.00 | 30.52 | 1.45
  7 |  450 | 1 | 0.934 | 0.0591 | 0.294 | 8.57 | 8.27 | 0.96
  8 |   10 | 1 | 0.988 | 0.0892 | 0.008 | 0.35 | 0.33 | 0.94
  8 |   80 | 1 | 0.999 | 0.0451 | 0.049 | 3.02 | 2.62 | 0.87
  8 |  150 | 1 | 0.997 | 0.1128 | 0.542 | 11.81 | 7.52 | 0.64
  8 |  270 | 1 | 0.989 | 0.0444 | 0.849 | 18.45 | 22.46 | 1.22
  8 |  450 | 1 | 0.983 | 0.0189 | 0.144 | 9.21 | 8.10 | 0.88
  9 |   10 | 1 | 0.974 | 0.1237 | 0.010 | 0.34 | 0.32 | 0.93
  9 |   80 | 1 | 0.998 | 0.0441 | 0.043 | 3.03 | 2.78 | 0.92
  9 |  150 | 1 | 0.996 | 0.0205 | 0.174 | 8.80 | 6.84 | 0.78
  9 |  270 | 1 | 0.978 | 0.0488 | 0.730 | 16.31 | 25.17 | 1.54
  9 |  450 | 1 | 0.952 | 0.0403 | 0.279 | 7.42 | 7.08 | 0.95
 10 |   10 | 1 | 0.976 | 0.0966 | 0.008 | 0.33 | 0.30 | 0.91
 10 |   80 | 1 | 0.997 | 0.0455 | 0.046 | 2.98 | 2.64 | 0.89
 10 |  150 | 1 | 0.995 | 0.0584 | 0.276 | 6.92 | 5.17 | 0.75
 10 |  270 | 1 | 0.985 | 0.0573 | 0.747 | 19.43 | 19.45 | 1.00
 10 |  450 | 1 | 0.962 | 0.1637 | 1.510 | 23.38 | 12.50 | 0.53

Flags:
[WARN] K=3 10 MHz: gen_max=0.33Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=4 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=5 10 MHz: gen_max=0.34Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=6 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=7 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=8 10 MHz: gen_max=0.31Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 270 MHz: gen_max=29.96Ω (≥90% per-MHz clip ceiling 30.06Ω) — magnitude saturation
[WARN] K=10 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=10 450 MHz: gen_max=13.96Ω (≥90% per-MHz clip ceiling 15.12Ω) — magnitude saturation
[OK]   K=3 10 MHz anchor: pearson_r=0.990, max_ratio=0.94
[OK]   K=3 80 MHz anchor: pearson_r=0.994, max_ratio=0.73
[OK]   K=3 150 MHz anchor: pearson_r=0.996, max_ratio=0.93
[OK]   K=3 270 MHz anchor: pearson_r=0.936, max_ratio=0.73
[OK]   K=3 450 MHz anchor: pearson_r=0.992, max_ratio=0.93
[OK]   K=4 10 MHz anchor: pearson_r=0.978, max_ratio=0.91
[OK]   K=4 80 MHz anchor: pearson_r=0.999, max_ratio=0.92
[OK]   K=4 150 MHz anchor: pearson_r=0.994, max_ratio=0.95
[OK]   K=4 270 MHz anchor: pearson_r=0.983, max_ratio=1.21
[OK]   K=4 450 MHz anchor: pearson_r=0.989, max_ratio=0.96
[OK]   K=5 10 MHz anchor: pearson_r=0.989, max_ratio=0.92
[OK]   K=5 80 MHz anchor: pearson_r=0.999, max_ratio=0.91
[OK]   K=5 150 MHz anchor: pearson_r=0.943, max_ratio=0.96
[OK]   K=5 450 MHz anchor: pearson_r=0.992, max_ratio=0.95
[OK]   K=6 10 MHz anchor: pearson_r=0.969, max_ratio=0.88
[OK]   K=6 80 MHz anchor: pearson_r=0.999, max_ratio=0.91
[OK]   K=6 150 MHz anchor: pearson_r=0.967, max_ratio=0.66
[OK]   K=6 270 MHz anchor: pearson_r=0.973, max_ratio=0.85
[OK]   K=6 450 MHz anchor: pearson_r=0.919, max_ratio=0.97
[OK]   K=7 10 MHz anchor: pearson_r=0.982, max_ratio=0.94
[OK]   K=7 80 MHz anchor: pearson_r=0.998, max_ratio=0.91
[OK]   K=7 150 MHz anchor: pearson_r=0.997, max_ratio=0.51
[OK]   K=7 450 MHz anchor: pearson_r=0.934, max_ratio=0.96
[OK]   K=8 10 MHz anchor: pearson_r=0.988, max_ratio=0.94
[OK]   K=8 80 MHz anchor: pearson_r=0.999, max_ratio=0.87
[OK]   K=8 150 MHz anchor: pearson_r=0.997, max_ratio=0.64
[OK]   K=8 270 MHz anchor: pearson_r=0.989, max_ratio=1.22
[OK]   K=8 450 MHz anchor: pearson_r=0.983, max_ratio=0.88
[OK]   K=9 10 MHz anchor: pearson_r=0.974, max_ratio=0.93
[OK]   K=9 80 MHz anchor: pearson_r=0.998, max_ratio=0.92
[OK]   K=9 150 MHz anchor: pearson_r=0.996, max_ratio=0.78
[WARN] K=9 270 MHz anchor: gen_max/real_max=1.54 — magnitude overshoot
[OK]   K=9 450 MHz anchor: pearson_r=0.952, max_ratio=0.95
[OK]   K=10 10 MHz anchor: pearson_r=0.976, max_ratio=0.91
[OK]   K=10 80 MHz anchor: pearson_r=0.997, max_ratio=0.89
[OK]   K=10 150 MHz anchor: pearson_r=0.995, max_ratio=0.75
[OK]   K=10 270 MHz anchor: pearson_r=0.985, max_ratio=1.00
[OK]   K=10 450 MHz anchor: pearson_r=0.962, max_ratio=0.53
=== END SWEEP QC ===
```
