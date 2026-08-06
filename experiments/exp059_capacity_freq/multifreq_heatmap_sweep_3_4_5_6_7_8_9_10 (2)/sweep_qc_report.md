# Sweep QC Report

- **Experiment:** `experiments/exp059_capacity_freq`
- **Checkpoint:** `experiments/exp059_capacity_freq/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 3–10 (8 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 3 | 10 | yes | 0.29 | 0.09 | 0.17 | yes |
| 3 | 80 | yes | 5.08 | 3.00 | 4.58 | no |
| 3 | 150 | yes | 4.34 | 1.05 | 2.38 | no |
| 3 | 270 | yes | 10.74 | 2.61 | 8.68 | no |
| 3 | 450 | yes | 7.53 | 2.23 | 4.29 | no |
| 3 | 550 | no | 13.57 | 3.88 | 7.48 | no |
| 4 | 10 | yes | 0.29 | 0.09 | 0.17 | yes |
| 4 | 80 | yes | 3.22 | 1.28 | 2.17 | no |
| 4 | 150 | yes | 3.35 | 0.81 | 2.01 | no |
| 4 | 270 | yes | 12.84 | 2.93 | 9.48 | no |
| 4 | 450 | yes | 7.88 | 2.35 | 4.79 | no |
| 4 | 550 | no | 9.34 | 2.79 | 5.27 | no |
| 5 | 10 | yes | 0.30 | 0.09 | 0.18 | yes |
| 5 | 80 | yes | 3.42 | 1.24 | 2.36 | no |
| 5 | 150 | yes | 3.10 | 0.75 | 1.63 | no |
| 5 | 270 | yes | 7.57 | 1.75 | 4.10 | no |
| 5 | 450 | yes | 7.31 | 2.31 | 4.25 | no |
| 5 | 550 | no | 11.33 | 3.32 | 6.29 | no |
| 6 | 10 | yes | 0.29 | 0.08 | 0.16 | yes |
| 6 | 80 | yes | 2.82 | 0.90 | 1.66 | no |
| 6 | 150 | yes | 12.04 | 5.40 | 10.59 | yes |
| 6 | 270 | yes | 11.29 | 2.70 | 8.65 | no |
| 6 | 450 | yes | 7.59 | 2.13 | 4.36 | no |
| 6 | 550 | no | 10.39 | 2.83 | 5.57 | no |
| 7 | 10 | yes | 0.29 | 0.08 | 0.14 | yes |
| 7 | 80 | yes | 2.89 | 0.84 | 1.56 | no |
| 7 | 150 | yes | 10.60 | 4.12 | 9.63 | no |
| 7 | 270 | yes | 9.24 | 2.33 | 6.24 | no |
| 7 | 450 | yes | 6.73 | 2.24 | 3.88 | no |
| 7 | 550 | no | 9.87 | 3.01 | 5.44 | no |
| 8 | 10 | yes | 0.30 | 0.08 | 0.16 | yes |
| 8 | 80 | yes | 2.85 | 0.82 | 1.55 | no |
| 8 | 150 | yes | 10.34 | 3.06 | 7.76 | no |
| 8 | 270 | yes | 9.95 | 2.67 | 7.17 | no |
| 8 | 450 | yes | 8.17 | 2.89 | 5.43 | no |
| 8 | 550 | no | 12.17 | 3.83 | 7.24 | no |
| 9 | 10 | yes | 0.28 | 0.08 | 0.16 | yes |
| 9 | 80 | yes | 2.47 | 0.81 | 1.49 | no |
| 9 | 150 | yes | 6.48 | 2.30 | 4.61 | no |
| 9 | 270 | yes | 13.32 | 3.68 | 9.83 | no |
| 9 | 450 | yes | 7.22 | 2.51 | 4.41 | no |
| 9 | 550 | no | 12.82 | 4.13 | 7.25 | no |
| 10 | 10 | yes | 0.28 | 0.08 | 0.15 | yes |
| 10 | 80 | yes | 2.66 | 0.77 | 1.46 | no |
| 10 | 150 | yes | 6.64 | 1.95 | 4.18 | no |
| 10 | 270 | yes | 11.26 | 2.83 | 7.33 | no |
| 10 | 450 | yes | 6.92 | 2.26 | 3.90 | no |
| 10 | 550 | no | 11.12 | 3.37 | 5.97 | no |

## Layout vs real (val, training anchors)

| K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|---|-----|---|-----------|--------|----------|----------|---------|-----------|
| 3 | 10 | 1 | 0.990 | 0.0686 | 0.005 | 0.36 | 0.30 | 0.85 |
| 3 | 80 | 1 | 0.994 | 0.1882 | 0.370 | 4.62 | 4.79 | 1.04 |
| 3 | 150 | 1 | 0.983 | 0.0301 | 0.107 | 4.81 | 3.86 | 0.80 |
| 3 | 270 | 1 | 0.880 | 0.9029 | 4.560 | 33.68 | 11.06 | 0.33 |
| 3 | 450 | 1 | 0.945 | 0.0781 | 0.367 | 10.84 | 7.97 | 0.74 |
| 4 | 10 | 1 | 0.978 | 0.0976 | 0.007 | 0.34 | 0.29 | 0.84 |
| 4 | 80 | 1 | 0.989 | 0.0946 | 0.150 | 3.21 | 3.30 | 1.03 |
| 4 | 150 | 1 | 0.980 | 0.0362 | 0.108 | 4.42 | 3.03 | 0.69 |
| 4 | 270 | 1 | 0.930 | 0.1379 | 0.900 | 16.91 | 11.72 | 0.69 |
| 4 | 450 | 1 | 0.852 | 0.1805 | 0.424 | 10.58 | 7.54 | 0.71 |
| 5 | 10 | 1 | 0.978 | 0.0950 | 0.007 | 0.36 | 0.30 | 0.84 |
| 5 | 80 | 1 | 0.995 | 0.0388 | 0.038 | 3.73 | 3.15 | 0.85 |
| 5 | 150 | 1 | 0.876 | 0.1143 | 0.207 | 4.14 | 2.82 | 0.68 |
| 5 | 270 | 1 | 0.922 | 0.0602 | 0.380 | 10.76 | 8.71 | 0.81 |
| 5 | 450 | 1 | 0.947 | 0.0942 | 0.329 | 9.24 | 7.86 | 0.85 |
| 6 | 10 | 1 | 0.991 | 0.0657 | 0.004 | 0.34 | 0.29 | 0.86 |
| 6 | 80 | 1 | 0.994 | 0.0403 | 0.047 | 3.46 | 3.03 | 0.87 |
| 6 | 150 | 1 | 0.929 | 0.5190 | 0.901 | 9.59 | 7.90 | 0.82 |
| 6 | 270 | 1 | 0.913 | 0.6444 | 4.117 | 34.37 | 12.24 | 0.36 |
| 6 | 450 | 1 | 0.906 | 0.0924 | 0.392 | 8.95 | 7.64 | 0.85 |
| 7 | 10 | 1 | 0.993 | 0.0644 | 0.004 | 0.33 | 0.30 | 0.91 |
| 7 | 80 | 1 | 0.992 | 0.0429 | 0.042 | 3.04 | 2.70 | 0.89 |
| 7 | 150 | 1 | 0.984 | 0.0613 | 0.523 | 13.61 | 10.78 | 0.79 |
| 7 | 270 | 1 | 0.910 | 0.1234 | 0.815 | 21.00 | 12.65 | 0.60 |
| 7 | 450 | 1 | 0.887 | 0.1172 | 0.394 | 8.57 | 6.89 | 0.80 |
| 8 | 10 | 1 | 0.995 | 0.0655 | 0.004 | 0.35 | 0.31 | 0.87 |
| 8 | 80 | 1 | 0.996 | 0.0407 | 0.044 | 3.02 | 2.76 | 0.91 |
| 8 | 150 | 1 | 0.993 | 0.0245 | 0.241 | 11.81 | 10.53 | 0.89 |
| 8 | 270 | 1 | 0.914 | 0.1737 | 1.232 | 18.45 | 12.96 | 0.70 |
| 8 | 450 | 1 | 0.772 | 0.2190 | 0.490 | 9.21 | 7.06 | 0.77 |
| 9 | 10 | 1 | 0.985 | 0.0839 | 0.006 | 0.34 | 0.30 | 0.88 |
| 9 | 80 | 1 | 0.996 | 0.0367 | 0.024 | 3.03 | 2.53 | 0.83 |
| 9 | 150 | 1 | 0.989 | 0.0525 | 0.300 | 8.80 | 8.48 | 0.96 |
| 9 | 270 | 1 | 0.738 | 0.2567 | 1.097 | 16.31 | 8.80 | 0.54 |
| 9 | 450 | 1 | 0.835 | 0.1617 | 0.593 | 7.42 | 7.96 | 1.07 |
| 10 | 10 | 1 | 0.968 | 0.0948 | 0.006 | 0.33 | 0.29 | 0.87 |
| 10 | 80 | 1 | 0.993 | 0.0403 | 0.033 | 2.98 | 2.64 | 0.89 |
| 10 | 150 | 1 | 0.988 | 0.0138 | 0.132 | 6.92 | 5.99 | 0.87 |
| 10 | 270 | 1 | 0.853 | 0.1900 | 1.412 | 19.43 | 11.46 | 0.59 |
| 10 | 450 | 1 | 0.874 | 0.5456 | 2.382 | 23.38 | 9.67 | 0.41 |

## Flags

- [WARN] K=3 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=4 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=5 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=6 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=6 150 MHz: gen_max=12.04Ω (≥90% per-MHz clip ceiling 13.03Ω) — magnitude saturation
- [WARN] K=7 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=8 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=10 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   K=3 10 MHz anchor: pearson_r=0.990, max_ratio=0.85
- [OK]   K=3 80 MHz anchor: pearson_r=0.994, max_ratio=1.04
- [OK]   K=3 150 MHz anchor: pearson_r=0.983, max_ratio=0.80
- [OK]   K=3 270 MHz anchor: pearson_r=0.880, max_ratio=0.33
- [OK]   K=3 450 MHz anchor: pearson_r=0.945, max_ratio=0.74
- [OK]   K=4 10 MHz anchor: pearson_r=0.978, max_ratio=0.84
- [OK]   K=4 80 MHz anchor: pearson_r=0.989, max_ratio=1.03
- [OK]   K=4 150 MHz anchor: pearson_r=0.980, max_ratio=0.69
- [OK]   K=4 270 MHz anchor: pearson_r=0.930, max_ratio=0.69
- [OK]   K=4 450 MHz anchor: pearson_r=0.852, max_ratio=0.71
- [OK]   K=5 10 MHz anchor: pearson_r=0.978, max_ratio=0.84
- [OK]   K=5 80 MHz anchor: pearson_r=0.995, max_ratio=0.85
- [OK]   K=5 150 MHz anchor: pearson_r=0.876, max_ratio=0.68
- [OK]   K=5 270 MHz anchor: pearson_r=0.922, max_ratio=0.81
- [OK]   K=5 450 MHz anchor: pearson_r=0.947, max_ratio=0.85
- [OK]   K=6 10 MHz anchor: pearson_r=0.991, max_ratio=0.86
- [OK]   K=6 80 MHz anchor: pearson_r=0.994, max_ratio=0.87
- [OK]   K=6 150 MHz anchor: pearson_r=0.929, max_ratio=0.82
- [OK]   K=6 270 MHz anchor: pearson_r=0.913, max_ratio=0.36
- [OK]   K=6 450 MHz anchor: pearson_r=0.906, max_ratio=0.85
- [OK]   K=7 10 MHz anchor: pearson_r=0.993, max_ratio=0.91
- [OK]   K=7 80 MHz anchor: pearson_r=0.992, max_ratio=0.89
- [OK]   K=7 150 MHz anchor: pearson_r=0.984, max_ratio=0.79
- [OK]   K=7 270 MHz anchor: pearson_r=0.910, max_ratio=0.60
- [OK]   K=7 450 MHz anchor: pearson_r=0.887, max_ratio=0.80
- [OK]   K=8 10 MHz anchor: pearson_r=0.995, max_ratio=0.87
- [OK]   K=8 80 MHz anchor: pearson_r=0.996, max_ratio=0.91
- [OK]   K=8 150 MHz anchor: pearson_r=0.993, max_ratio=0.89
- [OK]   K=8 270 MHz anchor: pearson_r=0.914, max_ratio=0.70
- [OK]   K=9 10 MHz anchor: pearson_r=0.985, max_ratio=0.88
- [OK]   K=9 80 MHz anchor: pearson_r=0.996, max_ratio=0.83
- [OK]   K=9 150 MHz anchor: pearson_r=0.989, max_ratio=0.96
- [WARN] K=9 270 MHz anchor: pearson_r=0.738 (<0.75) — weak spatial pattern
- [OK]   K=10 10 MHz anchor: pearson_r=0.968, max_ratio=0.87
- [OK]   K=10 80 MHz anchor: pearson_r=0.993, max_ratio=0.89
- [OK]   K=10 150 MHz anchor: pearson_r=0.988, max_ratio=0.87
- [OK]   K=10 270 MHz anchor: pearson_r=0.853, max_ratio=0.59
- [OK]   K=10 450 MHz anchor: pearson_r=0.874, max_ratio=0.41

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
  3 |     10 | yes    |    0.29 |    0.09 |    0.17 |     0.21 | yes
  3 |     80 | yes    |    5.08 |    3.00 |    4.58 |     9.78 | no
  3 |    150 | yes    |    4.34 |    1.05 |    2.38 |    13.03 | no
  3 |    270 | yes    |   10.74 |    2.61 |    8.68 |    30.06 | no
  3 |    450 | yes    |    7.53 |    2.23 |    4.29 |    15.12 | no
  3 |    550 | no     |   13.57 |    3.88 |    7.48 |    16.68 | no
  4 |     10 | yes    |    0.29 |    0.09 |    0.17 |     0.21 | yes
  4 |     80 | yes    |    3.22 |    1.28 |    2.17 |     9.78 | no
  4 |    150 | yes    |    3.35 |    0.81 |    2.01 |    13.03 | no
  4 |    270 | yes    |   12.84 |    2.93 |    9.48 |    30.06 | no
  4 |    450 | yes    |    7.88 |    2.35 |    4.79 |    15.12 | no
  4 |    550 | no     |    9.34 |    2.79 |    5.27 |    16.68 | no
  5 |     10 | yes    |    0.30 |    0.09 |    0.18 |     0.21 | yes
  5 |     80 | yes    |    3.42 |    1.24 |    2.36 |     9.78 | no
  5 |    150 | yes    |    3.10 |    0.75 |    1.63 |    13.03 | no
  5 |    270 | yes    |    7.57 |    1.75 |    4.10 |    30.06 | no
  5 |    450 | yes    |    7.31 |    2.31 |    4.25 |    15.12 | no
  5 |    550 | no     |   11.33 |    3.32 |    6.29 |    16.68 | no
  6 |     10 | yes    |    0.29 |    0.08 |    0.16 |     0.21 | yes
  6 |     80 | yes    |    2.82 |    0.90 |    1.66 |     9.78 | no
  6 |    150 | yes    |   12.04 |    5.40 |   10.59 |    13.03 | yes
  6 |    270 | yes    |   11.29 |    2.70 |    8.65 |    30.06 | no
  6 |    450 | yes    |    7.59 |    2.13 |    4.36 |    15.12 | no
  6 |    550 | no     |   10.39 |    2.83 |    5.57 |    16.68 | no
  7 |     10 | yes    |    0.29 |    0.08 |    0.14 |     0.21 | yes
  7 |     80 | yes    |    2.89 |    0.84 |    1.56 |     9.78 | no
  7 |    150 | yes    |   10.60 |    4.12 |    9.63 |    13.03 | no
  7 |    270 | yes    |    9.24 |    2.33 |    6.24 |    30.06 | no
  7 |    450 | yes    |    6.73 |    2.24 |    3.88 |    15.12 | no
  7 |    550 | no     |    9.87 |    3.01 |    5.44 |    16.68 | no
  8 |     10 | yes    |    0.30 |    0.08 |    0.16 |     0.21 | yes
  8 |     80 | yes    |    2.85 |    0.82 |    1.55 |     9.78 | no
  8 |    150 | yes    |   10.34 |    3.06 |    7.76 |    13.03 | no
  8 |    270 | yes    |    9.95 |    2.67 |    7.17 |    30.06 | no
  8 |    450 | yes    |    8.17 |    2.89 |    5.43 |    15.12 | no
  8 |    550 | no     |   12.17 |    3.83 |    7.24 |    16.68 | no
  9 |     10 | yes    |    0.28 |    0.08 |    0.16 |     0.21 | yes
  9 |     80 | yes    |    2.47 |    0.81 |    1.49 |     9.78 | no
  9 |    150 | yes    |    6.48 |    2.30 |    4.61 |    13.03 | no
  9 |    270 | yes    |   13.32 |    3.68 |    9.83 |    30.06 | no
  9 |    450 | yes    |    7.22 |    2.51 |    4.41 |    15.12 | no
  9 |    550 | no     |   12.82 |    4.13 |    7.25 |    16.68 | no
 10 |     10 | yes    |    0.28 |    0.08 |    0.15 |     0.21 | yes
 10 |     80 | yes    |    2.66 |    0.77 |    1.46 |     9.78 | no
 10 |    150 | yes    |    6.64 |    1.95 |    4.18 |    13.03 | no
 10 |    270 | yes    |   11.26 |    2.83 |    7.33 |    30.06 | no
 10 |    450 | yes    |    6.92 |    2.26 |    3.90 |    15.12 | no
 10 |    550 | no     |   11.12 |    3.37 |    5.97 |    16.68 | no

Layout vs real val heatmap (training anchors only):
  K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  3 |   10 | 1 | 0.990 | 0.0686 | 0.005 | 0.36 | 0.30 | 0.85
  3 |   80 | 1 | 0.994 | 0.1882 | 0.370 | 4.62 | 4.79 | 1.04
  3 |  150 | 1 | 0.983 | 0.0301 | 0.107 | 4.81 | 3.86 | 0.80
  3 |  270 | 1 | 0.880 | 0.9029 | 4.560 | 33.68 | 11.06 | 0.33
  3 |  450 | 1 | 0.945 | 0.0781 | 0.367 | 10.84 | 7.97 | 0.74
  4 |   10 | 1 | 0.978 | 0.0976 | 0.007 | 0.34 | 0.29 | 0.84
  4 |   80 | 1 | 0.989 | 0.0946 | 0.150 | 3.21 | 3.30 | 1.03
  4 |  150 | 1 | 0.980 | 0.0362 | 0.108 | 4.42 | 3.03 | 0.69
  4 |  270 | 1 | 0.930 | 0.1379 | 0.900 | 16.91 | 11.72 | 0.69
  4 |  450 | 1 | 0.852 | 0.1805 | 0.424 | 10.58 | 7.54 | 0.71
  5 |   10 | 1 | 0.978 | 0.0950 | 0.007 | 0.36 | 0.30 | 0.84
  5 |   80 | 1 | 0.995 | 0.0388 | 0.038 | 3.73 | 3.15 | 0.85
  5 |  150 | 1 | 0.876 | 0.1143 | 0.207 | 4.14 | 2.82 | 0.68
  5 |  270 | 1 | 0.922 | 0.0602 | 0.380 | 10.76 | 8.71 | 0.81
  5 |  450 | 1 | 0.947 | 0.0942 | 0.329 | 9.24 | 7.86 | 0.85
  6 |   10 | 1 | 0.991 | 0.0657 | 0.004 | 0.34 | 0.29 | 0.86
  6 |   80 | 1 | 0.994 | 0.0403 | 0.047 | 3.46 | 3.03 | 0.87
  6 |  150 | 1 | 0.929 | 0.5190 | 0.901 | 9.59 | 7.90 | 0.82
  6 |  270 | 1 | 0.913 | 0.6444 | 4.117 | 34.37 | 12.24 | 0.36
  6 |  450 | 1 | 0.906 | 0.0924 | 0.392 | 8.95 | 7.64 | 0.85
  7 |   10 | 1 | 0.993 | 0.0644 | 0.004 | 0.33 | 0.30 | 0.91
  7 |   80 | 1 | 0.992 | 0.0429 | 0.042 | 3.04 | 2.70 | 0.89
  7 |  150 | 1 | 0.984 | 0.0613 | 0.523 | 13.61 | 10.78 | 0.79
  7 |  270 | 1 | 0.910 | 0.1234 | 0.815 | 21.00 | 12.65 | 0.60
  7 |  450 | 1 | 0.887 | 0.1172 | 0.394 | 8.57 | 6.89 | 0.80
  8 |   10 | 1 | 0.995 | 0.0655 | 0.004 | 0.35 | 0.31 | 0.87
  8 |   80 | 1 | 0.996 | 0.0407 | 0.044 | 3.02 | 2.76 | 0.91
  8 |  150 | 1 | 0.993 | 0.0245 | 0.241 | 11.81 | 10.53 | 0.89
  8 |  270 | 1 | 0.914 | 0.1737 | 1.232 | 18.45 | 12.96 | 0.70
  8 |  450 | 1 | 0.772 | 0.2190 | 0.490 | 9.21 | 7.06 | 0.77
  9 |   10 | 1 | 0.985 | 0.0839 | 0.006 | 0.34 | 0.30 | 0.88
  9 |   80 | 1 | 0.996 | 0.0367 | 0.024 | 3.03 | 2.53 | 0.83
  9 |  150 | 1 | 0.989 | 0.0525 | 0.300 | 8.80 | 8.48 | 0.96
  9 |  270 | 1 | 0.738 | 0.2567 | 1.097 | 16.31 | 8.80 | 0.54
  9 |  450 | 1 | 0.835 | 0.1617 | 0.593 | 7.42 | 7.96 | 1.07
 10 |   10 | 1 | 0.968 | 0.0948 | 0.006 | 0.33 | 0.29 | 0.87
 10 |   80 | 1 | 0.993 | 0.0403 | 0.033 | 2.98 | 2.64 | 0.89
 10 |  150 | 1 | 0.988 | 0.0138 | 0.132 | 6.92 | 5.99 | 0.87
 10 |  270 | 1 | 0.853 | 0.1900 | 1.412 | 19.43 | 11.46 | 0.59
 10 |  450 | 1 | 0.874 | 0.5456 | 2.382 | 23.38 | 9.67 | 0.41

Flags:
[WARN] K=3 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=4 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=5 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=6 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=6 150 MHz: gen_max=12.04Ω (≥90% per-MHz clip ceiling 13.03Ω) — magnitude saturation
[WARN] K=7 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=8 10 MHz: gen_max=0.30Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=10 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   K=3 10 MHz anchor: pearson_r=0.990, max_ratio=0.85
[OK]   K=3 80 MHz anchor: pearson_r=0.994, max_ratio=1.04
[OK]   K=3 150 MHz anchor: pearson_r=0.983, max_ratio=0.80
[OK]   K=3 270 MHz anchor: pearson_r=0.880, max_ratio=0.33
[OK]   K=3 450 MHz anchor: pearson_r=0.945, max_ratio=0.74
[OK]   K=4 10 MHz anchor: pearson_r=0.978, max_ratio=0.84
[OK]   K=4 80 MHz anchor: pearson_r=0.989, max_ratio=1.03
[OK]   K=4 150 MHz anchor: pearson_r=0.980, max_ratio=0.69
[OK]   K=4 270 MHz anchor: pearson_r=0.930, max_ratio=0.69
[OK]   K=4 450 MHz anchor: pearson_r=0.852, max_ratio=0.71
[OK]   K=5 10 MHz anchor: pearson_r=0.978, max_ratio=0.84
[OK]   K=5 80 MHz anchor: pearson_r=0.995, max_ratio=0.85
[OK]   K=5 150 MHz anchor: pearson_r=0.876, max_ratio=0.68
[OK]   K=5 270 MHz anchor: pearson_r=0.922, max_ratio=0.81
[OK]   K=5 450 MHz anchor: pearson_r=0.947, max_ratio=0.85
[OK]   K=6 10 MHz anchor: pearson_r=0.991, max_ratio=0.86
[OK]   K=6 80 MHz anchor: pearson_r=0.994, max_ratio=0.87
[OK]   K=6 150 MHz anchor: pearson_r=0.929, max_ratio=0.82
[OK]   K=6 270 MHz anchor: pearson_r=0.913, max_ratio=0.36
[OK]   K=6 450 MHz anchor: pearson_r=0.906, max_ratio=0.85
[OK]   K=7 10 MHz anchor: pearson_r=0.993, max_ratio=0.91
[OK]   K=7 80 MHz anchor: pearson_r=0.992, max_ratio=0.89
[OK]   K=7 150 MHz anchor: pearson_r=0.984, max_ratio=0.79
[OK]   K=7 270 MHz anchor: pearson_r=0.910, max_ratio=0.60
[OK]   K=7 450 MHz anchor: pearson_r=0.887, max_ratio=0.80
[OK]   K=8 10 MHz anchor: pearson_r=0.995, max_ratio=0.87
[OK]   K=8 80 MHz anchor: pearson_r=0.996, max_ratio=0.91
[OK]   K=8 150 MHz anchor: pearson_r=0.993, max_ratio=0.89
[OK]   K=8 270 MHz anchor: pearson_r=0.914, max_ratio=0.70
[OK]   K=9 10 MHz anchor: pearson_r=0.985, max_ratio=0.88
[OK]   K=9 80 MHz anchor: pearson_r=0.996, max_ratio=0.83
[OK]   K=9 150 MHz anchor: pearson_r=0.989, max_ratio=0.96
[WARN] K=9 270 MHz anchor: pearson_r=0.738 (<0.75) — weak spatial pattern
[OK]   K=10 10 MHz anchor: pearson_r=0.968, max_ratio=0.87
[OK]   K=10 80 MHz anchor: pearson_r=0.993, max_ratio=0.89
[OK]   K=10 150 MHz anchor: pearson_r=0.988, max_ratio=0.87
[OK]   K=10 270 MHz anchor: pearson_r=0.853, max_ratio=0.59
[OK]   K=10 450 MHz anchor: pearson_r=0.874, max_ratio=0.41
=== END SWEEP QC ===
```
