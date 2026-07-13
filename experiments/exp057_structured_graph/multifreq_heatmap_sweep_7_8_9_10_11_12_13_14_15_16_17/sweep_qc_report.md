# Sweep QC Report

- **Experiment:** `experiments/exp056_graph_vae`
- **Checkpoint:** `experiments/exp056_graph_vae/checkpoints/checkpoint_epoch_400.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 7–17 (11 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 7 | 10 | yes | 0.29 | 0.07 | 0.13 | yes |
| 7 | 63 | yes | 2.04 | 0.58 | 1.05 | no |
| 7 | 150 | yes | 7.69 | 2.63 | 5.95 | no |
| 7 | 270 | yes | 9.33 | 1.83 | 3.81 | no |
| 7 | 450 | yes | 9.56 | 1.81 | 3.91 | no |
| 7 | 500 | yes | 13.26 | 3.08 | 5.98 | no |
| 8 | 10 | yes | 0.29 | 0.07 | 0.15 | yes |
| 8 | 63 | yes | 2.06 | 0.59 | 1.10 | no |
| 8 | 150 | yes | 7.24 | 2.41 | 5.58 | no |
| 8 | 270 | yes | 12.41 | 1.95 | 4.67 | no |
| 8 | 450 | yes | 9.55 | 2.64 | 5.41 | no |
| 8 | 500 | yes | 10.77 | 3.08 | 5.63 | no |
| 9 | 10 | yes | 0.28 | 0.08 | 0.15 | yes |
| 9 | 63 | yes | 1.97 | 0.58 | 1.07 | no |
| 9 | 150 | yes | 7.07 | 2.42 | 4.86 | no |
| 9 | 270 | yes | 21.33 | 3.73 | 11.05 | no |
| 9 | 450 | yes | 7.49 | 1.97 | 4.41 | no |
| 9 | 500 | yes | 11.29 | 3.11 | 6.46 | no |
| 10 | 10 | yes | 0.29 | 0.07 | 0.15 | yes |
| 10 | 63 | yes | 1.98 | 0.52 | 0.99 | no |
| 10 | 150 | yes | 6.59 | 1.69 | 3.55 | no |
| 10 | 270 | yes | 12.34 | 2.31 | 5.81 | no |
| 10 | 450 | yes | 8.12 | 2.41 | 4.34 | no |
| 10 | 500 | yes | 11.84 | 2.89 | 5.91 | no |
| 11 | 10 | yes | 0.27 | 0.07 | 0.13 | yes |
| 11 | 63 | yes | 1.95 | 0.52 | 0.96 | no |
| 11 | 150 | yes | 5.97 | 1.60 | 3.03 | no |
| 11 | 270 | yes | 8.66 | 2.06 | 4.74 | no |
| 11 | 450 | yes | 10.83 | 2.98 | 5.35 | no |
| 11 | 500 | yes | 12.08 | 3.06 | 6.75 | no |
| 12 | 10 | yes | 0.27 | 0.07 | 0.13 | yes |
| 12 | 63 | yes | 1.99 | 0.52 | 1.00 | no |
| 12 | 150 | yes | 7.24 | 1.75 | 4.01 | no |
| 12 | 270 | yes | 16.69 | 3.21 | 8.09 | no |
| 12 | 450 | yes | 7.38 | 2.38 | 4.17 | no |
| 12 | 500 | yes | 12.15 | 2.79 | 6.07 | no |
| 13 | 10 | yes | 0.27 | 0.06 | 0.13 | yes |
| 13 | 63 | yes | 1.93 | 0.49 | 0.93 | no |
| 13 | 150 | yes | 6.58 | 1.48 | 3.34 | no |
| 13 | 270 | yes | 14.42 | 2.62 | 7.56 | no |
| 13 | 450 | yes | 9.47 | 2.29 | 4.12 | no |
| 13 | 500 | yes | 10.66 | 2.60 | 4.79 | no |
| 14 | 10 | yes | 0.27 | 0.06 | 0.12 | yes |
| 14 | 63 | yes | 1.94 | 0.48 | 0.90 | no |
| 14 | 150 | yes | 6.09 | 1.35 | 2.89 | no |
| 14 | 270 | yes | 13.50 | 2.16 | 4.74 | no |
| 14 | 450 | yes | 10.04 | 2.53 | 5.29 | no |
| 14 | 500 | yes | 12.05 | 2.62 | 5.09 | no |
| 15 | 10 | yes | 0.27 | 0.07 | 0.14 | yes |
| 15 | 63 | yes | 1.84 | 0.50 | 0.98 | no |
| 15 | 150 | yes | 6.39 | 1.53 | 3.43 | no |
| 15 | 270 | yes | 8.09 | 2.30 | 5.59 | no |
| 15 | 450 | yes | 9.53 | 2.54 | 4.38 | no |
| 15 | 500 | yes | 12.13 | 2.92 | 5.42 | no |
| 16 | 10 | yes | 0.26 | 0.06 | 0.12 | yes |
| 16 | 63 | yes | 1.82 | 0.47 | 0.88 | no |
| 16 | 150 | yes | 4.99 | 1.18 | 2.31 | no |
| 16 | 270 | yes | 16.67 | 3.33 | 9.99 | no |
| 16 | 450 | yes | 9.30 | 2.40 | 4.32 | no |
| 16 | 500 | yes | 10.37 | 2.60 | 5.08 | no |
| 17 | 10 | yes | 0.27 | 0.07 | 0.13 | yes |
| 17 | 63 | yes | 1.85 | 0.48 | 0.91 | no |
| 17 | 150 | yes | 5.34 | 1.27 | 2.54 | no |
| 17 | 270 | yes | 13.87 | 2.93 | 6.85 | no |
| 17 | 450 | yes | 10.00 | 2.76 | 4.74 | no |
| 17 | 500 | yes | 10.36 | 2.68 | 5.26 | no |

## Layout vs real (val, training anchors)

| K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |
|---|-----|---|-----------|--------|----------|----------|---------|-----------|
| 7 | 10 | 1 | 0.983 | 0.0760 | 0.006 | 0.33 | 0.28 | 0.84 |
| 7 | 63 | 1 | 0.995 | 0.0480 | 0.052 | 2.35 | 2.06 | 0.88 |
| 7 | 150 | 1 | 0.934 | 1.6301 | 2.425 | 13.61 | 6.72 | 0.49 |
| 7 | 270 | 1 | 0.974 | 0.0577 | 0.612 | 21.00 | 21.04 | 1.00 |
| 7 | 450 | 1 | 0.875 | 0.1202 | 0.404 | 8.57 | 7.34 | 0.86 |
| 7 | 500 | 1 | 0.891 | 0.1524 | 0.805 | 13.78 | 12.73 | 0.92 |
| 8 | 10 | 1 | 0.989 | 0.0757 | 0.007 | 0.35 | 0.29 | 0.84 |
| 8 | 63 | 1 | 0.993 | 0.0530 | 0.058 | 2.31 | 1.99 | 0.86 |
| 8 | 150 | 1 | 0.984 | 0.0808 | 0.432 | 11.81 | 7.80 | 0.66 |
| 8 | 270 | 1 | 0.972 | 0.0550 | 0.655 | 18.45 | 20.31 | 1.10 |
| 8 | 450 | 1 | 0.952 | 0.0458 | 0.205 | 9.21 | 8.74 | 0.95 |
| 8 | 500 | 1 | 0.926 | 0.2773 | 1.895 | 25.12 | 14.98 | 0.60 |
| 9 | 10 | 1 | 0.977 | 0.0777 | 0.007 | 0.34 | 0.29 | 0.84 |
| 9 | 63 | 1 | 0.994 | 0.0496 | 0.054 | 2.32 | 2.03 | 0.88 |
| 9 | 150 | 1 | 0.993 | 0.0178 | 0.141 | 8.80 | 7.47 | 0.85 |
| 9 | 270 | 1 | 0.971 | 0.0357 | 0.417 | 16.31 | 15.60 | 0.96 |
| 9 | 450 | 1 | 0.848 | 0.1122 | 0.484 | 7.42 | 7.85 | 1.06 |
| 9 | 500 | 1 | 0.806 | 0.2134 | 1.024 | 13.97 | 11.88 | 0.85 |
| 10 | 10 | 1 | 0.961 | 0.1148 | 0.008 | 0.33 | 0.29 | 0.85 |
| 10 | 63 | 1 | 0.991 | 0.0460 | 0.043 | 2.15 | 1.97 | 0.91 |
| 10 | 150 | 1 | 0.993 | 0.0117 | 0.112 | 6.92 | 6.27 | 0.91 |
| 10 | 270 | 1 | 0.928 | 0.0996 | 0.994 | 19.43 | 16.77 | 0.86 |
| 10 | 450 | 1 | 0.905 | 0.4678 | 2.281 | 23.38 | 13.58 | 0.58 |
| 10 | 500 | 1 | 0.927 | 0.1829 | 1.337 | 17.88 | 13.94 | 0.78 |
| 11 | 10 | 1 | 0.969 | 0.0937 | 0.008 | 0.31 | 0.28 | 0.92 |
| 11 | 63 | 1 | 0.993 | 0.0475 | 0.044 | 2.21 | 1.96 | 0.89 |
| 11 | 150 | 1 | 0.991 | 0.0198 | 0.142 | 5.99 | 5.88 | 0.98 |
| 11 | 270 | 1 | 0.890 | 0.5033 | 3.450 | 36.48 | 19.97 | 0.55 |
| 11 | 450 | 1 | 0.976 | 0.0505 | 0.481 | 13.35 | 11.22 | 0.84 |
| 11 | 500 | 1 | 0.979 | 0.0165 | 0.253 | 11.86 | 13.16 | 1.11 |
| 12 | 10 | 1 | 0.973 | 0.0890 | 0.007 | 0.29 | 0.26 | 0.90 |
| 12 | 63 | 1 | 0.992 | 0.0426 | 0.032 | 2.19 | 1.95 | 0.89 |
| 12 | 150 | 1 | 0.989 | 0.0140 | 0.124 | 6.82 | 6.61 | 0.97 |
| 12 | 270 | 1 | 0.952 | 0.0386 | 0.215 | 12.45 | 15.83 | 1.27 |
| 12 | 450 | 1 | 0.960 | 0.1206 | 0.945 | 18.37 | 11.66 | 0.63 |
| 12 | 500 | 1 | 0.967 | 0.0276 | 0.303 | 12.36 | 12.92 | 1.05 |
| 13 | 10 | 1 | 0.980 | 0.0733 | 0.006 | 0.32 | 0.25 | 0.79 |
| 13 | 63 | 1 | 0.989 | 0.0449 | 0.036 | 2.01 | 1.86 | 0.93 |
| 13 | 150 | 1 | 0.997 | 0.0064 | 0.079 | 5.70 | 5.44 | 0.95 |
| 13 | 270 | 1 | 0.858 | 0.7577 | 4.059 | 25.15 | 13.22 | 0.53 |
| 13 | 450 | 1 | 0.888 | 0.2159 | 1.221 | 14.98 | 9.67 | 0.65 |
| 13 | 500 | 1 | 0.957 | 0.0350 | 0.298 | 11.32 | 11.84 | 1.05 |
| 14 | 10 | 1 | 0.977 | 0.1215 | 0.010 | 0.35 | 0.27 | 0.78 |
| 14 | 63 | 1 | 0.992 | 0.0413 | 0.032 | 2.20 | 1.94 | 0.88 |
| 14 | 150 | 1 | 0.996 | 0.0074 | 0.082 | 6.14 | 5.98 | 0.97 |
| 14 | 270 | 1 | 0.960 | 0.0543 | 0.348 | 13.38 | 16.28 | 1.22 |
| 14 | 450 | 1 | 0.632 | 0.2662 | 0.741 | 7.10 | 8.38 | 1.18 |
| 14 | 500 | 1 | 0.908 | 0.0969 | 0.444 | 11.46 | 12.39 | 1.08 |
| 15 | 10 | 1 | 0.979 | 0.0750 | 0.006 | 0.33 | 0.28 | 0.85 |
| 15 | 63 | 1 | 0.993 | 0.0422 | 0.034 | 2.11 | 1.86 | 0.88 |
| 15 | 150 | 1 | 0.992 | 0.0082 | 0.076 | 5.51 | 5.43 | 0.99 |
| 15 | 270 | 1 | 0.893 | 0.1096 | 0.834 | 12.78 | 11.09 | 0.87 |
| 15 | 450 | 1 | 0.816 | 0.5056 | 1.868 | 17.24 | 9.59 | 0.56 |
| 15 | 500 | 1 | 0.880 | 0.0980 | 0.559 | 10.48 | 10.78 | 1.03 |
| 16 | 10 | 1 | 0.974 | 0.1033 | 0.008 | 0.33 | 0.28 | 0.82 |
| 16 | 63 | 1 | 0.993 | 0.0425 | 0.031 | 2.12 | 1.89 | 0.89 |
| 16 | 150 | 1 | 0.994 | 0.0126 | 0.104 | 5.72 | 5.54 | 0.97 |
| 16 | 270 | 1 | 0.950 | 0.0892 | 0.880 | 16.74 | 13.44 | 0.80 |
| 16 | 450 | 1 | 0.946 | 0.0633 | 0.437 | 10.58 | 9.71 | 0.92 |
| 16 | 500 | 1 | 0.681 | 0.2995 | 1.104 | 11.38 | 14.99 | 1.32 |
| 17 | 10 | 1 | 0.987 | 0.0670 | 0.005 | 0.31 | 0.27 | 0.86 |
| 17 | 63 | 1 | 0.991 | 0.0420 | 0.030 | 2.00 | 1.77 | 0.88 |
| 17 | 150 | 1 | 0.992 | 0.0085 | 0.075 | 4.93 | 4.77 | 0.97 |
| 17 | 270 | 1 | 0.910 | 0.0893 | 0.460 | 16.53 | 14.74 | 0.89 |
| 17 | 450 | 1 | 0.889 | 0.0999 | 0.450 | 11.12 | 10.06 | 0.90 |
| 17 | 500 | 1 | 0.694 | 0.2640 | 0.833 | 11.00 | 12.04 | 1.10 |

## Flags

- [WARN] K=7 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=8 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=10 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=11 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=12 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=13 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=14 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=15 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=16 10 MHz: gen_max=0.26Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=17 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [OK]   K=7 10 MHz anchor: pearson_r=0.983, max_ratio=0.84
- [OK]   K=7 63 MHz anchor: pearson_r=0.995, max_ratio=0.88
- [OK]   K=7 150 MHz anchor: pearson_r=0.934, max_ratio=0.49
- [OK]   K=7 270 MHz anchor: pearson_r=0.974, max_ratio=1.00
- [OK]   K=7 450 MHz anchor: pearson_r=0.875, max_ratio=0.86
- [OK]   K=7 500 MHz anchor: pearson_r=0.891, max_ratio=0.92
- [OK]   K=8 10 MHz anchor: pearson_r=0.989, max_ratio=0.84
- [OK]   K=8 63 MHz anchor: pearson_r=0.993, max_ratio=0.86
- [OK]   K=8 150 MHz anchor: pearson_r=0.984, max_ratio=0.66
- [OK]   K=8 270 MHz anchor: pearson_r=0.972, max_ratio=1.10
- [OK]   K=8 450 MHz anchor: pearson_r=0.952, max_ratio=0.95
- [OK]   K=8 500 MHz anchor: pearson_r=0.926, max_ratio=0.60
- [OK]   K=9 10 MHz anchor: pearson_r=0.977, max_ratio=0.84
- [OK]   K=9 63 MHz anchor: pearson_r=0.994, max_ratio=0.88
- [OK]   K=9 150 MHz anchor: pearson_r=0.993, max_ratio=0.85
- [OK]   K=9 270 MHz anchor: pearson_r=0.971, max_ratio=0.96
- [OK]   K=10 10 MHz anchor: pearson_r=0.961, max_ratio=0.85
- [OK]   K=10 63 MHz anchor: pearson_r=0.991, max_ratio=0.91
- [OK]   K=10 150 MHz anchor: pearson_r=0.993, max_ratio=0.91
- [OK]   K=10 270 MHz anchor: pearson_r=0.928, max_ratio=0.86
- [OK]   K=10 450 MHz anchor: pearson_r=0.905, max_ratio=0.58
- [OK]   K=10 500 MHz anchor: pearson_r=0.927, max_ratio=0.78
- [OK]   K=11 10 MHz anchor: pearson_r=0.969, max_ratio=0.92
- [OK]   K=11 63 MHz anchor: pearson_r=0.993, max_ratio=0.89
- [OK]   K=11 150 MHz anchor: pearson_r=0.991, max_ratio=0.98
- [OK]   K=11 270 MHz anchor: pearson_r=0.890, max_ratio=0.55
- [OK]   K=11 450 MHz anchor: pearson_r=0.976, max_ratio=0.84
- [OK]   K=11 500 MHz anchor: pearson_r=0.979, max_ratio=1.11
- [OK]   K=12 10 MHz anchor: pearson_r=0.973, max_ratio=0.90
- [OK]   K=12 63 MHz anchor: pearson_r=0.992, max_ratio=0.89
- [OK]   K=12 150 MHz anchor: pearson_r=0.989, max_ratio=0.97
- [OK]   K=12 270 MHz anchor: pearson_r=0.952, max_ratio=1.27
- [OK]   K=12 450 MHz anchor: pearson_r=0.960, max_ratio=0.63
- [OK]   K=12 500 MHz anchor: pearson_r=0.967, max_ratio=1.05
- [OK]   K=13 10 MHz anchor: pearson_r=0.980, max_ratio=0.79
- [OK]   K=13 63 MHz anchor: pearson_r=0.989, max_ratio=0.93
- [OK]   K=13 150 MHz anchor: pearson_r=0.997, max_ratio=0.95
- [OK]   K=13 270 MHz anchor: pearson_r=0.858, max_ratio=0.53
- [OK]   K=13 450 MHz anchor: pearson_r=0.888, max_ratio=0.65
- [OK]   K=13 500 MHz anchor: pearson_r=0.957, max_ratio=1.05
- [OK]   K=14 10 MHz anchor: pearson_r=0.977, max_ratio=0.78
- [OK]   K=14 63 MHz anchor: pearson_r=0.992, max_ratio=0.88
- [OK]   K=14 150 MHz anchor: pearson_r=0.996, max_ratio=0.97
- [OK]   K=14 270 MHz anchor: pearson_r=0.960, max_ratio=1.22
- [WARN] K=14 450 MHz anchor: pearson_r=0.632 (<0.75) — weak spatial pattern
- [OK]   K=14 500 MHz anchor: pearson_r=0.908, max_ratio=1.08
- [OK]   K=15 10 MHz anchor: pearson_r=0.979, max_ratio=0.85
- [OK]   K=15 63 MHz anchor: pearson_r=0.993, max_ratio=0.88
- [OK]   K=15 150 MHz anchor: pearson_r=0.992, max_ratio=0.99
- [OK]   K=15 270 MHz anchor: pearson_r=0.893, max_ratio=0.87
- [OK]   K=15 500 MHz anchor: pearson_r=0.880, max_ratio=1.03
- [OK]   K=16 10 MHz anchor: pearson_r=0.974, max_ratio=0.82
- [OK]   K=16 63 MHz anchor: pearson_r=0.993, max_ratio=0.89
- [OK]   K=16 150 MHz anchor: pearson_r=0.994, max_ratio=0.97
- [OK]   K=16 270 MHz anchor: pearson_r=0.950, max_ratio=0.80
- [OK]   K=16 450 MHz anchor: pearson_r=0.946, max_ratio=0.92
- [WARN] K=16 500 MHz anchor: pearson_r=0.681 (<0.75) — weak spatial pattern
- [OK]   K=17 10 MHz anchor: pearson_r=0.987, max_ratio=0.86
- [OK]   K=17 63 MHz anchor: pearson_r=0.991, max_ratio=0.88
- [OK]   K=17 150 MHz anchor: pearson_r=0.992, max_ratio=0.97
- [OK]   K=17 270 MHz anchor: pearson_r=0.910, max_ratio=0.89
- [OK]   K=17 450 MHz anchor: pearson_r=0.889, max_ratio=0.90
- [WARN] K=17 500 MHz anchor: pearson_r=0.694 (<0.75) — weak spatial pattern

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp056_graph_vae
checkpoint: experiments/exp056_graph_vae/checkpoints/checkpoint_epoch_400.pt
mode: layout_qc  layout_source: val
K: 7–17 (11 values)  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 21.28

Per-MHz generation (layout decode, no GT):
  K | MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
  7 |     10 | yes    |    0.29 |    0.07 |    0.13 |     0.21 | yes
  7 |     63 | yes    |    2.04 |    0.58 |    1.05 |     2.91 | no
  7 |    150 | yes    |    7.69 |    2.63 |    5.95 |    13.03 | no
  7 |    270 | yes    |    9.33 |    1.83 |    3.81 |    30.06 | no
  7 |    450 | yes    |    9.56 |    1.81 |    3.91 |    15.12 | no
  7 |    500 | yes    |   13.26 |    3.08 |    5.98 |    16.68 | no
  8 |     10 | yes    |    0.29 |    0.07 |    0.15 |     0.21 | yes
  8 |     63 | yes    |    2.06 |    0.59 |    1.10 |     2.91 | no
  8 |    150 | yes    |    7.24 |    2.41 |    5.58 |    13.03 | no
  8 |    270 | yes    |   12.41 |    1.95 |    4.67 |    30.06 | no
  8 |    450 | yes    |    9.55 |    2.64 |    5.41 |    15.12 | no
  8 |    500 | yes    |   10.77 |    3.08 |    5.63 |    16.68 | no
  9 |     10 | yes    |    0.28 |    0.08 |    0.15 |     0.21 | yes
  9 |     63 | yes    |    1.97 |    0.58 |    1.07 |     2.91 | no
  9 |    150 | yes    |    7.07 |    2.42 |    4.86 |    13.03 | no
  9 |    270 | yes    |   21.33 |    3.73 |   11.05 |    30.06 | no
  9 |    450 | yes    |    7.49 |    1.97 |    4.41 |    15.12 | no
  9 |    500 | yes    |   11.29 |    3.11 |    6.46 |    16.68 | no
 10 |     10 | yes    |    0.29 |    0.07 |    0.15 |     0.21 | yes
 10 |     63 | yes    |    1.98 |    0.52 |    0.99 |     2.91 | no
 10 |    150 | yes    |    6.59 |    1.69 |    3.55 |    13.03 | no
 10 |    270 | yes    |   12.34 |    2.31 |    5.81 |    30.06 | no
 10 |    450 | yes    |    8.12 |    2.41 |    4.34 |    15.12 | no
 10 |    500 | yes    |   11.84 |    2.89 |    5.91 |    16.68 | no
 11 |     10 | yes    |    0.27 |    0.07 |    0.13 |     0.21 | yes
 11 |     63 | yes    |    1.95 |    0.52 |    0.96 |     2.91 | no
 11 |    150 | yes    |    5.97 |    1.60 |    3.03 |    13.03 | no
 11 |    270 | yes    |    8.66 |    2.06 |    4.74 |    30.06 | no
 11 |    450 | yes    |   10.83 |    2.98 |    5.35 |    15.12 | no
 11 |    500 | yes    |   12.08 |    3.06 |    6.75 |    16.68 | no
 12 |     10 | yes    |    0.27 |    0.07 |    0.13 |     0.21 | yes
 12 |     63 | yes    |    1.99 |    0.52 |    1.00 |     2.91 | no
 12 |    150 | yes    |    7.24 |    1.75 |    4.01 |    13.03 | no
 12 |    270 | yes    |   16.69 |    3.21 |    8.09 |    30.06 | no
 12 |    450 | yes    |    7.38 |    2.38 |    4.17 |    15.12 | no
 12 |    500 | yes    |   12.15 |    2.79 |    6.07 |    16.68 | no
 13 |     10 | yes    |    0.27 |    0.06 |    0.13 |     0.21 | yes
 13 |     63 | yes    |    1.93 |    0.49 |    0.93 |     2.91 | no
 13 |    150 | yes    |    6.58 |    1.48 |    3.34 |    13.03 | no
 13 |    270 | yes    |   14.42 |    2.62 |    7.56 |    30.06 | no
 13 |    450 | yes    |    9.47 |    2.29 |    4.12 |    15.12 | no
 13 |    500 | yes    |   10.66 |    2.60 |    4.79 |    16.68 | no
 14 |     10 | yes    |    0.27 |    0.06 |    0.12 |     0.21 | yes
 14 |     63 | yes    |    1.94 |    0.48 |    0.90 |     2.91 | no
 14 |    150 | yes    |    6.09 |    1.35 |    2.89 |    13.03 | no
 14 |    270 | yes    |   13.50 |    2.16 |    4.74 |    30.06 | no
 14 |    450 | yes    |   10.04 |    2.53 |    5.29 |    15.12 | no
 14 |    500 | yes    |   12.05 |    2.62 |    5.09 |    16.68 | no
 15 |     10 | yes    |    0.27 |    0.07 |    0.14 |     0.21 | yes
 15 |     63 | yes    |    1.84 |    0.50 |    0.98 |     2.91 | no
 15 |    150 | yes    |    6.39 |    1.53 |    3.43 |    13.03 | no
 15 |    270 | yes    |    8.09 |    2.30 |    5.59 |    30.06 | no
 15 |    450 | yes    |    9.53 |    2.54 |    4.38 |    15.12 | no
 15 |    500 | yes    |   12.13 |    2.92 |    5.42 |    16.68 | no
 16 |     10 | yes    |    0.26 |    0.06 |    0.12 |     0.21 | yes
 16 |     63 | yes    |    1.82 |    0.47 |    0.88 |     2.91 | no
 16 |    150 | yes    |    4.99 |    1.18 |    2.31 |    13.03 | no
 16 |    270 | yes    |   16.67 |    3.33 |    9.99 |    30.06 | no
 16 |    450 | yes    |    9.30 |    2.40 |    4.32 |    15.12 | no
 16 |    500 | yes    |   10.37 |    2.60 |    5.08 |    16.68 | no
 17 |     10 | yes    |    0.27 |    0.07 |    0.13 |     0.21 | yes
 17 |     63 | yes    |    1.85 |    0.48 |    0.91 |     2.91 | no
 17 |    150 | yes    |    5.34 |    1.27 |    2.54 |    13.03 | no
 17 |    270 | yes    |   13.87 |    2.93 |    6.85 |    30.06 | no
 17 |    450 | yes    |   10.00 |    2.76 |    4.74 |    15.12 | no
 17 |    500 | yes    |   10.36 |    2.68 |    5.26 |    16.68 | no

Layout vs real val heatmap (training anchors only):
  K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio
  7 |   10 | 1 | 0.983 | 0.0760 | 0.006 | 0.33 | 0.28 | 0.84
  7 |   63 | 1 | 0.995 | 0.0480 | 0.052 | 2.35 | 2.06 | 0.88
  7 |  150 | 1 | 0.934 | 1.6301 | 2.425 | 13.61 | 6.72 | 0.49
  7 |  270 | 1 | 0.974 | 0.0577 | 0.612 | 21.00 | 21.04 | 1.00
  7 |  450 | 1 | 0.875 | 0.1202 | 0.404 | 8.57 | 7.34 | 0.86
  7 |  500 | 1 | 0.891 | 0.1524 | 0.805 | 13.78 | 12.73 | 0.92
  8 |   10 | 1 | 0.989 | 0.0757 | 0.007 | 0.35 | 0.29 | 0.84
  8 |   63 | 1 | 0.993 | 0.0530 | 0.058 | 2.31 | 1.99 | 0.86
  8 |  150 | 1 | 0.984 | 0.0808 | 0.432 | 11.81 | 7.80 | 0.66
  8 |  270 | 1 | 0.972 | 0.0550 | 0.655 | 18.45 | 20.31 | 1.10
  8 |  450 | 1 | 0.952 | 0.0458 | 0.205 | 9.21 | 8.74 | 0.95
  8 |  500 | 1 | 0.926 | 0.2773 | 1.895 | 25.12 | 14.98 | 0.60
  9 |   10 | 1 | 0.977 | 0.0777 | 0.007 | 0.34 | 0.29 | 0.84
  9 |   63 | 1 | 0.994 | 0.0496 | 0.054 | 2.32 | 2.03 | 0.88
  9 |  150 | 1 | 0.993 | 0.0178 | 0.141 | 8.80 | 7.47 | 0.85
  9 |  270 | 1 | 0.971 | 0.0357 | 0.417 | 16.31 | 15.60 | 0.96
  9 |  450 | 1 | 0.848 | 0.1122 | 0.484 | 7.42 | 7.85 | 1.06
  9 |  500 | 1 | 0.806 | 0.2134 | 1.024 | 13.97 | 11.88 | 0.85
 10 |   10 | 1 | 0.961 | 0.1148 | 0.008 | 0.33 | 0.29 | 0.85
 10 |   63 | 1 | 0.991 | 0.0460 | 0.043 | 2.15 | 1.97 | 0.91
 10 |  150 | 1 | 0.993 | 0.0117 | 0.112 | 6.92 | 6.27 | 0.91
 10 |  270 | 1 | 0.928 | 0.0996 | 0.994 | 19.43 | 16.77 | 0.86
 10 |  450 | 1 | 0.905 | 0.4678 | 2.281 | 23.38 | 13.58 | 0.58
 10 |  500 | 1 | 0.927 | 0.1829 | 1.337 | 17.88 | 13.94 | 0.78
 11 |   10 | 1 | 0.969 | 0.0937 | 0.008 | 0.31 | 0.28 | 0.92
 11 |   63 | 1 | 0.993 | 0.0475 | 0.044 | 2.21 | 1.96 | 0.89
 11 |  150 | 1 | 0.991 | 0.0198 | 0.142 | 5.99 | 5.88 | 0.98
 11 |  270 | 1 | 0.890 | 0.5033 | 3.450 | 36.48 | 19.97 | 0.55
 11 |  450 | 1 | 0.976 | 0.0505 | 0.481 | 13.35 | 11.22 | 0.84
 11 |  500 | 1 | 0.979 | 0.0165 | 0.253 | 11.86 | 13.16 | 1.11
 12 |   10 | 1 | 0.973 | 0.0890 | 0.007 | 0.29 | 0.26 | 0.90
 12 |   63 | 1 | 0.992 | 0.0426 | 0.032 | 2.19 | 1.95 | 0.89
 12 |  150 | 1 | 0.989 | 0.0140 | 0.124 | 6.82 | 6.61 | 0.97
 12 |  270 | 1 | 0.952 | 0.0386 | 0.215 | 12.45 | 15.83 | 1.27
 12 |  450 | 1 | 0.960 | 0.1206 | 0.945 | 18.37 | 11.66 | 0.63
 12 |  500 | 1 | 0.967 | 0.0276 | 0.303 | 12.36 | 12.92 | 1.05
 13 |   10 | 1 | 0.980 | 0.0733 | 0.006 | 0.32 | 0.25 | 0.79
 13 |   63 | 1 | 0.989 | 0.0449 | 0.036 | 2.01 | 1.86 | 0.93
 13 |  150 | 1 | 0.997 | 0.0064 | 0.079 | 5.70 | 5.44 | 0.95
 13 |  270 | 1 | 0.858 | 0.7577 | 4.059 | 25.15 | 13.22 | 0.53
 13 |  450 | 1 | 0.888 | 0.2159 | 1.221 | 14.98 | 9.67 | 0.65
 13 |  500 | 1 | 0.957 | 0.0350 | 0.298 | 11.32 | 11.84 | 1.05
 14 |   10 | 1 | 0.977 | 0.1215 | 0.010 | 0.35 | 0.27 | 0.78
 14 |   63 | 1 | 0.992 | 0.0413 | 0.032 | 2.20 | 1.94 | 0.88
 14 |  150 | 1 | 0.996 | 0.0074 | 0.082 | 6.14 | 5.98 | 0.97
 14 |  270 | 1 | 0.960 | 0.0543 | 0.348 | 13.38 | 16.28 | 1.22
 14 |  450 | 1 | 0.632 | 0.2662 | 0.741 | 7.10 | 8.38 | 1.18
 14 |  500 | 1 | 0.908 | 0.0969 | 0.444 | 11.46 | 12.39 | 1.08
 15 |   10 | 1 | 0.979 | 0.0750 | 0.006 | 0.33 | 0.28 | 0.85
 15 |   63 | 1 | 0.993 | 0.0422 | 0.034 | 2.11 | 1.86 | 0.88
 15 |  150 | 1 | 0.992 | 0.0082 | 0.076 | 5.51 | 5.43 | 0.99
 15 |  270 | 1 | 0.893 | 0.1096 | 0.834 | 12.78 | 11.09 | 0.87
 15 |  450 | 1 | 0.816 | 0.5056 | 1.868 | 17.24 | 9.59 | 0.56
 15 |  500 | 1 | 0.880 | 0.0980 | 0.559 | 10.48 | 10.78 | 1.03
 16 |   10 | 1 | 0.974 | 0.1033 | 0.008 | 0.33 | 0.28 | 0.82
 16 |   63 | 1 | 0.993 | 0.0425 | 0.031 | 2.12 | 1.89 | 0.89
 16 |  150 | 1 | 0.994 | 0.0126 | 0.104 | 5.72 | 5.54 | 0.97
 16 |  270 | 1 | 0.950 | 0.0892 | 0.880 | 16.74 | 13.44 | 0.80
 16 |  450 | 1 | 0.946 | 0.0633 | 0.437 | 10.58 | 9.71 | 0.92
 16 |  500 | 1 | 0.681 | 0.2995 | 1.104 | 11.38 | 14.99 | 1.32
 17 |   10 | 1 | 0.987 | 0.0670 | 0.005 | 0.31 | 0.27 | 0.86
 17 |   63 | 1 | 0.991 | 0.0420 | 0.030 | 2.00 | 1.77 | 0.88
 17 |  150 | 1 | 0.992 | 0.0085 | 0.075 | 4.93 | 4.77 | 0.97
 17 |  270 | 1 | 0.910 | 0.0893 | 0.460 | 16.53 | 14.74 | 0.89
 17 |  450 | 1 | 0.889 | 0.0999 | 0.450 | 11.12 | 10.06 | 0.90
 17 |  500 | 1 | 0.694 | 0.2640 | 0.833 | 11.00 | 12.04 | 1.10

Flags:
[WARN] K=7 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=8 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 10 MHz: gen_max=0.28Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=10 10 MHz: gen_max=0.29Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=11 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=12 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=13 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=14 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=15 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=16 10 MHz: gen_max=0.26Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=17 10 MHz: gen_max=0.27Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[OK]   K=7 10 MHz anchor: pearson_r=0.983, max_ratio=0.84
[OK]   K=7 63 MHz anchor: pearson_r=0.995, max_ratio=0.88
[OK]   K=7 150 MHz anchor: pearson_r=0.934, max_ratio=0.49
[OK]   K=7 270 MHz anchor: pearson_r=0.974, max_ratio=1.00
[OK]   K=7 450 MHz anchor: pearson_r=0.875, max_ratio=0.86
[OK]   K=7 500 MHz anchor: pearson_r=0.891, max_ratio=0.92
[OK]   K=8 10 MHz anchor: pearson_r=0.989, max_ratio=0.84
[OK]   K=8 63 MHz anchor: pearson_r=0.993, max_ratio=0.86
[OK]   K=8 150 MHz anchor: pearson_r=0.984, max_ratio=0.66
[OK]   K=8 270 MHz anchor: pearson_r=0.972, max_ratio=1.10
[OK]   K=8 450 MHz anchor: pearson_r=0.952, max_ratio=0.95
[OK]   K=8 500 MHz anchor: pearson_r=0.926, max_ratio=0.60
[OK]   K=9 10 MHz anchor: pearson_r=0.977, max_ratio=0.84
[OK]   K=9 63 MHz anchor: pearson_r=0.994, max_ratio=0.88
[OK]   K=9 150 MHz anchor: pearson_r=0.993, max_ratio=0.85
[OK]   K=9 270 MHz anchor: pearson_r=0.971, max_ratio=0.96
[OK]   K=10 10 MHz anchor: pearson_r=0.961, max_ratio=0.85
[OK]   K=10 63 MHz anchor: pearson_r=0.991, max_ratio=0.91
[OK]   K=10 150 MHz anchor: pearson_r=0.993, max_ratio=0.91
[OK]   K=10 270 MHz anchor: pearson_r=0.928, max_ratio=0.86
[OK]   K=10 450 MHz anchor: pearson_r=0.905, max_ratio=0.58
[OK]   K=10 500 MHz anchor: pearson_r=0.927, max_ratio=0.78
[OK]   K=11 10 MHz anchor: pearson_r=0.969, max_ratio=0.92
[OK]   K=11 63 MHz anchor: pearson_r=0.993, max_ratio=0.89
[OK]   K=11 150 MHz anchor: pearson_r=0.991, max_ratio=0.98
[OK]   K=11 270 MHz anchor: pearson_r=0.890, max_ratio=0.55
[OK]   K=11 450 MHz anchor: pearson_r=0.976, max_ratio=0.84
[OK]   K=11 500 MHz anchor: pearson_r=0.979, max_ratio=1.11
[OK]   K=12 10 MHz anchor: pearson_r=0.973, max_ratio=0.90
[OK]   K=12 63 MHz anchor: pearson_r=0.992, max_ratio=0.89
[OK]   K=12 150 MHz anchor: pearson_r=0.989, max_ratio=0.97
[OK]   K=12 270 MHz anchor: pearson_r=0.952, max_ratio=1.27
[OK]   K=12 450 MHz anchor: pearson_r=0.960, max_ratio=0.63
[OK]   K=12 500 MHz anchor: pearson_r=0.967, max_ratio=1.05
[OK]   K=13 10 MHz anchor: pearson_r=0.980, max_ratio=0.79
[OK]   K=13 63 MHz anchor: pearson_r=0.989, max_ratio=0.93
[OK]   K=13 150 MHz anchor: pearson_r=0.997, max_ratio=0.95
[OK]   K=13 270 MHz anchor: pearson_r=0.858, max_ratio=0.53
[OK]   K=13 450 MHz anchor: pearson_r=0.888, max_ratio=0.65
[OK]   K=13 500 MHz anchor: pearson_r=0.957, max_ratio=1.05
[OK]   K=14 10 MHz anchor: pearson_r=0.977, max_ratio=0.78
[OK]   K=14 63 MHz anchor: pearson_r=0.992, max_ratio=0.88
[OK]   K=14 150 MHz anchor: pearson_r=0.996, max_ratio=0.97
[OK]   K=14 270 MHz anchor: pearson_r=0.960, max_ratio=1.22
[WARN] K=14 450 MHz anchor: pearson_r=0.632 (<0.75) — weak spatial pattern
[OK]   K=14 500 MHz anchor: pearson_r=0.908, max_ratio=1.08
[OK]   K=15 10 MHz anchor: pearson_r=0.979, max_ratio=0.85
[OK]   K=15 63 MHz anchor: pearson_r=0.993, max_ratio=0.88
[OK]   K=15 150 MHz anchor: pearson_r=0.992, max_ratio=0.99
[OK]   K=15 270 MHz anchor: pearson_r=0.893, max_ratio=0.87
[OK]   K=15 500 MHz anchor: pearson_r=0.880, max_ratio=1.03
[OK]   K=16 10 MHz anchor: pearson_r=0.974, max_ratio=0.82
[OK]   K=16 63 MHz anchor: pearson_r=0.993, max_ratio=0.89
[OK]   K=16 150 MHz anchor: pearson_r=0.994, max_ratio=0.97
[OK]   K=16 270 MHz anchor: pearson_r=0.950, max_ratio=0.80
[OK]   K=16 450 MHz anchor: pearson_r=0.946, max_ratio=0.92
[WARN] K=16 500 MHz anchor: pearson_r=0.681 (<0.75) — weak spatial pattern
[OK]   K=17 10 MHz anchor: pearson_r=0.987, max_ratio=0.86
[OK]   K=17 63 MHz anchor: pearson_r=0.991, max_ratio=0.88
[OK]   K=17 150 MHz anchor: pearson_r=0.992, max_ratio=0.97
[OK]   K=17 270 MHz anchor: pearson_r=0.910, max_ratio=0.89
[OK]   K=17 450 MHz anchor: pearson_r=0.889, max_ratio=0.90
[WARN] K=17 500 MHz anchor: pearson_r=0.694 (<0.75) — weak spatial pattern
=== END SWEEP QC ===
```
