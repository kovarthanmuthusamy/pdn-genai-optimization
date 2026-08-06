# Sweep QC Report

- **Experiment:** `experiments/exp059_capacity_freq`
- **Checkpoint:** `experiments/exp059_capacity_freq/checkpoints/last_model.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 3–10 (8 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 3 | 30 | no | 0.12 | 0.05 | 0.09 | no |
| 3 | 85 | no | 2.52 | 0.85 | 1.35 | no |
| 3 | 160 | no | 5.34 | 1.19 | 2.75 | no |
| 3 | 260 | no | 15.86 | 3.31 | 12.96 | no |
| 3 | 480 | no | 14.15 | 3.83 | 7.89 | yes |
| 3 | 530 | no | 15.65 | 3.80 | 7.92 | yes |
| 4 | 30 | no | 0.19 | 0.11 | 0.15 | yes |
| 4 | 85 | no | 3.25 | 1.37 | 2.11 | no |
| 4 | 160 | no | 5.28 | 1.25 | 2.77 | no |
| 4 | 260 | no | 24.34 | 4.59 | 18.72 | yes |
| 4 | 480 | no | 9.98 | 3.21 | 6.55 | no |
| 4 | 530 | no | 9.97 | 3.22 | 6.52 | no |
| 5 | 30 | no | 0.21 | 0.12 | 0.17 | yes |
| 5 | 85 | no | 3.74 | 1.26 | 2.20 | no |
| 5 | 160 | no | 6.36 | 1.44 | 3.52 | no |
| 5 | 260 | no | 11.29 | 1.96 | 5.19 | no |
| 5 | 480 | no | 11.39 | 3.27 | 6.90 | no |
| 5 | 530 | no | 11.40 | 2.93 | 6.37 | no |
| 6 | 30 | no | 0.19 | 0.08 | 0.13 | no |
| 6 | 85 | no | 3.14 | 1.05 | 1.83 | no |
| 6 | 160 | no | 4.78 | 1.17 | 2.61 | no |
| 6 | 260 | no | 22.30 | 4.73 | 17.61 | no |
| 6 | 480 | no | 12.07 | 3.57 | 6.91 | no |
| 6 | 530 | no | 13.71 | 3.63 | 7.32 | no |
| 7 | 30 | no | 0.19 | 0.08 | 0.12 | no |
| 7 | 85 | no | 3.35 | 1.01 | 1.84 | no |
| 7 | 160 | no | 4.68 | 0.91 | 2.04 | no |
| 7 | 260 | no | 8.62 | 1.77 | 4.93 | no |
| 7 | 480 | no | 10.36 | 2.72 | 5.50 | no |
| 7 | 530 | no | 10.74 | 2.64 | 5.65 | no |
| 8 | 30 | no | 0.19 | 0.08 | 0.12 | no |
| 8 | 85 | no | 3.00 | 0.88 | 1.62 | no |
| 8 | 160 | no | 4.90 | 1.56 | 3.57 | no |
| 8 | 260 | no | 12.29 | 2.24 | 6.74 | no |
| 8 | 480 | no | 11.19 | 2.99 | 6.14 | no |
| 8 | 530 | no | 12.06 | 3.05 | 6.45 | no |
| 9 | 30 | no | 0.17 | 0.07 | 0.12 | no |
| 9 | 85 | no | 2.73 | 0.88 | 1.60 | no |
| 9 | 160 | no | 5.69 | 1.98 | 3.48 | no |
| 9 | 260 | no | 24.06 | 4.40 | 14.93 | yes |
| 9 | 480 | no | 11.70 | 3.35 | 7.38 | no |
| 9 | 530 | no | 8.98 | 2.97 | 5.37 | no |
| 10 | 30 | no | 0.18 | 0.07 | 0.11 | no |
| 10 | 85 | no | 2.88 | 0.86 | 1.60 | no |
| 10 | 160 | no | 7.06 | 1.95 | 4.04 | no |
| 10 | 260 | no | 18.05 | 2.53 | 7.73 | no |
| 10 | 480 | no | 9.46 | 2.38 | 4.81 | no |
| 10 | 530 | no | 14.91 | 4.52 | 8.81 | no |

## Flags

- [WARN] K=3 480 MHz: gen_max=14.15Ω (≥90% per-MHz clip ceiling 14.29Ω) — magnitude saturation
- [WARN] K=3 530 MHz: gen_max=15.65Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
- [WARN] K=4 30 MHz: gen_max=0.19Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=4 260 MHz: gen_max=24.34Ω (≥90% per-MHz clip ceiling 26.62Ω) — magnitude saturation
- [WARN] K=5 30 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 260 MHz: gen_max=24.06Ω (≥90% per-MHz clip ceiling 26.62Ω) — magnitude saturation

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
  3 |     30 | no     |    0.12 |    0.05 |    0.09 |     0.21 | no
  3 |     85 | no     |    2.52 |    0.85 |    1.35 |     9.78 | no
  3 |    160 | no     |    5.34 |    1.19 |    2.75 |    13.03 | no
  3 |    260 | no     |   15.86 |    3.31 |   12.96 |    26.62 | no
  3 |    480 | no     |   14.15 |    3.83 |    7.89 |    14.29 | yes
  3 |    530 | no     |   15.65 |    3.80 |    7.92 |    16.68 | yes
  4 |     30 | no     |    0.19 |    0.11 |    0.15 |     0.21 | yes
  4 |     85 | no     |    3.25 |    1.37 |    2.11 |     9.78 | no
  4 |    160 | no     |    5.28 |    1.25 |    2.77 |    13.03 | no
  4 |    260 | no     |   24.34 |    4.59 |   18.72 |    26.62 | yes
  4 |    480 | no     |    9.98 |    3.21 |    6.55 |    14.29 | no
  4 |    530 | no     |    9.97 |    3.22 |    6.52 |    16.68 | no
  5 |     30 | no     |    0.21 |    0.12 |    0.17 |     0.21 | yes
  5 |     85 | no     |    3.74 |    1.26 |    2.20 |     9.78 | no
  5 |    160 | no     |    6.36 |    1.44 |    3.52 |    13.03 | no
  5 |    260 | no     |   11.29 |    1.96 |    5.19 |    26.62 | no
  5 |    480 | no     |   11.39 |    3.27 |    6.90 |    14.29 | no
  5 |    530 | no     |   11.40 |    2.93 |    6.37 |    16.68 | no
  6 |     30 | no     |    0.19 |    0.08 |    0.13 |     0.21 | no
  6 |     85 | no     |    3.14 |    1.05 |    1.83 |     9.78 | no
  6 |    160 | no     |    4.78 |    1.17 |    2.61 |    13.03 | no
  6 |    260 | no     |   22.30 |    4.73 |   17.61 |    26.62 | no
  6 |    480 | no     |   12.07 |    3.57 |    6.91 |    14.29 | no
  6 |    530 | no     |   13.71 |    3.63 |    7.32 |    16.68 | no
  7 |     30 | no     |    0.19 |    0.08 |    0.12 |     0.21 | no
  7 |     85 | no     |    3.35 |    1.01 |    1.84 |     9.78 | no
  7 |    160 | no     |    4.68 |    0.91 |    2.04 |    13.03 | no
  7 |    260 | no     |    8.62 |    1.77 |    4.93 |    26.62 | no
  7 |    480 | no     |   10.36 |    2.72 |    5.50 |    14.29 | no
  7 |    530 | no     |   10.74 |    2.64 |    5.65 |    16.68 | no
  8 |     30 | no     |    0.19 |    0.08 |    0.12 |     0.21 | no
  8 |     85 | no     |    3.00 |    0.88 |    1.62 |     9.78 | no
  8 |    160 | no     |    4.90 |    1.56 |    3.57 |    13.03 | no
  8 |    260 | no     |   12.29 |    2.24 |    6.74 |    26.62 | no
  8 |    480 | no     |   11.19 |    2.99 |    6.14 |    14.29 | no
  8 |    530 | no     |   12.06 |    3.05 |    6.45 |    16.68 | no
  9 |     30 | no     |    0.17 |    0.07 |    0.12 |     0.21 | no
  9 |     85 | no     |    2.73 |    0.88 |    1.60 |     9.78 | no
  9 |    160 | no     |    5.69 |    1.98 |    3.48 |    13.03 | no
  9 |    260 | no     |   24.06 |    4.40 |   14.93 |    26.62 | yes
  9 |    480 | no     |   11.70 |    3.35 |    7.38 |    14.29 | no
  9 |    530 | no     |    8.98 |    2.97 |    5.37 |    16.68 | no
 10 |     30 | no     |    0.18 |    0.07 |    0.11 |     0.21 | no
 10 |     85 | no     |    2.88 |    0.86 |    1.60 |     9.78 | no
 10 |    160 | no     |    7.06 |    1.95 |    4.04 |    13.03 | no
 10 |    260 | no     |   18.05 |    2.53 |    7.73 |    26.62 | no
 10 |    480 | no     |    9.46 |    2.38 |    4.81 |    14.29 | no
 10 |    530 | no     |   14.91 |    4.52 |    8.81 |    16.68 | no

Layout vs real: skipped (not layout_qc or no anchor val samples).

Flags:
[WARN] K=3 480 MHz: gen_max=14.15Ω (≥90% per-MHz clip ceiling 14.29Ω) — magnitude saturation
[WARN] K=3 530 MHz: gen_max=15.65Ω (≥90% per-MHz clip ceiling 16.68Ω) — magnitude saturation
[WARN] K=4 30 MHz: gen_max=0.19Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=4 260 MHz: gen_max=24.34Ω (≥90% per-MHz clip ceiling 26.62Ω) — magnitude saturation
[WARN] K=5 30 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 260 MHz: gen_max=24.06Ω (≥90% per-MHz clip ceiling 26.62Ω) — magnitude saturation
=== END SWEEP QC ===
```
