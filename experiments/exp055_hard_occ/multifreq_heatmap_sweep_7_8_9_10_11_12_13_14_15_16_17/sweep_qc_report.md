# Sweep QC Report

- **Experiment:** `experiments/exp055_hard_occ`
- **Checkpoint:** `experiments/exp055_hard_occ/checkpoints/checkpoint_epoch_400.pt`
- **Mode:** `layout_qc` / `val`
- **K:** 7–17 (11 values) | **PI ref:** 200.0 MHz
- **Clip ceiling @ 200 MHz (approx):** 21.28 Ω

## Per-MHz generation

| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |
|---|-----|------------------|-----------|---------|-----|------------|
| 7 | 20 | no | 0.24 | 0.07 | 0.13 | yes |
| 7 | 50 | no | 2.43 | 0.66 | 1.24 | no |
| 7 | 140 | no | 7.04 | 1.86 | 4.59 | no |
| 7 | 260 | no | 10.75 | 2.49 | 5.26 | no |
| 7 | 440 | no | 12.46 | 2.87 | 5.53 | no |
| 7 | 530 | no | 14.21 | 3.14 | 6.74 | no |
| 8 | 20 | no | 0.24 | 0.08 | 0.14 | yes |
| 8 | 50 | no | 2.47 | 0.69 | 1.29 | no |
| 8 | 140 | no | 7.70 | 2.02 | 4.50 | no |
| 8 | 260 | no | 14.65 | 2.96 | 8.32 | no |
| 8 | 440 | no | 9.66 | 2.36 | 4.53 | no |
| 8 | 530 | no | 12.88 | 2.92 | 5.84 | no |
| 9 | 20 | no | 0.21 | 0.08 | 0.14 | yes |
| 9 | 50 | no | 2.25 | 0.68 | 1.23 | no |
| 9 | 140 | no | 6.25 | 2.02 | 3.88 | no |
| 9 | 260 | no | 17.49 | 3.33 | 9.10 | no |
| 9 | 440 | no | 11.12 | 3.24 | 5.74 | no |
| 9 | 530 | no | 12.55 | 3.12 | 6.62 | no |
| 10 | 20 | no | 0.23 | 0.07 | 0.13 | yes |
| 10 | 50 | no | 2.30 | 0.62 | 1.15 | no |
| 10 | 140 | no | 6.90 | 1.79 | 3.66 | no |
| 10 | 260 | no | 10.50 | 2.78 | 6.29 | no |
| 10 | 440 | no | 10.93 | 2.73 | 4.83 | no |
| 10 | 530 | no | 12.98 | 2.85 | 6.05 | no |
| 11 | 20 | no | 0.21 | 0.07 | 0.12 | yes |
| 11 | 50 | no | 2.16 | 0.62 | 1.10 | no |
| 11 | 140 | no | 7.21 | 1.87 | 3.68 | no |
| 11 | 260 | no | 16.63 | 2.92 | 9.59 | no |
| 11 | 440 | no | 11.71 | 3.01 | 5.63 | no |
| 11 | 530 | no | 14.01 | 3.06 | 6.02 | no |
| 12 | 20 | no | 0.23 | 0.07 | 0.13 | yes |
| 12 | 50 | no | 2.33 | 0.60 | 1.13 | no |
| 12 | 140 | no | 7.67 | 1.81 | 3.79 | no |
| 12 | 260 | no | 11.21 | 2.37 | 5.58 | no |
| 12 | 440 | no | 12.73 | 3.17 | 5.94 | no |
| 12 | 530 | no | 12.86 | 3.05 | 6.89 | no |
| 13 | 20 | no | 0.20 | 0.07 | 0.12 | yes |
| 13 | 50 | no | 2.06 | 0.56 | 1.06 | no |
| 13 | 140 | no | 6.49 | 1.53 | 3.31 | no |
| 13 | 260 | no | 15.25 | 2.74 | 7.46 | no |
| 13 | 440 | no | 9.54 | 2.09 | 4.09 | no |
| 13 | 530 | no | 12.05 | 3.14 | 6.45 | no |
| 14 | 20 | no | 0.22 | 0.06 | 0.12 | yes |
| 14 | 50 | no | 2.31 | 0.57 | 1.09 | no |
| 14 | 140 | no | 6.49 | 1.57 | 3.32 | no |
| 14 | 260 | no | 14.60 | 2.62 | 6.82 | no |
| 14 | 440 | no | 9.67 | 2.42 | 4.67 | no |
| 14 | 530 | no | 12.16 | 2.84 | 5.61 | no |
| 15 | 20 | no | 0.21 | 0.07 | 0.13 | yes |
| 15 | 50 | no | 2.21 | 0.59 | 1.16 | no |
| 15 | 140 | no | 6.18 | 1.57 | 3.39 | no |
| 15 | 260 | no | 17.58 | 3.47 | 11.55 | no |
| 15 | 440 | no | 11.15 | 2.97 | 5.27 | no |
| 15 | 530 | no | 14.70 | 3.38 | 7.42 | no |
| 16 | 20 | no | 0.21 | 0.06 | 0.11 | yes |
| 16 | 50 | no | 2.14 | 0.54 | 1.01 | no |
| 16 | 140 | no | 5.66 | 1.37 | 2.80 | no |
| 16 | 260 | no | 20.64 | 3.82 | 11.08 | no |
| 16 | 440 | no | 11.80 | 2.69 | 5.42 | no |
| 16 | 530 | no | 13.06 | 3.14 | 6.19 | no |
| 17 | 20 | no | 0.20 | 0.07 | 0.12 | yes |
| 17 | 50 | no | 2.09 | 0.57 | 1.08 | no |
| 17 | 140 | no | 5.63 | 1.42 | 2.87 | no |
| 17 | 260 | no | 20.45 | 4.26 | 11.86 | no |
| 17 | 440 | no | 12.09 | 3.21 | 5.47 | no |
| 17 | 530 | no | 12.70 | 2.94 | 6.43 | no |

## Flags

- [WARN] K=7 20 MHz: gen_max=0.24Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=8 20 MHz: gen_max=0.24Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=9 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=10 20 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=11 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=12 20 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=13 20 MHz: gen_max=0.20Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=14 20 MHz: gen_max=0.22Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=15 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=16 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
- [WARN] K=17 20 MHz: gen_max=0.20Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation

## Copy block for agent

```
=== SWEEP QC (copy to agent) ===
experiment: experiments/exp055_hard_occ
checkpoint: experiments/exp055_hard_occ/checkpoints/checkpoint_epoch_400.pt
mode: layout_qc  layout_source: val
K: 7–17 (11 values)  samples: 1  pi_ref: 200.0 MHz
z_clip_ceiling_ohm@200MHz: 21.28

Per-MHz generation (layout decode, no GT):
  K | MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip
  7 |     20 | no     |    0.24 |    0.07 |    0.13 |     0.21 | yes
  7 |     50 | no     |    2.43 |    0.66 |    1.24 |     2.91 | no
  7 |    140 | no     |    7.04 |    1.86 |    4.59 |    13.03 | no
  7 |    260 | no     |   10.75 |    2.49 |    5.26 |    26.62 | no
  7 |    440 | no     |   12.46 |    2.87 |    5.53 |    16.19 | no
  7 |    530 | no     |   14.21 |    3.14 |    6.74 |    16.68 | no
  8 |     20 | no     |    0.24 |    0.08 |    0.14 |     0.21 | yes
  8 |     50 | no     |    2.47 |    0.69 |    1.29 |     2.91 | no
  8 |    140 | no     |    7.70 |    2.02 |    4.50 |    13.03 | no
  8 |    260 | no     |   14.65 |    2.96 |    8.32 |    26.62 | no
  8 |    440 | no     |    9.66 |    2.36 |    4.53 |    16.19 | no
  8 |    530 | no     |   12.88 |    2.92 |    5.84 |    16.68 | no
  9 |     20 | no     |    0.21 |    0.08 |    0.14 |     0.21 | yes
  9 |     50 | no     |    2.25 |    0.68 |    1.23 |     2.91 | no
  9 |    140 | no     |    6.25 |    2.02 |    3.88 |    13.03 | no
  9 |    260 | no     |   17.49 |    3.33 |    9.10 |    26.62 | no
  9 |    440 | no     |   11.12 |    3.24 |    5.74 |    16.19 | no
  9 |    530 | no     |   12.55 |    3.12 |    6.62 |    16.68 | no
 10 |     20 | no     |    0.23 |    0.07 |    0.13 |     0.21 | yes
 10 |     50 | no     |    2.30 |    0.62 |    1.15 |     2.91 | no
 10 |    140 | no     |    6.90 |    1.79 |    3.66 |    13.03 | no
 10 |    260 | no     |   10.50 |    2.78 |    6.29 |    26.62 | no
 10 |    440 | no     |   10.93 |    2.73 |    4.83 |    16.19 | no
 10 |    530 | no     |   12.98 |    2.85 |    6.05 |    16.68 | no
 11 |     20 | no     |    0.21 |    0.07 |    0.12 |     0.21 | yes
 11 |     50 | no     |    2.16 |    0.62 |    1.10 |     2.91 | no
 11 |    140 | no     |    7.21 |    1.87 |    3.68 |    13.03 | no
 11 |    260 | no     |   16.63 |    2.92 |    9.59 |    26.62 | no
 11 |    440 | no     |   11.71 |    3.01 |    5.63 |    16.19 | no
 11 |    530 | no     |   14.01 |    3.06 |    6.02 |    16.68 | no
 12 |     20 | no     |    0.23 |    0.07 |    0.13 |     0.21 | yes
 12 |     50 | no     |    2.33 |    0.60 |    1.13 |     2.91 | no
 12 |    140 | no     |    7.67 |    1.81 |    3.79 |    13.03 | no
 12 |    260 | no     |   11.21 |    2.37 |    5.58 |    26.62 | no
 12 |    440 | no     |   12.73 |    3.17 |    5.94 |    16.19 | no
 12 |    530 | no     |   12.86 |    3.05 |    6.89 |    16.68 | no
 13 |     20 | no     |    0.20 |    0.07 |    0.12 |     0.21 | yes
 13 |     50 | no     |    2.06 |    0.56 |    1.06 |     2.91 | no
 13 |    140 | no     |    6.49 |    1.53 |    3.31 |    13.03 | no
 13 |    260 | no     |   15.25 |    2.74 |    7.46 |    26.62 | no
 13 |    440 | no     |    9.54 |    2.09 |    4.09 |    16.19 | no
 13 |    530 | no     |   12.05 |    3.14 |    6.45 |    16.68 | no
 14 |     20 | no     |    0.22 |    0.06 |    0.12 |     0.21 | yes
 14 |     50 | no     |    2.31 |    0.57 |    1.09 |     2.91 | no
 14 |    140 | no     |    6.49 |    1.57 |    3.32 |    13.03 | no
 14 |    260 | no     |   14.60 |    2.62 |    6.82 |    26.62 | no
 14 |    440 | no     |    9.67 |    2.42 |    4.67 |    16.19 | no
 14 |    530 | no     |   12.16 |    2.84 |    5.61 |    16.68 | no
 15 |     20 | no     |    0.21 |    0.07 |    0.13 |     0.21 | yes
 15 |     50 | no     |    2.21 |    0.59 |    1.16 |     2.91 | no
 15 |    140 | no     |    6.18 |    1.57 |    3.39 |    13.03 | no
 15 |    260 | no     |   17.58 |    3.47 |   11.55 |    26.62 | no
 15 |    440 | no     |   11.15 |    2.97 |    5.27 |    16.19 | no
 15 |    530 | no     |   14.70 |    3.38 |    7.42 |    16.68 | no
 16 |     20 | no     |    0.21 |    0.06 |    0.11 |     0.21 | yes
 16 |     50 | no     |    2.14 |    0.54 |    1.01 |     2.91 | no
 16 |    140 | no     |    5.66 |    1.37 |    2.80 |    13.03 | no
 16 |    260 | no     |   20.64 |    3.82 |   11.08 |    26.62 | no
 16 |    440 | no     |   11.80 |    2.69 |    5.42 |    16.19 | no
 16 |    530 | no     |   13.06 |    3.14 |    6.19 |    16.68 | no
 17 |     20 | no     |    0.20 |    0.07 |    0.12 |     0.21 | yes
 17 |     50 | no     |    2.09 |    0.57 |    1.08 |     2.91 | no
 17 |    140 | no     |    5.63 |    1.42 |    2.87 |    13.03 | no
 17 |    260 | no     |   20.45 |    4.26 |   11.86 |    26.62 | no
 17 |    440 | no     |   12.09 |    3.21 |    5.47 |    16.19 | no
 17 |    530 | no     |   12.70 |    2.94 |    6.43 |    16.68 | no

Layout vs real: skipped (not layout_qc or no anchor val samples).

Flags:
[WARN] K=7 20 MHz: gen_max=0.24Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=8 20 MHz: gen_max=0.24Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=9 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=10 20 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=11 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=12 20 MHz: gen_max=0.23Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=13 20 MHz: gen_max=0.20Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=14 20 MHz: gen_max=0.22Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=15 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=16 20 MHz: gen_max=0.21Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
[WARN] K=17 20 MHz: gen_max=0.20Ω (≥90% per-MHz clip ceiling 0.21Ω) — magnitude saturation
=== END SWEEP QC ===
```
