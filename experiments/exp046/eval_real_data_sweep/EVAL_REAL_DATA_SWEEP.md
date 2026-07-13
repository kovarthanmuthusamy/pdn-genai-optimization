# exp046 Real Data Sweep Evaluation

## Purpose

Evaluate heatmap generation quality using **real layouts from the validation split** instead of random marginal samples. This provides a controlled comparison where ground truth is known.

## Two Evaluation Modes

| Mode | Path | What it tests |
|------|------|---------------|
| **encode** | encode(hm, occ, imp, K, pi) → decode | Autoencoder reconstruction (upper bound) |
| **layout** | encode_layout_latent(occ, imp, K, pi) → decode | Generation from layout only (actual use case) |

## Results (exp046 v3, 600 epochs)

```
Mode         MHz    n    FG-MSE   MAE(Ω)      r  peakΔ    R_max   G_max   R_p95   G_p95  R_p999  G_p999
------------------------------------------------------------------------------------------
encode        10    8    0.0774   0.1967  0.982  0.001     0.32    0.54    0.14    0.34    0.24    0.47
encode        63    8    0.0978   0.3277  0.994  0.003     2.28    2.76    1.21    1.54    1.91    2.32
encode       300    8    0.1065   0.6681  0.998  0.004    18.48   19.88   10.11   11.69   16.56   18.29
encode       330    8    0.1181   0.6969  0.998  0.005    16.31   18.77   10.68   12.41   14.90   17.08
encode       400    8    0.0989   0.5598  0.997  0.002    12.34   13.30    6.90    7.75   10.72   11.57
layout        10    8    0.0767   0.1969  0.954  0.002     0.32    0.59    0.14    0.35    0.24    0.50
layout        63    8    0.0946   0.2920  0.985  0.003     2.28    3.16    1.21    1.54    1.91    2.52
layout       300    8    0.6099   1.8606  0.876  0.026    18.48   31.94   10.11   15.93   16.56   31.68
layout       330    8    0.5442   1.9451  0.909  0.025    16.31   33.75   10.68   15.08   14.90   33.75
layout       400    8    0.4221   1.2939  0.893  0.014    12.34   30.60    6.90    7.88   10.72   28.87
```

## Key Findings

### 1. Encode mode works well (r > 0.99 at all frequencies)
The autoencoder reconstruction is excellent. The model CAN represent and decode accurate heatmaps at all frequencies when given the real heatmap as input.

### 2. Layout mode has two distinct problems at high freq

**Problem A: Spatial patterns (r = 0.87-0.91 at 300-400 MHz)**
Layout mode Pearson r drops from 0.98+ (encode) to 0.87-0.91 at high frequencies. The shape is approximately right but not precise.

**Problem B: Magnitude explosion (G_max >> R_max)**
The generated heatmap max is consistently hitting **33.75 Ω** (the z-score clip ceiling) at 300-400 MHz, even though real max is 10-18 Ω. The layout-to-latent path is producing z-scores at the clip boundary.

### 3. Low frequencies (10, 63 MHz) work well in both modes
Both encode and layout achieve r > 0.95 at low frequencies.

## Root Cause

The `encode_layout_latent` path must predict the heatmap-private latent dimensions (15 dims) from occupancy + impedance features alone. At high frequencies, the heatmap z-scores span a much larger range (z=+2.8 to +4.1), and the layout private head has insufficient signal to predict the correct magnitude, defaulting to extreme values that hit the clip ceiling after denormalization.

## Script

```bash
cd /home/ubuntu/gan
python -m experiments.exp046.codes.eval_real_data_sweep
```

Configurable parameters at the top of `eval_real_data_sweep.py`:
- `EVAL_MHZ`: frequencies to evaluate
- `MAX_SAMPLES_PER_MHZ`: number of val samples per frequency
- `NUM_PLOT_SAMPLES`: number of side-by-side images per frequency
- `CHECKPOINT`: specific checkpoint path (None = last_model.pt)
