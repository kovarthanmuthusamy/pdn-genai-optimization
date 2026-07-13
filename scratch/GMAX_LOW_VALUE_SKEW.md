# Gmax dataset low-value skew check

Script: `scratch/check_gmax_low_value_skew.py`  
Report: `scratch/gmax_low_value_skew_report.json`

## Question

Are normalized heatmaps (`data_multifreq_gmax`) skewed in the lowest values?

## Answer: **Yes — strongly right-skewed**

Sampled **1,920 maps** (120 per MHz × 16 bins).

| Metric | Value |
|--------|-------|
| FG pixel skew (linear norm) | **4.85** |
| Exact-zero pixels (mean per map) | **18%** |
| FG mass in norm [0.02, 0.05) | **41.5%** |
| FG mass in norm [0.05, 0.10) | **20.7%** |

Linear `phys / global_max` compresses most foreground into **0.02–0.10** norm (≈1–5 Ω), with a long tail to 1.0.

## Training threshold mismatch

| Threshold | Norm | Physical |
|-----------|------|----------|
| Dataset FG (`fg_norm_threshold`) | 0.000919 | 0.05 Ω |
| Config (`heatmap_fg_threshold`) | **0.0016** | **0.087 Ω** |

Training `_prepare_batch_gmax` zeros all pixels `< 0.0016`:
- **2.7%** of FG pixels globally clipped to zero
- **63.2%** of FG at **10 MHz** are below config thr (p50=0.00138)

This explains weak/empty generations at low MHz in compare plots.

## Per-MHz low tail (FG p50 norm)

| MHz | p10 | p50 | p90 | % below cfg thr |
|-----|-----|-----|-----|-----------------|
| 10 | 0.00098 | 0.00138 | 0.00246 | **63%** |
| 63 | 0.00594 | 0.00912 | 0.01953 | 0% |
| 400 | 0.01513 | 0.04174 | 0.10290 | 0% |

Low MHz maps live almost entirely in the bottom **0.3%** of the norm range.

## Transform comparison (400-map subsample)

| Space | Skew | p90/p10 spread |
|-------|------|----------------|
| Linear norm | 4.24 | **14.5×** |
| `log1p(phys)/log1p(gmax)` | 0.72 | 5.8× |
| `sqrt(norm)` | 1.46 | **3.8×** |

Log or sqrt remapping would spread the low end for learning; linear `/gmax` is the root compression.

## Recommended next steps

1. **Immediate** — set `heatmap_fg_threshold` to **0.000919** (match dataset stats); remove batch zeroing above stats thr or use stats thr only.
2. **Low MHz** — consider per-frequency scale or `log1p` norm targets for 10–63 MHz (or all freqs).
3. **Loss** — weight low-norm FG bins higher (inverse-CDF or sqrt norm in loss space).
4. **Revisit gmax** — global p99.5 max (54.4 Ω) may be too large for typical maps (per-map max median much lower), crushing dynamic range.

Re-run check:
```bash
.venv/bin/python scratch/check_gmax_low_value_skew.py
```
