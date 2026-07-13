# Heatmap quantile-inverse weighting (skew correction)

## Problem

Gmax + log1p targets are **right-skewed**: ~41% of FG pixels sit in norm [0.02, 0.05). Plain FG-mean Huber is dominated by the crowded **low** bins, so peaks and mid-high structure get weak gradients.

## Fix

`heatmap_quantile_weight` blends the base Huber term with **quantile-inverse** pixel weights (per sample, in train space):

- Split FG target values into `heatmap_quantile_bins` quantile bins (default 10).
- Weight each pixel ∝ `1 / (bin_count ^ power)` — **rare bins (usually high Ω) get more weight**.
- Blend with uniform Huber:  
  `base = (1 - w) * uniform + w * quantile_weighted`  
  where `w = heatmap_quantile_weight` (default **0.85**).

## Config (`config.yaml`)

```json
"heatmap_quantile_weight": 0.85,
"heatmap_quantile_bins": 10,
"heatmap_quantile_power": 0.75
```

| Key | Meaning |
|-----|---------|
| `heatmap_quantile_weight` | `0` = off; `1` = fully quantile-weighted base Huber |
| `heatmap_quantile_bins` | Quantile bins per map |
| `heatmap_quantile_power` | `<1` softens extreme inverse weights |

## Note

This **does not** upweight the crowded low tail further (that would reinforce flat outputs). It rebalances toward **underrepresented high/mid bins**. Spatial detail in the bulk is still covered by `pattern`, `grad`, and `spread` losses.

Requires **fresh training** (loss change only; compatible with v3+log1p checkpoints only if you accept loss mismatch — prefer new run).

## Code

`experiments/exp043/codes/gmax_heatmap_loss.py` — `_fg_quantile_inverse_weights`, `_fg_huber_base`

## GPU vectorization (2026-06)

The first implementation used a **per-sample Python loop** with `torch.quantile()` per batch element. That forced many GPU syncs and pushed epoch time from ~45s to ~172s (with `layout_distill` also enabled).

**Vectorized path** (same math, batched on GPU):

- `torch.nanquantile` over `(B, N_fg)` masked FG values for all quantile edges at once
- `torch.searchsorted` + `one_hot` for bin assignment
- Degenerate rows (flat FG span) fall back to min–max linear edges
- fp16-safe: quantiles computed in float32; weights cast back to target dtype

Benchmark (B=160, lite heatmap loss, CUDA): quantile adds ~**16 ms/iter** (~2 s/epoch) vs ~**100+ s/epoch** with the old loop.

Both **`layout_distill_weight`** and **`heatmap_quantile_weight`** can stay enabled without the 4× slowdown.
