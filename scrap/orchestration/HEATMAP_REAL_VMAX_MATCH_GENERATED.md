# Heatmap Real vmax = Generated vmax

### 📝 Summary of Changes

- Added `HEATMAP_REAL_VMAX_MATCH_GENERATED` config flag in `run_multifreq_sweep_pipeline.py` and `scrap/comparison/compare.py`.
- When `True`, the Real heatmap panel uses the Generated panel's FG percentile vmax (`real_vmax = gen_vmax`); both panels share the same color scale.
- Complements existing `HEATMAP_SHARED_COLOR_SCALE` (`gen_vmax = real_vmax`). The two flags are mutually exclusive.

### 🚀 Implementation Details

| Flag | Effect |
|------|--------|
| `HEATMAP_SHARED_COLOR_SCALE = True` | Generated colorbar vmax = Real's p99.9 FG vmax |
| `HEATMAP_REAL_VMAX_MATCH_GENERATED = True` | Real colorbar vmax = Generated's p99.9 FG vmax |
| Both `False` (default) | Independent per-panel vmax |

Pipeline wiring: `_apply_compare_config()` passes the flag via `set_compare_overrides()` into `compare.py` before heatmap plots are built.

In percentile mode (`HEATMAP_VMAX_MODE = "percentile"`), `_plot_heatmap_comparisons()` computes `vmaxp_real` and `vmaxp_gen` from foreground pixels, then applies the link rule. Stats/global shared vmax modes ignore this flag.

### 🛠️ Verification & Execution Results

- Unit check of vmax pairing logic: **passed** (`real_from_gen` → both panels use `vmaxp_gen`).
- Linter: no issues on modified files.

**Usage** — in `run_multifreq_sweep_pipeline.py`:

```python
HEATMAP_REAL_VMAX_MATCH_GENERATED = True
```

Pipeline startup log reports `vmax_link=real_from_gen` when enabled.
