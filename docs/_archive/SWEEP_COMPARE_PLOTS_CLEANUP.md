# Sweep script cleanup + compare plot metrics

### 📝 Summary of Changes

- **Compare diff panel**: shows signed amplitude map `Real − Gen` (RdBu) instead of normalized pattern diff.
- **Diff panel title**: Pearson `r` plus `Δmax = real_max − gen_max` with sign legend (− → gen higher, + → gen lower).
- **Real / Generated panels**: two-line titles (`headline` + scale stats), Ω colorbar labels.
- **Metrics**: added `max_diff_ohm` to `sim_compare_metrics.*` and primary QC columns.
- **Print cleanup**: consolidated multi-K QC report (prior fix), compact generate logs, shorter pipeline/compare stdout.

### 🚀 Implementation Details

**Sign convention** (`max_diff_ohm = real_max − gen_max`):
| Δmax | Meaning |
|------|---------|
| negative | generated peak **higher** than real |
| positive | generated peak **lower** than real |

**Diff heatmap**: blue = gen higher locally, red = real higher locally (same sign as Δmax).

**Plot speed** (3×3 figure benchmark): ~7.1 s → ~2.0 s (**3.6× faster**).
- `HEATMAP_PLOT_DPI = 130` (was 300)
- `HEATMAP_INTERPOLATION = "bilinear"` (was bicubic)
- `constrained_layout=True` + dropped `bbox_inches="tight"` (avoids double-render layout pass)
- Tune via `HEATMAP_PLOT_DPI` / `HEATMAP_INTERPOLATION` (`"nearest"` = fastest).

**Files**:
- `scrap/comparison/compare.py` — plot layout + print cleanup
- `scrap/comparison/heatmap_sim_metrics.py` — `max_diff_ohm` metric
- `scrap/generation/run_multifreq_heatmap_sweep.py` — compact generation logs
- `scrap/orchestration/run_multifreq_sweep_pipeline.py` — shorter config/step banners

### 🛠️ Verification & Execution Results

```text
python -m py_compile scrap/comparison/compare.py ...
# OK

max_diff_ohm sign test: real=2, gen=3 → Δmax=-1 (gen > real)  # OK
```

Re-run compare step to regenerate `generated_vs_real_heatmap.png` under each `freq_*MHz/K*/`.
