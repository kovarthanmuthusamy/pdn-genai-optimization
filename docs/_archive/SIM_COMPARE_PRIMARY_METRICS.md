# Sim Compare Primary Metrics Update

### 📝 Summary of Changes

- **`scrap/orchestration/run_multifreq_sweep_pipeline.py`** — added `SIM_METRICS_PRIMARY_COLUMNS` config and docs; pipeline pushes primary columns into metrics writer.
- **`scrap/comparison/heatmap_sim_metrics.py`** — MD tables, agent copy block, and stdout summaries now lead with `pearson_r`, `pattern_mae`, `max_ratio` (not `mae_ohm`).
- **`scrap/orchestration/SIM_COMPARE_METRICS.md`** — documents primary vs secondary metrics.

### 🚀 Implementation Details

CAD sweep QC should judge generated vs ECADStar-simulated heatmaps on:

1. **Shape** — `pearson_r`, `pattern_mae`
2. **Peak scale** — `max_ratio` (`gen_max / real_max`)

`mae_ohm` and `mape_pct` stay in `sim_compare_metrics.csv` but are deprioritized in `.md` / logs because 10 MHz rows often have `mae_ohm=0` and `mape_pct` is unstable.

Re-run compare or `METRICS_ONLY=True` to regenerate `sim_compare_metrics.md` with the new format.

### 🛠️ Verification & Execution Results

- Synthetic 32×32 unit check: MD contains "Primary QC" header; per-sample table columns match `SIM_METRICS_PRIMARY_COLUMNS`; agent copy block leads with `r=`, `pattern_mae=`, `max_ratio=`.
