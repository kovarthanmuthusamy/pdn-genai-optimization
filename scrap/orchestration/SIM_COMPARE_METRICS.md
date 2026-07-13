# Simulated Real vs Generated — Multi-Metric Evaluation

### 📝 Summary of Changes

- Added **`scrap/comparison/heatmap_sim_metrics.py`** — computes a full metric set per (MHz, K, sample) comparing **generated** `heatmap_physical.npy` vs **ECADStar-simulated real** `.map` files in `K*/Real/` (post-move), **not** dataset val heatmaps.
- **`scrap/comparison/compare.py`** — collects metrics during compare, writes bundle under `OUTPUT_ROOT`:
  - `sim_compare_metrics.csv` — flat table for spreadsheets/agents
  - `sim_compare_metrics.json` — per-sample rows + means by MHz and by K
  - `sim_compare_metrics.md` — human/agent copy block with one-line summaries
- **`build_comparison_report.py`** — links to metrics markdown when present.
- **Pipeline** — documents metrics output path in config summary.
- **`METRICS_ONLY`** — run only the evaluation (no generate/simulate/move/plots/report).

### 🚀 Implementation Details

**Primary QC metrics** (use for sweep verdicts — wired from `run_multifreq_sweep_pipeline.py`):

| Metric | Meaning |
|---|---|
| `pearson_r` | Spatial pattern correlation (FG pixels, Ω) |
| `pattern_mae` | MAE on min-max normalized maps (scale-invariant shape) |
| `max_ratio` | `gen_max_ohm / real_max_ohm` — peak magnitude match |

**Secondary / use with care:** `mae_ohm` is often **0 at 10 MHz** (low FG Ω); `mape_pct` blows up when real Ω ≈ 0. Full columns remain in CSV.

**Data sources:**
- **Real**: `freq_{MHz}MHz/K{n}/Real/Heatmap_real_{i}*/Z_*MHz.map` (after ECADStar + move step)
- **Generated**: `freq_{MHz}MHz/K{n}/data_sample_{i}/heatmap_physical.npy`

**Note:** `sweep_qc_eval.py` / `val_heatmap_summary.csv` still compare against **dataset** val heatmaps (pre-simulation QC). The new files are the post-simulation evaluation the agent should use.

### 🛠️ Verification & Execution Results

- Unit test on synthetic 32×32 maps: 26 metric fields computed; CSV/JSON/MD written successfully.

**Metrics only** (requires existing `Real/` + generated heatmaps under `OUTPUT_ROOT`):

```python
# scrap/orchestration/run_multifreq_sweep_pipeline.py
METRICS_ONLY = True
```

```bash
python scrap/orchestration/run_multifreq_sweep_pipeline.py
```

After run, inspect:

```bash
cat experiments/exp050/multifreq_heatmap_sweep_30/sim_compare_metrics.md
```
