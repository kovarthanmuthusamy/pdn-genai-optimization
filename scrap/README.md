# scrap/

| Folder | Purpose |
|--------|---------|
| [`generation/`](generation/) | VAE inference, PEB build, K-sweep, multifreq heatmap sweep |
| [`comparison/`](comparison/) | Gen vs real heatmap / impedance / occupancy |
| [`orchestration/`](orchestration/) | End-to-end sweep, move PI, HTML reports |

**Orchestration entry scripts** (no shims — run these directly):

- `scrap/orchestration/run_multifreq_sweep_pipeline.py`
- `scrap/orchestration/multifreq_move_and_compare.py`
- `scrap/orchestration/move_pi_to_real.py`
- `scrap/orchestration/build_comparison_report.py`
- `scrap/orchestration/compare_multifreq_datasets.py`
