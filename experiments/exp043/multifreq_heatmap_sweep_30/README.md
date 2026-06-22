# exp038 multifreq heatmap sweep (fixed K)

Checkpoint: `experiments/exp043/checkpoints/last_model.pt`
Fixed K: **30**
PI frequencies (5): 10, 63, 300, 330, 400
Training anchors: [10.0, 63.0, 80.0, 130.0, 150.0, 200.0, 230.0, 250.0, 270.0, 300.0, 330.0, 400.0, 500.0] MHz

## CAD workflow

1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.
2. Batch-simulate the `.peb` in ECADStar (manual).
3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.

PEB path: `experiments/exp043/multifreq_heatmap_sweep_30/pi_distribution_K30_freq_sweep.peb` (PI-Distribution only, no PI-Spectrum).

## How to read results

- **`freq_*MHz/K{n}/`** — generated heatmaps per frequency (compare vs `Real/` after move).
- **`generate_heatmap_vs_mhz.png`** — if max/mean are flat, heatmap branch may ignore PI_freq.
- **`val_heatmap_summary.csv`** — optional offline val metrics.

## Outputs

- `experiments/exp043/multifreq_heatmap_sweep_30/generate_summary.csv`
- `experiments/exp043/multifreq_heatmap_sweep_30/generate_heatmap_vs_mhz.png`
- `experiments/exp043/multifreq_heatmap_sweep_30/pi_distribution_K30_freq_sweep.peb`
