# exp038 multifreq heatmap sweep (fixed K)

Checkpoint: `experiments/exp039_improved_heatmap/checkpoints/last_model.pt`
Fixed K: **29**
PI frequencies (5): 10, 70, 120, 270, 400
Training anchors: [10.0, 63.0, 80.0, 130.0, 150.0, 200.0, 250.0, 270.0, 330.0, 400.0, 500.0] MHz

## CAD workflow

1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.
2. Batch-simulate the `.peb` in ECADStar (manual).
3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.

PEB path: `experiments/exp039_improved_heatmap/multifreq_heatmap_sweep_29K/pi_distribution_K29_freq_sweep.peb` (PI-Distribution only, no PI-Spectrum).

## How to read results

- **`freq_*MHz/K{n}/`** — generated heatmaps per frequency (compare vs `Real/` after move).
- **`generate_heatmap_vs_mhz.png`** — if max/mean are flat, heatmap branch may ignore PI_freq.
- **`val_heatmap_summary.csv`** — optional offline val metrics.

## Outputs

- `experiments/exp039_improved_heatmap/multifreq_heatmap_sweep_29K/generate_summary.csv`
- `experiments/exp039_improved_heatmap/multifreq_heatmap_sweep_29K/generate_heatmap_vs_mhz.png`
- `experiments/exp039_improved_heatmap/multifreq_heatmap_sweep_29K/pi_distribution_K29_freq_sweep.peb`
