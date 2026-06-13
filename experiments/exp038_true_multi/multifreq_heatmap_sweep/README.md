# exp038 multifreq heatmap sweep (fixed K)

Checkpoint: `experiments/exp038_true_multi/checkpoints/checkpoint_epoch_900.pt`
Fixed K: **5**
PI frequencies (2): 80, 250
Training anchors: [10.0, 63.0, 130.0, 200.0, 270.0, 400.0, 500.0] MHz

## CAD workflow

1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.
2. Batch-simulate the `.peb` in ECADStar (manual).
3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.

PEB path: `experiments/exp038_true_multi/multifreq_heatmap_sweep/pi_distribution_K5_freq_sweep.peb` (PI-Distribution only, no PI-Spectrum).

## How to read results

- **`freq_*MHz/K{n}/`** — generated heatmaps per frequency (compare vs `Real/` after move).
- **`generate_heatmap_vs_mhz.png`** — if max/mean are flat, heatmap branch may ignore PI_freq.
- **`val_heatmap_summary.csv`** — optional offline val metrics.

## Outputs

- `experiments/exp038_true_multi/multifreq_heatmap_sweep/generate_summary.csv`
- `experiments/exp038_true_multi/multifreq_heatmap_sweep/generate_heatmap_vs_mhz.png`
- `experiments/exp038_true_multi/multifreq_heatmap_sweep/pi_distribution_K5_freq_sweep.peb`
