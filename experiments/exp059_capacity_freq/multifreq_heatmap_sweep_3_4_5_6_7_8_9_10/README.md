# multifreq heatmap sweep

Checkpoint: `experiments/exp059_capacity_freq/checkpoints/last_model.pt`
K values: **3_4_5_6_7_8_9_10**
PI frequencies (6): 30, 85, 160, 260, 480, 530
Training anchors: [10.0, 63.0, 80.0, 100.0, 120.0, 150.0, 170.0, 180.0, 200.0, 230.0, 250.0, 270.0, 280.0, 300.0, 330.0, 350.0, 370.0, 390.0, 400.0, 420.0, 430.0, 450.0, 470.0, 500.0] MHz

## CAD workflow

1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.
2. Batch-simulate the `.peb` in ECADStar (manual).
3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.

PEB path: `experiments/exp059_capacity_freq/multifreq_heatmap_sweep_3_4_5_6_7_8_9_10/pi_distribution_K3_4_5_6_7_8_9_10_freq_sweep.peb` (PI-Distribution only, run mode = heatmap).

## How to read results

- **`freq_*MHz/K{n}/`** — generated heatmaps per frequency (compare vs `Real/` after move).
- **`generate_heatmap_vs_mhz.png`** — if max/mean are flat, heatmap branch may ignore PI_freq.
- **`val_heatmap_summary.csv`** — optional offline val metrics.

## Outputs

- `experiments/exp059_capacity_freq/multifreq_heatmap_sweep_3_4_5_6_7_8_9_10/generate_summary.csv`
- `experiments/exp059_capacity_freq/multifreq_heatmap_sweep_3_4_5_6_7_8_9_10/generate_heatmap_vs_mhz.png`
- `experiments/exp059_capacity_freq/multifreq_heatmap_sweep_3_4_5_6_7_8_9_10/pi_distribution_K3_4_5_6_7_8_9_10_freq_sweep.peb`
