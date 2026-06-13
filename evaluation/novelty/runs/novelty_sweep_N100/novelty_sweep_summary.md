# VAE novelty sweep summary (many K)

This sweep generates and scores samples for **K=1..52** with **N=100 samples per K**.

## Where the outputs are

- Per-K generated samples + per-K novelty CSVs:
  - `scrap/generated_samples/novelty_sweep_N100/K{K}/`
- Aggregate summary CSV:
  - `scrap/generated_samples/novelty_sweep_N100/novelty_sweep_summary.csv`
- Dataset K-index cache (speeds up repeats):
  - `datasets/data_norm/k_index_cache.npz`

## Run settings

- Checkpoint: `experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt`
- Dataset: `datasets/data_norm`
- Heatmap pooling: `16x16` (mean-pooled from 64x64 before MSE)
- Baseline samples: `baseline_n=200`
- Train cap per K: `max_train=2000` (no effect here since each K had <= ~650 samples)

## Totals

- Total generated: **5200** samples
- Total occupancy exact matches (gen occupancy == some training occupancy): **275 / 5200**

## Ks with occupancy exact matches (>0)

These Ks had at least one generated sample whose *nearest* training neighbor matched occupancy exactly.

| K | train N | occ_exact (out of 100) | occ_ham median | combined median | baseline median | ratio (combined/base) |
|---:|-------:|-----------------------:|---------------:|----------------:|----------------:|----------------------:|
| 1  | 52  | 15  | 2 | 12.7040 | 0.2059 | 61.69 |
| 2  | 622 | 10  | 2 | 5.8468  | 0.0882 | 66.31 |
| 47 | 606 | 2   | 4 | 3.6195  | 0.0809 | 44.73 |
| 49 | 610 | 1   | 2 | 3.6242  | 0.0392 | 92.35 |
| 50 | 601 | 47  | 2 | 3.5494  | 0.0386 | 91.89 |
| 51 | 52  | 100 | 0 | 3.4304  | 0.0389 | 88.29 |
| 52 | 1   | 100 | 0 | 3.4566  | NA     | NA    |

Notes:
- For **K=52**, there is only **one possible occupancy pattern** (all ones), so occupancy exact matches are expected.
- For very high K values (like **K=51**), the number of distinct possible patterns is small (52 patterns), so exact occupancy matches are also much more likely even without memorization.

## Quick interpretation

- `occ_exact` being non-zero at extreme K is expected due to combinatorics (few possible patterns).
- What would be more concerning is: `occ_exact` high *and* `combined` distances close to the train→train baseline. In this sweep, the combined median distances are much larger than baseline for every K, so this looks more like **non-memorization / distribution mismatch** than copy-paste memorization.
