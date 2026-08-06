---
title: evaluate_vae
type: code
path: experiments/exp037_lat_change/codes/evaluate_vae.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 670
tags: [code, exp037_lat_change]
---

# evaluate_vae

> evaluate_vae.py — Post-training evaluation for exp025_latent_size_change

**Source:** `experiments/exp037_lat_change/codes/evaluate_vae.py` · 670 lines
**Experiment:** [[exp037_lat_change]]

## Purpose

```text
evaluate_vae.py — Post-training evaluation for exp025_latent_size_change

Runs 4 diagnostics:
  1. Variation within K     — are 50 samples at same K actually diverse?
  2. Nearest-neighbour dist — are generated samples novel (not memorised)?
  3. Latent interpolation   — is the decoder manifold smooth?
  4. Prior sampling sanity  — does z ~ N(0,1) produce valid output?

Usage:
    python3 experiments/exp025_latent_size_change/codes/evaluate_vae.py
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `'experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt'` |
| `DATA_DIR` | `'datasets/data_norm'` |
| `OUTPUT_DIR` | `'experiments/exp030_adding_physic/eval_results'` |
| `LATENT_DIM` | `48` |
| `DEVICE` | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `K_TEST_PRESENT` | `26` |
| `K_TEST_RARE` | `2` |
| `N_VARIATION` | `50` |
| `N_NN_TRAIN` | `500` |
| `N_PRIOR` | `200` |
| `INTERP_STEPS` | `8` |
| `TEMPS` | `[0.6, 0.8, 1.0, 1.2, 1.5]` |
| `FREQ_PATH` | `'configs/Frequency_data_hz.npy'` |
| `TARGET_IMP_PATH` | `'configs/target_impedance.npy'` |
| `BINARY_MASK_PATH` | `'configs/binary_mask.npy'` |
| `NORM_STATS_PATH` | `'datasets/data_norm/normalization_stats.json'` |

## Functions

- **`load_model(checkpoint_path: str, latent_dim: int, device: torch.device)`**
- **`encode_batch(model, batch, device)`** — Run encoder on a dataset batch dict → fused mu (B, D).
- **`test_variation(model, latent_stats, device, out_dir: Path)`**
- **`test_nn_distance(model, latent_stats, device, out_dir: Path)`**
- **`test_interpolation(model, device, out_dir: Path)`**
- **`test_prior_sampling(model, latent_stats, device, out_dir: Path)`**
- **`print_summary(var_results, nn_results, prior_results)`**
- **`plot_model_scorecard(var_results, nn_results, prior_results, out_dir: Path)`** — Single-page model health scorecard.
- **`main()`**

## Imports

- [[dataloader]]
- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`
