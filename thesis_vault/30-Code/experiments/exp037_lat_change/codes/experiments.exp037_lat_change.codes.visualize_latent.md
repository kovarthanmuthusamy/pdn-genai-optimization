---
title: visualize_latent
type: code
path: experiments/exp037_lat_change/codes/visualize_latent.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 560
tags: [code, exp037_lat_change]
---

# visualize_latent

> visualize_latent.py — Latent space visualizations for exp025_latent_size_change

**Source:** `experiments/exp037_lat_change/codes/visualize_latent.py` · 560 lines
**Experiment:** [[exp037_lat_change]]

## Purpose

```text
visualize_latent.py — Latent space visualizations for exp025_latent_size_change

Encodes the full dataset → collects all fused mu vectors → produces:

  Plot 1 — t-SNE 2D scatter    colored by K
  Plot 2 — PCA  2D scatter     colored by K
  Plot 3 — PCA explained variance  (how many dims the model uses)
  Plot 4 — Per-dim mu / sigma  (detect posterior collapse)
  Plot 5 — 2D decoder grid     (traverse top-2 PCA axes, decode at each grid point)
  Plot 6 — Fused sigma histogram per K bucket  (is sigma_reg working?)
    Plot 7 — KL per latent dim   (information usage per dim)
    Plot 8 — KL distribution per K bucket
    Plot 9 — Linear probe: predict K from mu (R²)
    Plot 10 — Modality-vs-fused disagreement vs K (experts vs PoE fused)

Usage:
    cd /home/ubuntu/genai_pdn
    python3 experiments/exp030_adding_physic/codes/visualize_latent.py
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `'experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt'` |
| `DATA_DIR` | `'datasets/data_norm'` |
| `OUTPUT_DIR` | `'experiments/exp030_adding_physic/latent_visuals_1'` |
| `LATENT_DIM` | `32` |
| `BATCH_SIZE` | `128` |
| `MAX_SAMPLES` | `30000` |
| `TSNE_PERPLEXITY` | `40` |
| `NORM_STATS_PATH` | `'datasets/data_norm/normalization_stats.json'` |
| `BINARY_MASK_PATH` | `'configs/binary_mask.npy'` |
| `FREQ_PATH` | `'configs/Frequency_data_hz.npy'` |
| `TARGET_IMP_PATH` | `'configs/target_impedance.npy'` |
| `DEVICE` | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |

## Functions

- **`load_model(checkpoint_path)`**
- **`encode_dataset(model, data_dir, max_samples=MAX_SAMPLES, collect_experts=True)`** — Encode dataset subset → return fused stats + optional per-modality expert stats.
- **`kl_diag_standard_normal(mu, logvar)`** — KL(q(z|x)=N(mu,diag(exp(logvar))) || p(z)=N(0,I)). Returns (N,D).
- **`_bucket_specs()`**
- **`k_cmap(K_arr)`** — Return RGBA colors mapped from K values (0–52).
- **`plot_tsne(mu, K_arr, out_dir)`**
- **`plot_pca_scatter(mu, K_arr, out_dir)`**
- **`plot_pca_variance(mu, out_dir)`**
- **`plot_per_dim(mu, sigma, out_dir)`**
- **`plot_decoder_grid(model, pca, out_dir, K_val=26, grid_n=8)`** — Traverse the top-2 PCA axes and decode at each grid point.
- **`plot_sigma_by_K(sigma, K_arr, out_dir)`** — Show distribution of per-sample mean sigma, bucketed by K range.
- **`plot_kl_per_dim(mu, logvar, out_dir)`**
- **`plot_kl_by_K(mu, logvar, K_arr, out_dir)`**
- **`plot_probe_K_from_mu(mu, K_arr, out_dir)`**
- **`plot_expert_disagreement(experts, fused_mu, fused_logvar, K_arr, out_dir)`**
- **`main()`**

## Imports

- [[dataloader]]
- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `sklearn`, `src_vae`, `torch`
