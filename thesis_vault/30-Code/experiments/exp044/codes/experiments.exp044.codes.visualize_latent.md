---
title: visualize_latent
type: code
path: experiments/exp044/codes/visualize_latent.py
group: experiments/exp044/codes
experiment: exp044
loc: 357
tags: [code, exp044]
---

# visualize_latent

> visualize_latent.py — Latent space visualizations for exp044 (PI_freq-conditioned VAE).

**Source:** `experiments/exp044/codes/visualize_latent.py` · 357 lines
**Experiment:** [[exp044]]

## Purpose

```text
visualize_latent.py — Latent space visualizations for exp044 (PI_freq-conditioned VAE).

Encodes a multifreq dataset subset and writes:
  - t-SNE / PCA colored by K and by PI frequency (MHz)
  - PCA variance, per-dim mu/sigma, KL diagnostics
  - Decoder grids along top-2 PCs at several PI_freq (MHz)
  - Linear probes: K←mu and MHz←mu
  - Expert-vs-fused disagreement vs K and vs MHz

Usage:
    cd ~/gan
    python experiments/exp044/codes/visualize_latent.py
    python experiments/exp044/codes/visualize_latent.py --ckpt checkpoints/last_model.pt
```

## Constants

| Name | Value |
|------|-------|
| `_CODES` | `Path(__file__).resolve().parent` |
| `TSNE_PERPLEXITY` | `40` |
| `MAX_SAMPLES_DEFAULT` | `30000` |
| `GRID_MHZ` | `(63.0, 200.0, 400.0)` |
| `GRID_K` | `26` |

## Functions

- **`_k_cmap(K_arr)`**
- **`_mhz_cmap(mhz_arr)`**
- **`kl_diag_standard_normal(mu, logvar)`**
- **`plot_tsne(mu, color_arr, norm, out_path: Path, title: str, cbar_label: str)`**
- **`plot_pca_scatter(mu, color_arr, norm, out_path: Path, title: str, cbar_label: str)`**
- **`plot_pca_variance(mu, out_dir: Path)`**
- **`plot_per_dim(mu, sigma, out_dir: Path)`**
- **`plot_decoder_grid(model, pca, device, paths, mhz: float, K_val: int, grid_n: int, out_dir: Path)`**
- **`plot_sigma_by_buckets(sigma, labels, bucket_fn, out_path: Path, title: str)`**
- **`_k_buckets(K_arr)`**
- **`_mhz_anchor_buckets(mhz_arr)`**
- **`plot_probe(y, pred_labels, out_path: Path, title: str, xlabel: str)`**
- **`plot_probes(mu, K_arr, mhz_arr, out_dir: Path)`**
- **`plot_kl_by_buckets(mu, logvar, bucket_fn, out_path: Path, title: str)`**
- **`main()`**

## External dependencies

`exp044_eval_common`, `matplotlib`, `numpy`, `sklearn`, `torch`
