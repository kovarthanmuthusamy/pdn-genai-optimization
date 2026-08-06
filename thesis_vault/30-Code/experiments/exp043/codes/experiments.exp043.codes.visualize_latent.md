---
title: visualize_latent
type: code
path: experiments/exp043/codes/visualize_latent.py
group: experiments/exp043/codes
experiment: exp043
loc: 347
tags: [code, exp043, runnable]
---

# visualize_latent

> Latent-space visualizations for exp043 PI_freq-conditioned VAE.

**Source:** `experiments/exp043/codes/visualize_latent.py` · 347 lines
**Experiment:** [[exp043]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp043/codes/visualize_latent.py`

## Purpose

```text
Latent-space visualizations for exp043 PI_freq-conditioned VAE.

Run:
    python experiments/exp043/codes/visualize_latent.py
```

## Constants

| Name | Value |
|------|-------|
| `_CODES` | `Path(__file__).resolve().parent` |
| `TSNE_PERPLEXITY` | `40` |
| `MAX_SAMPLES_DEFAULT` | `30000` |
| `GRID_MHZ` | `(63.0, 200.0, 400.0)` |
| `GRID_K` | `26` |
| `MAX_SAMPLES` | `MAX_SAMPLES_DEFAULT` |
| `GRID_MHZ_LIST` | `list(GRID_MHZ)` |

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

`exp043_eval_common`, `matplotlib`, `numpy`, `sklearn`, `torch`
