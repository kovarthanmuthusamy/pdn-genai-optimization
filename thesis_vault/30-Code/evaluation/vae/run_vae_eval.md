---
title: run_vae_eval
type: code
path: evaluation/vae/run_vae_eval.py
group: evaluation/vae
loc: 1396
tags: [code, evaluation]
---

# run_vae_eval

> Evaluate a trained VAE checkpoint (reconstruction + latent diagnostics).

**Source:** `evaluation/vae/run_vae_eval.py` · 1396 lines

## Purpose

```text
Evaluate a trained VAE checkpoint (reconstruction + latent diagnostics).

This script is intentionally lightweight:
- No extra dependencies beyond numpy/torch/matplotlib
- Writes a compact Markdown report with a few plots
- Writes a CSV with per-sample metrics for deeper inspection

Default target checkpoint:
  experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt

Run:
  python evaluation/vae/run_vae_eval.py
```

## Constants

| Name | Value |
|------|-------|
| `CONFIG` | `EvalConfig()` |

## Classes

- **`EvalConfig`**
- **`VAEModel(Protocol)`**
- **`LoadedModel`**

## Functions

- **`_find_project_root(start: Path)`**
- **`_parse_sample_id(name: str)`**
- **`_load_model(*, project_root: Path, checkpoint_path: Path, device: torch.device)`**
- **`_load_ids(*, occ_dir: Path)`**
- **`_iter_batches(ids: list[int], *, batch_size: int)`**
- **`_load_batch(*, dataset_root: Path, batch_ids: list[int])`**
- **`_topk_match_rate(probs: torch.Tensor, occ_true: torch.Tensor)`** — Top-K set match rate, where K is taken from the target occupancy count.
- **`_plot_recon_vs_k(*, k: np.ndarray, hm_mse: np.ndarray, imp_mse: np.ndarray, occ_bce: np.ndarray, out_path: Path)`**
- **`_plot_kl_per_dim(*, kl_per_dim: np.ndarray, out_path: Path)`**
- **`_plot_occ_rates(*, occ_real_rate: np.ndarray, occ_gen_rate: np.ndarray, out_path: Path)`**
- **`_pca_from_mu(mu: np.ndarray)`** — Compute PCA from covariance (no sklearn).
- **`_nan_to_none_list(x: np.ndarray)`**
- **`_plot_latent_pca_scatter(*, mu: np.ndarray, K: np.ndarray, mean: np.ndarray, components: np.ndarray, evr: np.ndarray, out_path: Path, max_points: int=30000, seed: int=0)`**
- **`_plot_latent_pca_variance(*, evr: np.ndarray, dims_90: int, dims_95: int, dims_99: int, out_path: Path)`**
- **`_plot_latent_per_dim_mu_sigma(*, mu_std: np.ndarray, sigma_mean: np.ndarray, out_path: Path, dead_sigma_thresh: float=0.95, dead_mu_std_thresh: float=0.05)`**
- **`_plot_hist_by_k_bucket(*, values: np.ndarray, K: np.ndarray, out_path: Path, xlabel: str, title: str, target_vlines: list[tuple[float, str, str, str]] | None=None)`** — Histogram values by K buckets.
- **`_linear_probe_k_from_mu(mu: np.ndarray, K: np.ndarray, *, seed: int=0)`** — Fit a linear regression K<-mu and report (R², MAE) on a random 80/20 split.
- **`_plot_probe_k_from_mu(*, y_true: np.ndarray, y_pred: np.ndarray, r2: float, mae: float, out_path: Path)`**
- **`_plot_expert_disagreement_by_k(*, K_counts: np.ndarray, per_modality: dict[str, dict[str, np.ndarray]], out_path: Path)`**
- **`_fmt(x: float)`**
- **`_extract_latent_mu_std(*, latent_stats: dict | None, latent_dim: int)`** — Return (mu_per_dim, agg_std_per_dim) in numpy float32.
- **`_build_per_k_latent_tables(*, per_k_latent_stats: dict | None, global_mu: np.ndarray, global_std: np.ndarray, latent_dim: int)`** — Build lookup tables for K=0..52.
- **`run_eval(cfg: EvalConfig)`**
- **`main()`**

## External dependencies

`experiments`, `matplotlib`, `numpy`, `torch`
