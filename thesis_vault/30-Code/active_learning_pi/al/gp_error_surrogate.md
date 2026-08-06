---
title: gp_error_surrogate
type: code
path: active_learning_pi/al/gp_error_surrogate.py
group: active_learning_pi/al
loc: 708
tags: [code, active_learning_pi, uncommitted]
---

# gp_error_surrogate

> Latent-space error-GP surrogate for active-learning acquisition.

**Source:** `active_learning_pi/al/gp_error_surrogate.py` · 708 lines
**Git:** uncommitted — not yet tracked

## Purpose

```text
Latent-space error-GP surrogate for active-learning acquisition.

Fits a GP (SVGP or sklearn) that maps a cheap VAE latent feature ``x`` to a
scalar residual ``y`` (VAE error vs ECAD on labeled layouts), then scores
unlabeled candidates with UCB:

    a(x) = mu_e(x) + kappa * sigma_e(x)

Production feature path (``pred``): self-prediction bootstrap without ECAD —

    occ → encode_occupancy_latent → decode Ĥ → re-encode(Ĥ) → z_pred

Falsification CLI: ``gp_error_surrogate_test.py`` (imports regressors / errors here).
```

## Classes

- **`SklearnGP`** — Thin wrapper giving sklearn GP a uniform fit/predict interface.
- **`SVGPRegressor`** — Sparse Variational GP (gpytorch): inducing points + ARD-RBF + variational ELBO.
- **`ErrorGPArtifact`** — Fitted error-GP ready for UCB (+ optional novelty) scoring.

## Functions

- **`peak_xy(hm: np.ndarray)`**
- **`fg_mask(hm: np.ndarray, *, bg: float, margin: float=0.5)`**
- **`soft_peak_xy(hm: np.ndarray, *, bg: float, margin: float=0.5)`**
- **`sharp_peak_xy(hm: np.ndarray, *, bg: float, margin: float=0.5, temperature: float=0.035)`**
- **`peak_dist_px(ay: float, ax: float, by: float, bx: float)`**
- **`errors_from_norm(pred: np.ndarray, gt: np.ndarray, *, bg: float=-2.6792, peak_temperature: float=0.035)`** — pred, gt: (H, W) normalised heatmaps (channel 0).
- **`impedance_mse(pred: np.ndarray, gt: np.ndarray)`** — Mean squared error on impedance spectrum vectors (any matching shape).
- **`physical_p99_abs_err(pred_phys: np.ndarray, gt_phys: np.ndarray)`** — Absolute error of foreground p99 between physical heatmaps (Ω).
- **`build_joint_target(y_hm: np.ndarray, y_imp: np.ndarray, *, alpha_hm: float=1.0, alpha_imp: float=1.0)`** — Scale-normalize heatmap and impedance residuals then weighted-sum.
- **`knn_novelty(z_query: np.ndarray, z_bank: np.ndarray, *, k: int=10)`** — Distance to k-th nearest neighbor in ``z_bank`` (higher = more novel / sparse).
- **`make_regressor(kind: str, device, *, n_inducing: int=256, epochs: int=80, verbose: bool=False)`**
- **`pred_bootstrap_features(model, *, occupancy: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, impedance: torch.Tensor | None=None, path: FeaturePath='pred')`** — Compute GP input latent ``z`` and predicted heatmap.
- **`collect_residuals(engine, loader, *, path: FeaturePath, n_samples: int, device: torch.device, cfg: dict, return_occupancy: bool=False)`** — Collect (Z, Y_dict, MHZ, KK[, OCC]) on labeled batches for GP fitting / falsification.
- **`fit_error_gp(engine, loader, *, path: FeaturePath='pred', target: ErrorTarget='mse', gp: str='svgp', n_fit: int=4000, n_inducing: int=256, svgp_epochs: int=80, kappa: float=0.5, novelty_weight: float=0.0, novelty_k: int=10, score_mode: str='mu', joint_alpha_hm: float=1.0, joint_alpha_imp: float=1.0, device: torch.device | None=None, cfg: dict | None=None, verbose: bool=True, seed: int=0)`** — Fit error-GP on labeled loader residuals; return scoring artifact.
- **`build_fit_loader(experiment_dir: str, data_dir: str | None, groot: Path, cfg_yaml: dict | None=None)`** — Shuffled loader over the experiment's multifreq dataset (for residual collection).
- **`resolve_gp_error_cfg(cfg: dict)`** — Defaults for ``cfg['gp_error']`` block.
- **`save_artifact_meta(artifact: ErrorGPArtifact, path: Path)`** — Save lightweight JSON metadata (not the full GP weights).

## Imports

- [[norm_stats]]
- [[robust_stats]]

## Imported by

- [[evaluate_hole_finding]]
- [[gp_error_surrogate_test]]
- [[inference_pool]]
- [[validate_acquisition_ab]]

## External dependencies

`gpytorch`, `importlib`, `numpy`, `sklearn`, `src_vae`, `torch`
