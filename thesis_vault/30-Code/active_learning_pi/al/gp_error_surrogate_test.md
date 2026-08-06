---
title: gp_error_surrogate_test
type: code
path: active_learning_pi/al/gp_error_surrogate_test.py
group: active_learning_pi/al
loc: 500
tags: [code, active_learning_pi, uncommitted]
---

# gp_error_surrogate_test

> Falsification test for the GP error-surrogate AL idea.

**Source:** `active_learning_pi/al/gp_error_surrogate_test.py` · 500 lines
**Git:** uncommitted — not yet tracked

## Purpose

```text
Falsification test for the GP error-surrogate AL idea.

Question this answers (cheaply, before building the full AL loop):
    "Is the VAE's prediction error SMOOTH in its latent z?"
    i.e. can a Gaussian Process trained on (z -> error) for some layouts
    predict the error on *held-out* layouts?

If yes  -> the latent-space error-GP acquisition is viable.
If no   -> the latent can't support it (switch to physics features, or the
           occ-only bottleneck makes error unlearnable there).

Method
------
For a set of real, ground-truth-labelled layouts (the val split):
  x = VAE latent z            (full-encode  OR  occ-only-encode path)
  y = VAE prediction error    (peak-location px / structural / MAE),
      all computed in NORMALISED heatmap space so no denorm ambiguity.
      Peak targets include both hard argmax (diagnostic) and smooth soft-argmax
      (training-aligned surrogate — use this for GP acquisition).
Then: random split -> fit sklearn GP on a subset -> predict on held-out ->
report Spearman/Pearson(pred_error, true_error). A kNN-in-z baseline is
reported as an assumption-light cross-check, plus GP sigma calibration.

Run:
    python active_learning_pi/al/gp_error_surrogate_test.py         --exp experiments/exp059_capacity_freq         --path occ --n-samples 3000
```

## Constants

| Name | Value |
|------|-------|
| `_GP_OPTS` | `{'kind': 'sklearn', 'n_inducing': 256, 'epochs': 80, 'verbose': False}` |

## Functions

- **`_load_engine_and_loader(exp: str, checkpoint: str | None, device: torch.device)`**
- **`collect(engine, val_loader, *, path: str, n_samples: int, device: torch.device, cfg: dict)`**
- **`collect_multi(engine, val_loader, paths, *, n_samples, device, cfg)`** — Collect latents + errors for several paths on the SAME layouts (apples-to-
- **`_structured_split(holdout, MHZ, KK, values, test_frac, seed)`** — Return (idx_fit, idx_test, description).
- **`_knn_baseline(z_fit, y_fit, z_test, k: int=10)`** — Assumption-light smoothness check: predict error = mean of k nearest
- **`_make_regressor(kind, device)`**
- **`_report_target(name, y, z, idx_fit, idx_test, n_fit_cap, idx_ref=None, gp_kind='sklearn', device=None)`**
- **`_evaluate(Z, Y, MHZ, KK, args, *, targets, device=None)`** — Split (structured OOD or random) + scale-on-fit + per-target GP report.
- **`main()`**

## Imports

- [[gp_error_surrogate]]
- [[norm_stats]]

## External dependencies

`importlib`, `numpy`, `scipy`, `sklearn`, `src_vae`, `torch`
