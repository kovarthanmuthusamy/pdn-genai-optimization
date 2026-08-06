---
title: validate_acquisition_ab
type: code
path: active_learning_pi/al/validate_acquisition_ab.py
group: active_learning_pi/al
loc: 321
tags: [code, active_learning_pi, runnable, uncommitted]
---

# validate_acquisition_ab

> Equal-budget acquisition A/B: GP-UCB (+novelty) vs random vs optional MC.

**Source:** `active_learning_pi/al/validate_acquisition_ab.py` · 321 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python active_learning_pi/al/validate_acquisition_ab.py`

## Purpose

```text
Equal-budget acquisition A/B: GP-UCB (+novelty) vs random vs optional MC.

Falsifies the claim that residual-GP acquisition selects higher *true* VAE error
than random (and MC) at the same simulation budget — **without new ECAD**.

Method
------
1. Collect labeled residuals on the production ``pred`` path (occ → decode → re-encode).
2. Hold out a test slice; fit GP (+ novelty bank) on the remainder.
3. Rank the holdout by: GP acquisition, random, and optionally MC ``auto_bad``.
4. Report Spearman(score, true y) and mean true y in the top-k (equal budget).

Pass criterion (thesis):
  mean_true_y(top-k GP) > mean_true_y(top-k random)  AND  Spearman(GP) > 0

Run:
    python active_learning_pi/al/validate_acquisition_ab.py

Agent notes:
    - What: cheap labeled holdout A/B for GP vs random (optional MC).
    - Usage: edit CONFIG below, then run. Writes JSON under ``OUT_DIR``.
    - Config keys: EXPERIMENT, N_SAMPLES, BUDGET_K, INCLUDE_MC, GP_* .
```

## Constants

| Name | Value |
|------|-------|
| `EXPERIMENT` | `'experiments/exp059_capacity_freq'` |
| `CHECKPOINT` | `None` |
| `DATA_DIR` | `None` |
| `PATH` | `'pred'` |
| `TARGET` | `'p99_ae'` |
| `N_SAMPLES` | `2000` |
| `TEST_FRAC` | `0.4` |
| `BUDGET_K` | `80` |
| `SEED` | `0` |
| `GP_KIND` | `'sklearn'` |
| `N_INDUCING` | `128` |
| `SVGP_EPOCHS` | `40` |
| `KAPPA` | `0.5` |
| `SCORE_MODE` | `'mu'` |
| `NOVELTY_WEIGHT` | `0.25` |
| `NOVELTY_K` | `10` |
| `JOINT_ALPHA_HM` | `1.0` |
| `JOINT_ALPHA_IMP` | `1.0` |
| `INCLUDE_MC` | `False` |
| `MC_MAX` | `120` |
| `DEVICE` | `'cpu'` |
| `OUT_DIR` | `'active_learning_pi/runs/acquisition_ab_validation_exp059'` |

## Functions

- **`_topk_mean(scores: np.ndarray, y: np.ndarray, k: int)`**
- **`_spearman(scores: np.ndarray, y: np.ndarray)`**
- **`main()`**

## Imports

- [[candidates]]
- [[config]]
- [[decision_report]]
- [[gp_error_surrogate]]
- [[inference_pool]]
- [[repo_paths]]

## External dependencies

`importlib`, `numpy`, `repo_paths`, `scipy`, `sklearn`, `torch`
