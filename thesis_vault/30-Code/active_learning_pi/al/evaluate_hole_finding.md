---
title: evaluate_hole_finding
type: code
path: active_learning_pi/al/evaluate_hole_finding.py
group: active_learning_pi/al
loc: 518
tags: [code, active_learning_pi, runnable, uncommitted]
---

# evaluate_hole_finding

> Solid hole-finding evaluation for residual-GP active learning.

**Source:** `active_learning_pi/al/evaluate_hole_finding.py` · 518 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python active_learning_pi/al/evaluate_hole_finding.py`

## Purpose

```text
Solid hole-finding evaluation for residual-GP active learning.

Question answered
-----------------
Does acquisition preferentially select layouts where the VAE is *actually wrong*
("holes"), better than chance / random at equal budget?

This is stronger than within-selected Spearman on an ECAD batch (which only ranks
among already-selected layouts). Here every holdout layout is labeled, so we can
compare GP's top-k against the *true* worst-k oracle and against random.

Primary metrics (PASS criteria)
-------------------------------
1. **Enrichment / hole capture** at budget K:
   Among the true top-Q% worst residuals on the holdout, what fraction fall in
   GP's top-K? Compare to random expectation K/N.
   PASS: capture_rate ≥ 1.5 × random_rate  AND  capture_rate ≥ 0.25 (at Q=20%).

2. **Equal-budget lift**:
   mean(true_y | GP top-K) / mean(true_y | random top-K)
   PASS: lift ≥ 1.3  AND  Spearman(score, y) > 0.

3. **Pool enrichment**:
   mean(true_y | GP top-K) / mean(true_y | full holdout)
   PASS: ≥ 1.5 (selected set is substantially worse than average).

4. **Structured mid-band hole** (optional honesty check):
   Hold out mid-band MHz (e.g. 150–280) from GP fit; score held-out band.
   PASS: mean(y | GP top-K ∩ midband) / mean(y | random top-K ∩ midband) ≥ 1.2
   when enough mid-band points exist.

Run:
    python active_learning_pi/al/evaluate_hole_finding.py

Writes:
    ``OUT_DIR/LATEST_hole_finding.json``
    ``OUT_DIR/HOLE_FINDING_REPORT.md``
```

## Constants

| Name | Value |
|------|-------|
| `EXPERIMENT` | `'experiments/exp059_capacity_freq'` |
| `CHECKPOINT` | `None` |
| `DATA_DIR` | `None` |
| `PATH` | `'pred'` |
| `TARGET` | `'p99_ae'` |
| `N_SAMPLES` | `2500` |
| `TEST_FRAC` | `0.4` |
| `BUDGETS` | `[40, 80, 160]` |
| `HOLE_QUANTILE` | `0.2` |
| `SEED` | `0` |
| `GP_KIND` | `'sklearn'` |
| `N_INDUCING` | `128` |
| `SVGP_EPOCHS` | `40` |
| `DEVICE` | `'cuda:0'` |
| `MIDBAND_MHZ` | `(150.0, 280.0)` |
| `RUN_STRUCTURED_MIDBAND` | `True` |
| `OUT_DIR` | `'active_learning_pi/runs/hole_finding_exp059'` |
| `MIN_LIFT_VS_RANDOM` | `1.3` |
| `MIN_POOL_ENRICHMENT` | `1.5` |
| `MIN_CAPTURE_VS_RANDOM` | `1.5` |
| `MIN_CAPTURE_ABS` | `0.25` |
| `MIN_SPEARMAN` | `0.0` |
| `MIN_MIDBAND_LIFT` | `1.2` |

## Functions

- **`_spearman(scores: np.ndarray, y: np.ndarray)`**
- **`_topk_idx(scores: np.ndarray, k: int)`**
- **`_hole_mask(y: np.ndarray, q: float)`** — True for layouts in the worst ``q`` fraction of residual y.
- **`_metrics_at_budget(scores: np.ndarray, y: np.ndarray, *, k: int, hole_q: float, rng: np.random.RandomState, n_random_trials: int=50)`**
- **`_verdict(budgets: dict[str, dict], midband: dict | None)`**
- **`_render_md(report: dict)`**
- **`_fmt(v, digits: int=4)`**
- **`main()`**

## Imports

- [[active_learning_pi.al.paths]]
- [[config]]
- [[gp_error_surrogate]]
- [[inference_pool]]
- [[repo_paths]]

## External dependencies

`importlib`, `numpy`, `repo_paths`, `scipy`, `sklearn`, `torch`
