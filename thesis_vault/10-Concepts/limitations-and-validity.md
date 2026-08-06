---
title: limitations-and-validity
type: concept
source: docs/limitations-and-validity.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/limitations-and-validity.md` — edit the source file, then re-run `tools/build_vault.py`.

# Limitations and threats to validity

Explicit scope limits and validity threats for thesis reporting. Prefer stating these in Methods / Discussion rather than overclaiming.

---

## 1. Scope limitations

| Limit | Implication |
|-------|-------------|
| **Single board / design** | Surrogate does not transfer to a new PCB without retraining (or a future design-conditioned model) |
| **Fixed slot set (52) and decap type** | Catalog changes require new data and likely re-architecture |
| **K≤30 training filter** | Claims outside this budget need new data or clear out-of-scope labeling |
| **Discrete MHz anchors** | Off-anchor behavior is interpolated / cross-freq trained — denser bands still help |
| **Surrogate ≠ sign-off** | Engineer must re-simulate proposed placements |

---

## 2. Stage-1 / Stage-2 alignment

Latent optimization (`pipelines/latent/optimize.py`) historically defaults to **exp038** VAE + impedance surrogate checkpoints, while the newest surrogates are **exp057–exp059**.

**Threat:** reporting “exp057 inverse-design results” without matching checkpoints is invalid.

**Mitigation:** port and re-validate, or report Stage 2 as an earlier stack with an explicit version table.

---

## 3. Evaluation threats

| Threat | Mitigation |
|--------|------------|
| Frequency split within the same layout leaks into val | Use **layout-level** holdout (exp059 fix) |
| AL “improvement” without random acquisition baseline | Run equal-budget random control |
| High Pearson with wrong peak scale | Always report `max_ratio` / peak Ω metrics |
| Overlay fine-tune overfits AL batch | Track held-out / pre-vs-post on fresh ECAD; monitor train–val gap |
| MC uncertainty weakly correlated with true error | Prefer residual/GP acquisition or validate Spearman before claiming AL efficacy |
| Stats leakage in normalization | Fit median/IQR on **train layouts only** |

---

## 4. Modeling threats

| Threat | Mitigation |
|--------|------------|
| Unbounded z → Ω blow-up at high MHz | Physics / peak / overshoot terms; ECAD QC |
| Mid-band heatmap blur (conditional mean) | Capacity + multi-scale FiLM + spectral loss (exp059); verify on ECAD |
| STE bias in Stage 2 | Multi-seed search; ECAD verify; optional discrete local search polish |
| Occ ↔ imp decoder inconsistency | Joint trunk (exp057); surrogate for \(Z(f)\); consistency metrics |
| GAN detail hallucination | Deliberately avoided as primary loss for peak-critical maps |

---

## 5. Operational / data threats

| Threat | Note |
|--------|------|
| Incomplete ECAD appends | Stale `dataset_meta` / row counts → silent train corruption |
| ECADStar automation failures | Now headless (`engineer.exe --batch`); environmental, not model bugs — document setup |
| Symlink / path drift | Use `repo_paths.py`; verify layout store integrity |

---

## 6. Recommended validity statement (template)

> The surrogate is trained on a single board design with decap budgets \(K\le 30\). Feasibility and spectra reported by the surrogate or Stage-2 optimizer are predictive; all candidate placements intended for design use were / must be re-verified with the ground-truth PI solver. Active-learning gains are reported relative to [random / previous checkpoint] at equal simulation budget.

Adapt to what you actually ran.


## Implemented by

- [[decision_report]] — `active_learning_pi/al/decision_report.py`
