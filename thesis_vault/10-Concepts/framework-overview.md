---
title: framework-overview
type: concept
source: docs/framework-overview.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/framework-overview.md` — edit the source file, then re-run `tools/build_vault.py`.

# Research framework overview (thesis)

Condensed problem statement and two-stage framework for the master’s thesis. For implementation depth, see the linked topic docs. The project README (`../README.md`) is the narrative source; this file is the citation-oriented summary.

---

## 1. Engineering problem

A PCB **power delivery network (PDN)** must keep the impedance \(Z(f)\) seen by the IC below a frequency-dependent **target mask** \(Z_{\mathrm{target}}(f)\) over the band of interest (here **1–600 MHz**, sampled at **231** bins).

The designer’s lever is a binary decap placement

\[
\mathbf{b}\in\{0,1\}^{52},\qquad K=\|\mathbf{b}\|_0,
\]

with budget \(K\) constrained by cost/area. The map \(\mathbf{b}\mapsto Z(\cdot)\) is available only through an expensive field solver. Exhaustive search over \(2^{52}\) placements is intractable.

**Conventional loop:** place → simulate spectrum → if peaks violate the mask, run spatial PI-distribution → move decaps near hotspots → repeat.

---

## 2. Research approach

Replace the slow loop with:

1. A **learned differentiable surrogate** of one fixed board (multi-input VAE).
2. **Gradient-based inverse design** in latent space under budget \(K\).
3. Optional **active learning** to refine the surrogate on high-error / high-uncertainty layouts.
4. A **performance analysis** layer: frequency-conditioned heatmap decode at peak frequencies for hotspot interpretation.

> **Validity.** Surrogate spectra and heatmaps are predictions. Proposed designs require **ground-truth PI re-simulation** before sign-off.

---

## 3. Stages and code map

| Stage | Thesis role | Primary code |
|-------|-------------|--------------|
| Surrogate training | Learn joint model of occupancy, spectrum, heatmaps | `experiments/exp057_structured_graph/` (current); see also exp058/059 |
| Impedance surrogate | Direct \(\mathbf{b}\to\hat Z(f)\) for optimization | `experiments/exp038_true_multi/codes/surrogate_impedance.py` |
| Latent optimization | Inverse design per \(K\) | `pipelines/latent/optimize.py` |
| Active learning | Selective ECAD labels + fine-tune | `pipelines/active_learning/`, `active_learning_pi/` |
| Analysis | Hotspot maps from heatmap decoder | VAE decode at peak MHz |
| Evaluation | Off-anchor, novelty, held-out | `evaluation/`, training `off_anchor_eval.csv` |

---

## 4. Formal objective (Stage 2)

For each budget \(K\), find a placement that is **feasible**,

\[
\hat Z(f;\mathbf{b}) \le Z_{\mathrm{target}}(f) - m \quad \forall f,
\]

(with safety margin \(m\)), and among feasible candidates prefer **lowest peak impedance** \(\max_f \hat Z(f)\). If no seed is feasible, record an explicit **no-solution** for that \(K\).

Target file: `configs/target_impedance.npy` (shape aligned with 231 bins). Frequency grid: `configs/Frequency_data_hz.npy` (1 MHz–600 MHz).

---

## 5. Scope boundaries (state clearly in the thesis)

| In scope | Out of scope (today) |
|----------|----------------------|
| Single board / one design family | Cross-board generalization |
| Fixed decap type / slot set (52) | Multi-value / multi-type decap catalogs |
| \(K \le 30\) in current train filter | High-\(K\) regimes without retraining |
| Surrogate-assisted proposals | Replacing final SI/PI sign-off |

---

## 6. Document map for writing chapters

| Chapter need | Doc |
|--------------|-----|
| Methods — model | [[model-architecture|model-architecture.md]], [[experiment-lineage|experiment-lineage.md]] |
| Methods — data / norm / loss | [[dataset|dataset.md]], [[normalization-and-losses|normalization-and-losses.md]] |
| Methods — inverse design | [[latent-optimization|latent-optimization.md]], [[impedance-surrogate|impedance-surrogate.md]] |
| Methods — AL | [[active-learning|active-learning.md]], [[gp-error-surrogate|gp-error-surrogate.md]] |
| Experiments / metrics | [[evaluation-metrics|evaluation-metrics.md]], [[evaluation-suite|evaluation-suite.md]] |
| Threats to validity | [[limitations-and-validity|limitations-and-validity.md]] |
| Pipelines | [[data-pipeline|data-pipeline.md]], [[training-procedure|training-procedure.md]] |


## Implemented by

- [[optimize]] — `pipelines/latent/optimize.py`
- [[run]] — `pipelines/active_learning/run.py`
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]] — `experiments/exp059_capacity_freq/codes/train_vae_simple.py`
