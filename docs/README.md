# Documentation (thesis reference)

Curated technical references for the master’s thesis: generative surrogate + inverse design for PCB PDN decap placement.

Implementation chatter and one-off bug fixes live in [`_archive/`](./_archive/) — not for thesis citation.

**Current production architecture (documented):** `exp059_capacity_freq` (capacity + multi-scale FiLM; AL target).  
**Legacy:** exp057 structured-graph, exp058 asymmetric KL — see [experiment-lineage.md](./experiment-lineage.md).

---

## Start here

| Document | Thesis use |
|----------|------------|
| [framework-overview.md](./framework-overview.md) | Problem, stages, scope — chapter map |
| [limitations-and-validity.md](./limitations-and-validity.md) | Threats to validity / claims hygiene |

## Methods — model & lineage

| Document | Thesis use |
|----------|------------|
| [model-architecture.md](./model-architecture.md) | exp054→057 architecture |
| [gnn-rationale.md](./gnn-rationale.md) | Why GNN (occ + spectrum); exp055→056→057 evidence |
| [experiment-lineage.md](./experiment-lineage.md) | Full arc incl. 043, 048–053, 058–059 |
| [normalization-and-losses.md](./normalization-and-losses.md) | Robust norm + Tier-A / peak losses |
| [training-procedure.md](./training-procedure.md) | Training loop structure |

## Methods — data & pipelines

| Document | Thesis use |
|----------|------------|
| [dataset.md](./dataset.md) | Layout store, anchors, K≤30, sampling |
| [data-pipeline.md](./data-pipeline.md) | Sim → append → normalize |

## Methods — inverse design

| Document | Thesis use |
|----------|------------|
| [latent-optimization.md](./latent-optimization.md) | Stage-2 latent search, STE, selection |
| [impedance-surrogate.md](./impedance-surrogate.md) | Occupancy → spectrum forward model |

## Methods — active learning

| Document | Thesis use |
|----------|------------|
| [active-learning.md](./active-learning.md) | Option-B MC AL cycle (exp057) |
| [gp-error-surrogate.md](./gp-error-surrogate.md) | Residual GP / UCB acquisition |

## Experiments & metrics

| Document | Thesis use |
|----------|------------|
| [evaluation-metrics.md](./evaluation-metrics.md) | Primary QC (Pearson, pattern MAE, max_ratio) |
| [evaluation-suite.md](./evaluation-suite.md) | Held-out eval, novelty, diagnostics |

## Figures

| Path | Content |
|------|---------|
| [`figures/exp050_sweep_k2_20/`](./figures/exp050_sweep_k2_20/) | Early multifreq sweep figures |

## Related outside `docs/`

| Path | Content |
|------|---------|
| [`../README.md`](../README.md) | Full project narrative |
| [`../active_learning_pi/README.md`](../active_learning_pi/README.md) | AL package entry |
| [`../active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md`](../active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md) | Full GP-AL framework |
| [`../evaluation/README.md`](../evaluation/README.md) | Evaluation package index |
| [`../pipelines/README.md`](../pipelines/README.md) | Pipeline runners |
