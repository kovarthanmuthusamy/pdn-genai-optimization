# Latent optimization (Stage 2 — inverse design)

Gradient search in a **frozen** VAE latent space to propose decap placements that meet a target impedance mask under budget \(K\).

**Entry:** `pipelines/latent/optimize.py`  
**Related:** [impedance-surrogate.md](./impedance-surrogate.md), [framework-overview.md](./framework-overview.md)

---

## 1. Forward chain (per Adam step)

```text
z  →  occupancy decoder  →  soft probs p ∈ ℝ⁵²
   →  hard top-K + STE  →  binary b with ‖b‖₀ = K
   →  impedance model   → Ẑ(f) ∈ ℝ²³¹
   →  loss vs Z_target  →  ∇z
```

Default impedance model: **occupancy→spectrum surrogate** (`USE_SURROGATE=True`). Fallback: VAE impedance decoder head.

---

## 2. Straight-through estimator (STE)

Hard top-\(K\) is discrete and would block gradients. STE keeps a binary forward value while allowing identity-like gradients through soft probabilities so spectrum losses can reshape which slots are selected.

**Without STE:** optimization degenerates to seed sampling (no meaningful \(\partial\mathcal L/\partial z\) through the discrete mask).  
**With STE:** surrogate always sees an in-distribution \(K\)-hot vector; \(z\) still receives updates.

Implementation: `_ste_topk` in `optimize.py`.

---

## 3. Objective (composite loss)

Typical terms (weights are CONFIG constants):

| Term | Role |
|------|------|
| Exceed | Penalize frequencies above \(Z_{\mathrm{target}} - m\) |
| Gap reward | Reward headroom below the mask |
| Peak / dual top-\(K\) | Align resonance peaks (index + magnitude) |
| Anti-resonance | Physics prior against spurious series dips |
| \(z\) prior / posterior boundary | Keep \(z\) near aggregate posterior (on-manifold) |
| Diversity | Push parallel seeds toward distinct placements |
| Occupancy confidence | Prefer decisive (near-0/1) probabilities |

Loss can be computed in **log** or **ohm** space (`LOSS_SPACE`). Objective mode includes `gap_max` (default) and `feasible_map`.

---

## 4. Per-\(K\) procedure

1. Draw `NUM_CANDIDATE_SEEDS` latent seeds (random or fixed).
2. Optimize each seed for `NUM_STEPS` with Adam (`LR`, grad clip).
3. Mark **feasible** if \(\hat Z(f) \le Z_{\mathrm{target}}(f) - m\) for all bins (margin `BOUNDARY_MARGIN`).
4. Among feasible seeds, select by `SELECT_METRIC` (default `max_ohm` = lowest peak).
5. If none feasible → write `no_solution.json` for that \(K\).

Default `K_LIST`: \(1\ldots 25\) (edit CONFIG). Training data currently emphasizes \(K\le 30\); keep claims aligned.

---

## 5. Outputs

Under `data/latent_runs/<experiment>/<run>/K##/`:

| Artifact | Meaning |
|----------|---------|
| `best_latent.npy` | Selected latent |
| `best_occupancy_topk.npy` | Binary placement |
| `best_metrics.json` | Peak / feasibility metrics |
| `no_solution.json` | Explicit failure for that \(K\) |
| `run_config.json` | Seeds, weights, checkpoints used |

Reports: `pipelines/latent/generate_run_report.py`, HTML via `build_report.py`.  
ECAD verify path: `export_peb.py` → simulate → `compare_report.py`.

---

## 6. Default knobs (illustrative)

| Constant | Typical default |
|----------|-----------------|
| `NUM_STEPS` | 1200 |
| `LR` | \(5\times 10^{-2}\) |
| `NUM_CANDIDATE_SEEDS` | 32 |
| `BOUNDARY_MARGIN` | 0.1 |
| `EXCEED_WEIGHT` | 25 |
| `USE_SURROGATE` | True |

---

## 7. Thesis-critical limitation (Stage-1 / Stage-2 alignment)

As of the current codebase, `optimize.py` defaults to **`exp038_true_multi`** checkpoints for the occupancy decoder and impedance surrogate. The latest training lineage is **exp057–exp059**. Wiring Stage 2 to a newer VAE requires matching:

- occupancy decode (soft vs hard top-\(K\), GNN heads),
- latent dim / structured layout,
- a re-trained or compatible `surrogate_impedance` checkpoint,
- normalization stats paths.

Until that wiring is done, thesis claims about “end-to-end exp057 inverse design” must either (a) complete the port and re-validate with ECAD, or (b) clearly state Stage 2 results on the exp038 stack.

**Falsification for any Stage-2 claim:** ECAD re-sim of proposed placements; report feasibility and peak error in **physical** Ω, not only surrogate loss.

---

## 8. Pros / cons

| Pros | Cons |
|------|------|
| Continuous search avoids combinatorial enumerate | Surrogate can invent on-manifold-looking but wrong spectra |
| Explicit no-solution per \(K\) matches engineer decisions | STE is a biased gradient estimator |
| Parallel seeds + diversity improve coverage | Many loss weights → need ablation or sensitivity analysis |
| Direct use of discrete \(K\)-hot for surrogate | Stage-1/2 checkpoint mismatch risks invalidating latest-model claims |
