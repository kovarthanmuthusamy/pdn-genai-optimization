# Impedance surrogate (occupancy → spectrum)

A small feed-forward network that maps a **hard binary** occupancy vector to a predicted PI spectrum. Used as the preferred forward model inside Stage-2 latent optimization.

**Code (canonical):** `experiments/exp038_true_multi/codes/surrogate_impedance.py`  
**Train:** `python experiments/exp038_true_multi/codes/surrogate_impedance.py`  
**Checkpoint (typical):** `experiments/exp038_true_multi/checkpoints/surrogate_best.pt`

---

## 1. Why a separate surrogate?

| Path | Input | Issue for inverse design |
|------|-------|--------------------------|
| VAE impedance decoder | Latent \(z\) (+ conditioning) | Coupled to generative heads; may disagree with hard occupancy readout |
| **Occupancy surrogate** | Exact \(K\)-hot \(\mathbf{b}\) | Matches the discrete placement the engineer / ECAD will use |

The optimizer’s STE path produces a binary vector with exactly \(K\) ones. Querying a model trained on the same discrete inputs reduces distribution shift between optimization and CAD.

---

## 2. Mapping

\[
\mathbf{b}\in\{0,1\}^{52} \;\longrightarrow\; \hat Z_{\mathrm{norm}} \in \mathbb{R}^{1\times 231}
\]

- **No PI frequency conditioning** in the surrogate (spectrum is broadband).
- Training target is the layout’s stored impedance (normalized log-\(z\) space consistent with the dataset’s `normalization_stats.json`).
- Spectrum losses reused from `impedance_spectrum_loss` (peak / dual top-\(K\) terms available).

---

## 3. Role in the pipeline

```text
Latent opt STE occupancy  →  surrogate(b)  →  Ẑ(f)  →  loss vs target
Optional: USE_SURROGATE=False → VAE decode_impedance instead
```

Feasibility sampling utilities: `pipelines/latent/find_feasible.py`.

---

## 4. Pros / cons

| Pros | Cons |
|------|------|
| In-distribution for hard placements | Extra model to train and keep in sync with data |
| Faster / simpler than full VAE forward for \(Z(f)\) | Currently tied to exp038 data/checkpoint paths |
| Clear ablation: surrogate ON vs VAE imp head | Does not predict spatial heatmaps (analysis still uses VAE) |

---

## 5. Thesis checklist

1. State train split and dataset for the surrogate.
2. Report surrogate vs ECAD spectrum error (e.g. peak MAE, band-wise) on a **held-out layout** set.
3. When claiming Stage-2 success, re-sim optimized placements — surrogate feasibility ≠ physical feasibility.
4. If porting to exp057+ data (`K≤30` unbounded), retrain the surrogate on that same distribution.
