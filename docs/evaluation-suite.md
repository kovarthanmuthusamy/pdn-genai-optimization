# Evaluation suite

Beyond per-epoch training metrics and AL cycle reports, the repository includes dedicated evaluation entry points for reconstruction quality, held-out tests, novelty/memorization, and latent diagnostics.

---

## 1. What to use when

| Question | Tool |
|----------|------|
| Off-anchor heatmap quality during train | `metrics/off_anchor_eval.csv` (per experiment) |
| Gen vs ECAD sweep QC | Sim-compare primary metrics — [evaluation-metrics.md](./evaluation-metrics.md) |
| AL cycle pre/post | `CYCLE_EVAL_REPORT.md` — [active-learning.md](./active-learning.md) |
| Held-out reconstruction / diagnostics | `evaluation/vae/run_vae_eval.py`, `run_vae_eval_heldout.py` |
| Memorization / nearest-neighbour novelty | `evaluation/novelty/scripts/` |
| Latent structure / traversal | `evaluation/latent_traversal/`, `pipelines/analysis/` |

Package index: `evaluation/README.md`.

---

## 2. VAE evaluation

```bash
python evaluation/vae/run_vae_eval.py
```

Typical outputs under `evaluation/vae/`:

- `vae_eval_report.md`
- `vae_eval_metrics.json`
- `vae_eval_per_sample.csv`
- `plots/*.png`

**Held-out full test:** build eval set with `pipelines/data/processing_eval.py`, normalize, then:

```bash
python evaluation/vae/run_vae_eval_heldout.py
```

Set `CHECKPOINT`, `DATASET_ROOT`, `OUT_DIR` in the script CONFIG block.

**Thesis tip:** prefer layout-holdout datasets consistent with exp059 split hygiene.

---

## 3. Novelty / memorization

Scripts under `evaluation/novelty/scripts/`:

| Script | Role |
|--------|------|
| `run_vae_novelty_test.py` | Single novelty test |
| `run_vae_novelty_sweep.py` | Sweep settings |
| `vae_novelty_report.py` | Report |
| `summarize_novelty_sweep.py` | Aggregate |

**Why it matters:** generative PDN proposals should not be trivial nearest neighbours of the training set if the claim is “search of unseen placements.” Report distance-to-train (occupancy Hamming / latent) alongside ECAD quality.

---

## 4. Metrics to standardize in thesis tables

| Family | Examples |
|--------|----------|
| Shape | Pearson \(r\), pattern MAE |
| Peak scale | `max_ratio`, peak Ω MAE / p99 MAE |
| Spectrum | Peak alignment, band-wise MAE vs ECAD |
| Feasibility | Fraction of freqs under target; binary feasible Y/N |
| AL | Pre vs post ECAD error; vs random baseline |
| Novelty | NN distance in occupancy / latent |

Avoid sole reliance on train loss or on-anchor reconstruction.

---

## 5. Related docs

- [evaluation-metrics.md](./evaluation-metrics.md) — primary QC definitions  
- [limitations-and-validity.md](./limitations-and-validity.md) — threats  
- [gp-error-surrogate.md](./gp-error-surrogate.md) — residual ranking validation  
