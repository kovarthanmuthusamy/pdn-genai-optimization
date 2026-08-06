# Experiment lineage (selected)

Chronology of generative-surrogate experiments relevant to the thesis narrative. Folders live under `experiments/`. Checkpoints are generally **not** transferable across architecture breaks.

---

## 1. Narrative arc

```text
PoE baseline (exp043)
  → norm / loss tuning (exp048–053)
  → self-contained trainer + K≤30 (exp054)
  → hard occ + peak losses (exp055)
  → occupancy GNN (exp056)
  → structured latent + spectrum GNN (exp057)   ← legacy architecture baseline
  → asymmetric KL / AL-oriented train (exp058)  ← legacy AL-path ablation
  → capacity + multi-scale FiLM + spectral loss (exp059)  ← current model + AL
```

Detailed architecture for exp054–057: [model-architecture.md](./model-architecture.md).

---

## 2. Table (methods-relevant)

| Experiment | Key change | Thesis relevance |
|------------|------------|------------------|
| **exp043** | Multi-input PoE VAE + frequency expert + FiLM heatmap | Architectural baseline |
| **exp048** | Robust clipped per-MHz norm; heavy loss stack | Showed spatial ceiling from z-clip |
| **exp049** | Unbounded robust norm | Better shape; magnitude risk @ high MHz |
| **exp050** | Tier-A losses (no percentile training; grad + phys blob) | Core loss philosophy |
| **exp052–053** | Unbounded Pearson / peak log1p variants | Intermediate peak-loss experiments |
| **exp054_K_30** | Vendored trainer; K≤30 unbounded data | Stable training baseline |
| **exp055_hard_occ** | Hard top-\(K\) before heatmap decode; top-region Huber | CAD-aligned decode; peak amplitude |
| **exp056_graph_vae** | PCB-slot occupancy GNN | Spatial inductive bias on slots; see [gnn-rationale.md](./gnn-rationale.md) |
| **exp057_structured_graph** | Structured 65-d latent; spectrum GNN; occ↔imp trunk | Legacy structured-graph baseline + spectrum GNN package |
| **exp058_asymmetric_kl** | Fresh train; lower β / higher free-bits; asymmetric layout/occ mix for AL path | Legacy AL-path ablation (occ-only similar to 059) |
| **exp059_capacity_freq** | Higher latent capacity; multi-scale FiLM; mid-band spectral loss; layout-holdout split; U-Net skips removed | **Current** model + residual-GP AL track |

Earlier folders (`exp029`–`exp047`) are historical prototypes; cite only if needed for a specific ablation.

---

## 3. exp058 — AL-oriented training (summary)

**Goal:** train from scratch (not resume exp057) with a recipe closer to the AL deployment path (heavy layout / occ-only mix, milder KL).

| Knob | Direction vs exp057 |
|------|---------------------|
| Init | Fresh |
| `beta_final` | Lower (~0.03) |
| `free_bits` | Higher (~0.12) |
| Train mix | Asymmetric layout / occ-only |
| AL scoring | Prefer occ-only; promote via `layout_cross` |

Config / notes: `experiments/exp058_asymmetric_kl/notes.md`, AL config `active_learning_pi/config/exp058.json`.

---

## 4. exp059 — Mid-band capacity and conditioning (summary)

**Observed failure mode:** low/high anchors sharp; mid-band (~150–280 MHz) heatmaps collapse toward a blurry conditional mean.

**Interventions (isolate variables — GAN deliberately not enabled):**

- Larger latent (`65→128`) and heatmap-private dims (`15→40`)
- Wider frequency conditioning (`cond_dim`, Fourier features) + **multi-scale FiLM** at every decoder scale
- Frequency-weighted **2D spectral (FFT) loss** with mid-band boost (~200 MHz center)
- Full-encode training first (AL distill off) to measure capacity/conditioning gains
- **Layout-level holdout** (all MHz of a `design_id` in train XOR val) — critical for residual/GP evaluation
- U-Net skip connections removed so decode matches Stage-2 (latent+FiLM only)

**Pros:** Direct attack on mid-band blur; holdout fix prevents leakage.  
**Cons:** Larger model; spectral loss adds hyperparams; still needs ECAD confirmation before claiming physical fidelity.

---

## 5. How to cite in the thesis

1. Present **exp059** as the current capacity/FiLM model and AL target; cite **exp057** for the structured-graph architecture baseline.
2. Use **exp048–050** when discussing normalization and Tier-A losses ([normalization-and-losses.md](./normalization-and-losses.md)).
3. Use **exp058** only as a legacy AL-path ablation (occ-only comparable to 059).
4. Always name the **checkpoint folder** and data split when reporting numbers. See [gp-error-surrogate.md](./gp-error-surrogate.md).
