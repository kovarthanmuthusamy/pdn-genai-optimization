---
title: model-architecture
type: concept
source: docs/model-architecture.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/model-architecture.md` — edit the source file, then re-run `tools/build_vault.py`.

# Surrogate model architecture (exp054 → exp057)

Technical lineage of the multi-input product-of-experts (PoE) VAE used as a differentiable PDN surrogate. Losses are summarized here only where they changed with architecture; see [[normalization-and-losses|normalization-and-losses.md]] for the full loss rationale.

**Current model:** `experiments/exp057_structured_graph` (~10.7M parameters, latent dim 65).

---

## 1. Shared problem formulation

The surrogate jointly models, for a single board:

| Modality | Representation |
|----------|----------------|
| Decap occupancy | Binary vector \(\mathbf{b}\in\{0,1\}^{52}\) with budget \(K=\|\mathbf{b}\|_0\) |
| PI spectrum | Magnitude impedance over 231 frequency bins (≈1–600 MHz) |
| PI spatial map | 64×64 foreground impedance heatmap at selected MHz anchors |

Conditioning includes decap budget \(K\) and inspection frequency. Training uses a PoE latent with a layout-private path for heatmap generation at inference (no GT heatmap).

---

## 2. Experiment lineage

| Experiment | Folder | Main change | Why |
|------------|--------|-------------|-----|
| **exp054** | `exp054_K_30` | Self-contained trainer; unbounded robust norm; Tier-A heatmap stack; off-anchor eval | Stable baseline for K≤30 data |
| **exp055** | `exp055_hard_occ` | Hard top-\(K\) occupancy before heatmap decode; top-region Huber for peaks/valleys | Align decode with CAD-discrete layouts; improve peak amplitude |
| **exp056** | `exp056_graph_vae` | Occupancy encoder/decoder → PCB-slot GNN | Exploit spatial adjacency of 52 slots |
| **exp057** | `exp057_structured_graph` | Structured latent + spectrum GNN + occ↔imp joint trunk | Factor latent by expert role; couple occupancy and impedance |

Checkpoints are **not** interchangeable across these steps (encoder/decoder and latent layout differ).

---

## 3. exp054 — Self-contained multi-input PoE VAE

### Role

Baseline multi-modal VAE with vendored training stack (no dependency on older `exp038` at runtime).

### Module layout (still the pattern for later exps)

| Module | Role |
|--------|------|
| `train_vae_simple.py` | Entry: config, loss hooks, yaml apply |
| `train_core.py` | Epoch loop, checkpointing, plots |
| `dataloader_multifreq.py` | Cross-frequency pairs on layout store |
| `eval_off_anchor.py` | Interval-gated off-anchor spatial metrics |
| `heatmap_peak_losses.py` | Peak/valley / Tier-A terms |
| `physics_loss.py` | Optional physics RI / critic / AR terms |

### Intentional external deps

- `src_vae.others.*` (norm stats, logging)
- `experiments/exp043/codes/freq_conditioning` (FiLM frequency conditioning)

### Training data (typical)

- Normalized set: `datasets/data_multifreq_train_norm_unbounded` with **K≤30** filter
- Design-based train/val split; cross-freq alt-MHz pairs on the train path

See [[training-procedure|training-procedure.md]] for the epoch loop.

---

## 4. exp055 — Hard occupancy + top-region Huber

### Binary occupancy decode

Soft 52-d occupancy probabilities are converted to a **hard top-\(K\)** binary mask before the heatmap decoder’s spatial conditioning. This matches the discrete CAD placement used at deployment.

| Config key | Role |
|------------|------|
| `occupancy_binary_decode` | Enable hard top-\(K\) before heatmap decode |
| `occupancy_binary_ste_train` | Straight-through estimator (optional; off by default) |

**Pros:** Decode path matches inference; reduces soft-occupancy ambiguity.  
**Cons:** Non-differentiable hard step unless STE is enabled; STE can bias gradients.

### Top-region Huber (replaces centroid-only peak loss)

Centroid loss only matches the **position** of a hotspot. Top-region Huber applies Huber on **log1p amplitude** restricted to GT pixels above p95 (peak) / below p05 (valley):

- Position: only extreme-region pixels contribute
- Amplitude: Huber on log1p values in that region

Typical weights (opt-in via yaml): peak/valley topregion weight 2.5; centroid weight 0.

**Pros:** Tightens peak magnitude at hard frequencies (e.g. 270/400 MHz) where position-only loss was insufficient.  
**Cons:** Sensitive to percentile thresholds and FG mask quality; can overweight rare extreme pixels if FG is wrong.

---

## 5. exp056 — Occupancy graph VAE

Architecture-only change relative to exp055: occupancy uses message passing on the **52-slot PCB grid**.

### Graph topology

- Nodes: decap slots (`LABELS_ORDERED` / coordinates from `libs/data_creation/occupancy.py`)
- Edges: 4-neighbors on the 7×8 board grid + self-loops; symmetric normalized adjacency
- Implementation: pure PyTorch GNN (no `torch_geometric`); typically 3 layers, hidden 128

| Component | exp055 | exp056 |
|-----------|--------|--------|
| Occupancy encoder | MLP 52→64 | `OccGraphEncoder` → 64-d pool |
| Occupancy decoder | MLP → 52 logits | `OccGraphDecoder` per-node logits |
| Heatmap / impedance / PoE | unchanged | unchanged |

**Pros:** Inductive bias for spatially neighboring slots; shared with later exp057.  
**Cons:** Slightly more parameters on the occ path; cannot resume exp055 checkpoints.

---

## 6. exp057 — Structured latent + spectrum GNN (current)

Built from exp056 with three Tier-A upgrades. **Loss terms unchanged** relative to exp055/056; architecture and latent factorization change.

### Structured latent (\(d=65\))

| Block | Dims | Intended experts |
|-------|------|------------------|
| `z_shared` | 42 | Occupancy (occ expert) |
| `z_peak` | 8 | Impedance spectrum (imp expert) |
| `z_spatial` | 15 | Heatmap + frequency PoE (private) |

### Joint layout encoder

```text
occ_feat = OccGraphEncoder(occ)           # PCB 52-node GNN
imp_feat = ImpSpectrumGraphEncoder(imp)   # 231-bin chain GNN
joint    = fusion(cat(occ_feat, imp_feat))
→ occ & imp PoE experts use joint + K
```

### Impedance decode

```text
imp = ImpSpectrumGraphDecoder(
        cat(z_shared + z_peak, K, OccGraphEncoder(occ))
      )
```

Key modules: `codes/graph_occ.py`, `codes/graph_imp.py`. Shared GNN config: `graph_hidden_dim`, `graph_num_layers`, `graph_dropout`; `imp_peak_dim: 8`.

**Pros:** Explicit factorization for spectrum vs spatial maps; spectrum treated as a chain graph (adjacent frequency bins); occupancy conditions impedance decode.  
**Cons:** Larger design surface (latent splits, two GNNs); harder to ablate without careful controls; cannot resume exp056 weights.

### Train entry

```bash
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m experiments.exp057_structured_graph.codes.train_vae_simple
```

---

## 7. What to cite in the thesis methods chapter

1. Multi-input PoE VAE with layout-private heatmap path (exp054 baseline).
2. Hard top-\(K\) occupancy for CAD-aligned decode (exp055).
3. Top-region Huber for peak/valley amplitude (exp055).
4. PCB-slot GNN for occupancy (exp056) — rationale + numbers: [[gnn-rationale|gnn-rationale.md]].
5. Structured 65-d latent + impedance spectrum GNN + occ↔imp coupling (exp057) — same doc for evidence vs over-claiming.

For normalization and loss literature, use [[normalization-and-losses|normalization-and-losses.md]]. For the AL fine-tune of exp057, use [[active-learning|active-learning.md]].

**Later tracks (not a drop-in continuation of this doc’s architecture table):** exp058 (asymmetric KL / AL-path mix) and exp059 (higher capacity, multi-scale FiLM, spectral loss, layout holdout) — see [[experiment-lineage|experiment-lineage.md]].


## Implemented by

- [[experiments.exp059_capacity_freq.codes.vae_poe_freq]] — `experiments/exp059_capacity_freq/codes/vae_poe_freq.py`
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]] — `experiments/exp059_capacity_freq/codes/vae_multi_input_simple.py`
- [[exp059_common]] — `experiments/exp059_capacity_freq/codes/exp059_common.py`
