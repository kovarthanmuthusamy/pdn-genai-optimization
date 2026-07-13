# exp057 — Structured latent + GNN impedance + occ↔imp coupling

### 📝 Summary of Changes

- Added **`experiments/exp057_structured_graph/`** from exp056 with Tier A core upgrades:
  1. **Structured latent** `z = [z_shared (42) | z_peak (8) | z_spatial (15)]` (total 65)
  2. **GNN impedance** encoder/decoder on **231-bin chain graph** (`graph_imp.py`)
  3. **Occ↔imp joint trunk** + imp decode conditioned on **GNN(occ)**
- **Losses unchanged** — same `train_vae_simple.py` / `train_core.py` terms as exp055/056.
- exp056 **PCB-slot GNN** for occupancy retained (`graph_occ.py`).

### 🚀 Implementation Details

**Latent layout (default `latent_dim=65`)**

| Block | Dims | Experts |
|-------|------|---------|
| `z_shared` | 42 | occupancy (occ expert) |
| `z_peak` | 8 | impedance (imp expert) |
| `z_spatial` | 15 | heatmap + freq PoE (private) |

**Joint layout encoder**

```text
occ_feat = OccGraphEncoder(occ)          # PCB 52-node GNN
imp_feat = ImpSpectrumGraphEncoder(imp)  # 231-bin chain GNN
joint    = fusion(cat(occ_feat, imp_feat))
→ occ & imp PoE experts use joint + K
```

**Impedance decode (Tier 3)**

```text
imp = ImpSpectrumGraphDecoder(cat(z_shared+z_peak, K, OccGraphEncoder(occ)))
```

**New files**

- `codes/graph_imp.py` — spectrum GNN
- `codes/graph_occ.py` — from exp056

**Config keys**

- `imp_peak_dim`: 8
- `graph_hidden_dim`, `graph_num_layers`, `graph_dropout` — shared by occ + imp GNNs

### 🛠️ Verification & Execution Results

```bash
# Forward smoke
latent 65 = shared 42 + peak 8 + spatial 15; encode/decode/layout_latent OK

# Train
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m experiments.exp057_structured_graph.codes.train_vae_simple
```

- Model ~10.7M params; dataloader 582,997 samples — loads OK
- **Cannot** resume exp056 checkpoints (imp encoder/decoder + latent layout differ)

