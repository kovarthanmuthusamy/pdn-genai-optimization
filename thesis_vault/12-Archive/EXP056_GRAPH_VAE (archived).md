---
title: EXP056_GRAPH_VAE (archived)
type: archive
source: docs/_archive/EXP056_GRAPH_VAE.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# exp056 — Graph VAE (occupancy GNN)

### 📝 Summary of Changes

- Added **`experiments/exp056_graph_vae/`** as a copy of `exp055_hard_occ` with **architecture-only** change: occupancy encoder/decoder use graph message passing on the 52-slot PCB grid.
- New module: `experiments/exp056_graph_vae/codes/graph_occ.py` (`OccGraphEncoder`, `OccGraphDecoder`).
- **All loss terms, weights, training loop, and config keys** match exp055 (heatmap, impedance spectrum, occupancy BCE/focal, KL, cross-freq, physics, etc.).
- Heatmap U-Net path, impedance MLP, PoE fusion, layout-private head unchanged.

### 🚀 Implementation Details

**Graph topology**

- 52 nodes = decap slots (`LABELS_ORDERED` / `_LABEL_TO_COORD` from `libs/data_creation/occupancy.py`).
- Edges = 4-neighbors on the 7×8 board grid + self-loops; symmetric normalized adjacency.
- Pure PyTorch GNN (3 `GraphConvLayer`s); no `torch_geometric`.

**Architecture swap (only occupancy path)**

| Component | exp055 | exp056 |
|-----------|--------|--------|
| `occupancy_encoder` | MLP 52→64 | `OccGraphEncoder` → 64-d pool |
| `occupancy_decoder` | MLP → 52 logits | `OccGraphDecoder` per-node logits |
| Heatmap / imp / PoE | same | same |

**Config** (`experiments/exp056_graph_vae/config.yaml`)

- `experiment_dir`: `experiments/exp056_graph_vae`
- `data_dir`: `datasets/data_multifreq_train_norm_unbounded`
- `graph_hidden_dim`: 128, `graph_num_layers`: 3, `graph_dropout`: 0.1
- `resume_checkpoint`: null (fresh run)

### 🛠️ Verification & Execution Results

```bash
# Forward smoke test
python -c "from experiments.exp056_graph_vae.codes.exp056_common import load_yaml_config, vae_model_kwargs; ..."

# Train (GPU 1)
cd /home/ubuntu/genai_pdn
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m experiments.exp056_graph_vae.codes.train_vae_simple
```

- Model forward: `z (B,65)`, `occ_r (B,52)`, `imp_r (B,1,231)`, `hm_r (B,1,64,64)` — OK
- Total params ~12.1M (occ graph enc ~104k, dec ~128k)
- Training entry loads 582,997 samples (K≤30 unbounded dataset), builds loaders — OK

**Note:** Cannot load exp055 checkpoints into exp056 (occupancy encoder/decoder keys differ).
