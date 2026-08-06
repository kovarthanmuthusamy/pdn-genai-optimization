---
title: EXP055_BINARY_OCCUPANCY (archived)
type: archive
source: docs/_archive/EXP055_BINARY_OCCUPANCY.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# EXP055: Binary occupancy decode (from exp054)

### 📝 Summary of Changes
- Created `experiments/exp055_K_30/` as self-contained copy of exp054 codes
- Added `occupancy_binary.py` — top-K binarization before heatmap decode
- Model `decode()` + off-anchor eval aligned to CAD-discrete layouts
- Fresh `config.yaml`: 400 epochs, no resume, `occupancy_binary_decode: true`

### 🚀 Implementation Details
- **Self-contained**: vendored `train_core`, dataloaders, losses, eval (same layout as exp054)
- **Binary decode**: soft 52-d probs → hard top-K before `_decode_heatmap_tensor` spatial conditioning
- **Config flags**: `occupancy_binary_decode`, `occupancy_binary_ste_train` (STE off by default)
- **External deps** (intentional): `src_vae.others.*`, `experiments/exp043/codes/freq_conditioning`

### 🛠️ Verification & Execution Results
```bash
python3 -c "from experiments.exp055_K_30.codes import train_vae_simple"
timeout 45 python -m experiments.exp055_K_30.codes.train_vae_simple
```
