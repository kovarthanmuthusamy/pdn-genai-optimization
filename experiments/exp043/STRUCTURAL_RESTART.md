# exp043 structural restart (heatmap pattern)

Fresh training run — **do not resume** old checkpoints (architecture changed).

## Problem

Previous 4×4 heatmap bottleneck + dynrange/peak-heavy losses allowed edge-stripe shortcuts: high `gen_max` with `gen_p95≈0` and low spatial Pearson r (~0.24–0.35) even at training anchors.

## Structural changes

### 1. Heatmap encoder/decoder (`vae_multi_input_simple.py`)

| Component | Before | After |
|-----------|--------|-------|
| Bottleneck | 4×4 (128 ch) | **8×8** (128 ch) |
| `heatmap_enc2_conv` last conv | stride 2 | **stride 1** (keeps 8×8) |
| `heatmap_dec_deconv2` | single `ConvTranspose2d(16→1)` | `ConvTranspose2d(32→16)` + **`Conv2d(16→1)`** refine at 64×64 |

Spatial path: 64 → 16 → **8** (encode), **8** → 16 → 32 → 64 (decode).

### 2. Pattern losses (`gmax_heatmap_loss.py`)

- **`heatmap_pattern_weight`** — FG min-max normalized Pearson/cosine loss `(1−r)²` on spatial morphology.
- **`heatmap_spread_weight`** — penalizes sparse fields (low FG mean / p95) that cheat dynrange via thin edge spikes.

### 3. Config rebalance (`config.yaml`)

| Key | Old | New | Rationale |
|-----|-----|-----|-----------|
| `heatmap_dynrange_weight` | 4.0 | **2.0** | less max-only shortcut |
| `heatmap_peak_weight` | 4.25 | **2.5** | same |
| `heatmap_grad_weight` | 3.2 | **4.0** | sharper spatial structure |
| `heatmap_lap_weight` | 3.2 | **4.0** | smoother H-shaped fields |
| `heatmap_contrast_weight` | 1.5 | **2.5** | FG vs BG separation |
| `heatmap_pattern_weight` | — | **5.0** | direct pattern match |
| `heatmap_spread_weight` | — | **3.5** | anti empty-field |
| `cross_freq_layout_z_only` | — | **true** | matches sweep inference |
| `global_max_ohm` | hardcoded | **removed** | loaded from `normalization_stats.json` |
| `resume_checkpoint` | — | **null** | fresh start |

## How to start

```bash
cd /home/ubuntu/gan
python3 experiments/exp043/codes/train_vae_simple.py
```

Outputs go to `experiments/exp043/runs/run_<UTC>/` (checkpoints, logs, metrics). Old run artifacts are untouched.

## Verify before training

```bash
python3 scratch/_smoke_exp043_struct.py
```

Expected: `OK torch.Size([4, 1, 64, 64])` and finite loss.

## Files touched

- `experiments/exp043/codes/vae_multi_input_simple.py`
- `experiments/exp043/codes/gmax_heatmap_loss.py`
- `experiments/exp043/codes/train_vae_simple.py` (Config fields)
- `experiments/exp043/codes/gmax_training_patch.py` (startup log)
- `experiments/exp043/config.yaml`
