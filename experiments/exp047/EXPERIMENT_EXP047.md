# exp047 — Fix Layout-to-Heatmap Path

Based on exp046 v3 architecture. Targets the layout-to-heatmap generation quality which is the actual inference use case (latent optimization produces a layout, then heatmaps are generated at peak frequencies).

## Problem (from exp046 eval)

Encode mode (autoencoder) achieves r > 0.997 at all frequencies, but layout mode (generation from occupancy+impedance only) suffers from:

1. **Magnitude explosion**: Gen_max hits z-clip ceiling (33.75 ohm) at 300-400 MHz
2. **Spatial degradation**: Pearson r drops from 0.998 to 0.87-0.91

## Changes from exp046

### 1. Output-level distillation (new)
On layout batches, also decode the teacher z (encode path) and compute pixel-level Huber loss between layout-decoded and teacher-decoded heatmaps. Config: `output_distill_weight: 2.0`.

### 2. Deeper layout_private_head
Replaced 2-layer MLP (128 hidden) with 3-layer MLP (256 -> 128 -> 15) for both mu and logvar. More capacity for frequency-dependent spatial mapping.

### 3. layout_train_prob increased to 0.50
Half the batches now train the layout path directly (was 0.30). Since layout mode IS the use case, it needs proportionally more gradient signal.

### 4. Tanh soft-clamp on private head output
`priv_mu = 4.0 * tanh(priv_mu / 4.0)` prevents extreme latent values that decode to clip-ceiling heatmaps.

### 5. Logvar distillation + stronger latent distill
- Distillation loss now includes logvar matching (weight 0.3)
- `latent_distill_weight` increased from 1.0 to 2.0

## Config diff from exp046

```
layout_train_prob:            0.30 -> 0.50
latent_distill_weight:        1.0  -> 2.0
latent_distill_logvar_weight: (new)   0.3
output_distill_weight:        (new)   2.0
```

## Architecture

Same as exp046 v3 (latent_dim=65, shared=50, private=15) with:
- Deeper `layout_private_mu`: Linear(144,256) -> LN -> LeakyReLU -> Linear(256,128) -> LN -> LeakyReLU -> Linear(128,15)
- Same for `layout_private_logvar`
- Tanh soft-clamp on private mu output

## Validation

Run `eval_real_data_sweep.py` after training to compare layout mode metrics against exp046 baseline.

---

## High-freq layout pass (exp047 continued training)

Targets per-sample failures at 300–400 MHz (blurry peaks, wrong hotspot, clip ceiling).

### Training changes

| Change | Config |
|--------|--------|
| **No early stop** | `early_stop_*` = 0, `num_epochs: 400` |
| **Sharper layout losses** | `layout_sharpening_weight: 1.0`, `layout_spatial_loss_mult: 1.75`, `layout_peak_grad_mult: 1.5`, higher `heatmap_peak/grad/pearson` weights |
| **High-freq loss boost** | `layout_high_freq_loss_mult: 2.0` when PI ≥ 250 MHz |
| **Cross-freq layout mix** | `cross_freq_layout_mix_prob: 0.45` — layout z for cross-freq decode |
| **High-freq cross-freq pairs** | `high_freq_pair_bias: 0.55`, `cross_freq_high_freq_loss_mult: 2.5` |
| **Wider private head** | `layout_private_hidden: 384` (was 256) |
| **Freq FiLM on private μ** | `use_layout_private_freq_film: true` |

### Resume

`resume_checkpoint: "latest"` — loads compatible weights from `last_model.pt`; new/wider private-head and FiLM layers init randomly (`strict=False`).

### Run

```bash
cd /home/ubuntu/gan
python experiments/exp047/codes/train_vae_simple.py
```

