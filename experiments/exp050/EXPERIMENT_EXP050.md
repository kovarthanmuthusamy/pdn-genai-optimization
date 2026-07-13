# exp050 — Tier A Lean Heatmap Training

## Loss stack (only these)

| Term | Config | Role |
|------|--------|------|
| FG huber | (in `heatmap_loss_tier_a`) | Anchor full hotspot |
| Grad vector | `heatmap_grad_vector_weight: 2.5` | FG-masked ∂x/∂y |
| Grad direction | `heatmap_grad_direction_weight: 1.5` | Flow angle where |∇T|>0.08 |
| Phys Ω blob | `heatmap_peak_phys_weight: 3.0` | Top-k under + overshoot |
| peak_loc | `heatmap_peak_loc_weight: 0.75` | Hotspot position |
| latent_distill | `latent_distill_weight: 2.0` | Layout teacher |

All scaled by `heatmap_weight: 5.5` except phys blob and distill (added separately).

## Removed (not in code)

Percentile losses, z dynrange blob, intensity peak, lap/contrast/bg, layout sharpen,
Pearson training, grad magnitude sub-term.

## Train

```bash
cd /home/ubuntu/gan
.venv/bin/python -m experiments.exp050.codes.train_vae_simple
```

Dataset: `data_multi_norm_unbounded`

## Resume run (epoch 300 -> 700)

num_epochs: 700, resume_checkpoint: 300, stronger heatmap weights.
