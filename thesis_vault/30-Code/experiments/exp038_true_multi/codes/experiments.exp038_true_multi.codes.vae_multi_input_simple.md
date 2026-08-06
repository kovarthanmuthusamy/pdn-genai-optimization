---
title: vae_multi_input_simple
type: code
path: experiments/exp038_true_multi/codes/vae_multi_input_simple.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 695
tags: [code, exp038_true_multi]
---

# vae_multi_input_simple

**Source:** `experiments/exp038_true_multi/codes/vae_multi_input_simple.py` · 695 lines
**Experiment:** [[exp038_true_multi]]

## Constants

| Name | Value |
|------|-------|
| `IMP_DROPOUT` | `0.15` |

## Classes

- **`DeepMLP(nn.Module)`** — Deep MLP with LayerNorm, SiLU activations, and optional dropout.
- **`SelfAttn2d(nn.Module)`** — Lightweight spatial self-attention for 2D feature maps (B, C, H, W).
- **`MultiInputVAE(nn.Module)`** — Multi-modal VAE with a shared latent space + heatmap-private tail dims.

## Imports

- [[experiments.exp038_true_multi.codes.freq_conditioning]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[pi_freq_utils]]

## Imported by

- [[exp041_eval_common]]
- [[experiments.exp038_true_multi.codes.eval_val_recon]]
- [[experiments.exp038_true_multi.codes.inference_vae]]
- [[experiments.exp038_true_multi.codes.save_epoch1_losses]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.eval_val_recon]]
- [[experiments.exp039_improved_heatmap.codes.codes.inference_vae]]
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.exp039_eval_common]]
- [[experiments.exp039_improved_heatmap.codes.inference_vae]]
- [[experiments.exp039_improved_heatmap.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.eval_val_recon]]
- [[experiments.exp040.codes.codes.inference_vae]]
- [[experiments.exp040.codes.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.exp039_eval_common]]
- [[experiments.exp040.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.eval_val_recon]]
- [[experiments.exp041.codes.codes.inference_vae]]
- [[experiments.exp041.codes.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.inference_vae]]
- [[experiments.exp041.codes.save_epoch1_losses]]
- [[experiments.exp042.codes.vae_poe_freq]]
- [[experiments.exp044.codes.vae_poe_freq]]
- [[optimize]]
- [[vae_factorized_freq]]

## External dependencies

`src_vae`, `torch`
