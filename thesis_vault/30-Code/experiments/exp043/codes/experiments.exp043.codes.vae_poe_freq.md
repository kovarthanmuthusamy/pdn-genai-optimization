---
title: vae_poe_freq
type: code
path: experiments/exp043/codes/vae_poe_freq.py
group: experiments/exp043/codes
experiment: exp043
loc: 234
tags: [code, exp043]
---

# vae_poe_freq

> Multi-input VAE with PI_freq Product-of-Experts expert (exp043).

**Source:** `experiments/exp043/codes/vae_poe_freq.py` · 234 lines
**Experiment:** [[exp043]]

## Purpose

```text
Multi-input VAE with PI_freq Product-of-Experts expert (exp043).

Purpose:
    Extend ``MultiInputVAE`` with a frequency-only PoE expert fused with layout experts for multifreq training.

Run:
    Import only — instantiated by ``train_vae_simple``, ``inference_vae``, ``exp043_eval_common``.

Agent notes:
    - What: Core exp043 model class ``MultiInputVAEPoeFreq`` (shared latent + optional private heatmap dims).
    - Usage: Import and construct with ``latent_dim``, ``cond_dim``, ``use_freq_poe_expert``; not run directly.
    - Key symbols: ``MultiInputVAEPoeFreq``, ``encode_layout_latent``, ``encode_cross_modal``
    - Flag: ``use_freq_poe_expert`` (default True) enables MHz-conditioned PoE fusion.
```

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion: layout experts + optional PI_freq expert on full or private latent dims.

## Imports

- [[experiments.exp043.codes.vae_multi_input_simple]]

## Imported by

- [[_bench_train_step]]
- [[_breakdown_hm_loss]]
- [[_smoke_exp043_struct]]
- [[_test_expert_kl]]
- [[exp043_eval_common]]
- [[experiments.exp043.codes.inference_vae]]
- [[experiments.exp043.codes.train_vae_simple]]

## External dependencies

`torch`
