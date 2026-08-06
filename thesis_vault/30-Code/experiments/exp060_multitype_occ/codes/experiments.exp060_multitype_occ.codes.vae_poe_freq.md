---
title: vae_poe_freq
type: code
path: experiments/exp060_multitype_occ/codes/vae_poe_freq.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 299
tags: [code, exp060_multitype_occ, uncommitted]
---

# vae_poe_freq

> Multi-input VAE — freq PoE private dims + encode-first (no U-Net skips).

**Source:** `experiments/exp060_multitype_occ/codes/vae_poe_freq.py` · 299 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with PI_freq expert on private dims (no U-Net skips).

## Imports

- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]

## External dependencies

`torch`
