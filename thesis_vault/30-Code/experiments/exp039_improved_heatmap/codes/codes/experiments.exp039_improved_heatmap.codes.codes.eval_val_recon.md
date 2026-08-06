---
title: eval_val_recon
type: code
path: experiments/exp039_improved_heatmap/codes/codes/eval_val_recon.py
group: experiments/exp039_improved_heatmap/codes/codes
experiment: exp039_improved_heatmap
loc: 155
tags: [code, exp039_improved_heatmap, runnable]
---

# eval_val_recon

> Validation reconstruction metrics for exp038 (occ accuracy + impedance peaks).

**Source:** `experiments/exp039_improved_heatmap/codes/codes/eval_val_recon.py` · 155 lines
**Experiment:** [[exp039_improved_heatmap]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp039_improved_heatmap/codes/codes/eval_val_recon.py`

## Purpose

```text
Validation reconstruction metrics for exp038 (occ accuracy + impedance peaks).

Run:
    python experiments/exp039_improved_heatmap/codes/eval_val_recon.py

Run:
    python experiments/exp039_improved_heatmap/codes/eval_val_recon.py
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir() and (p / 'experiments').i…` |
| `CHECKPOINT_NAME` | `'last_model.pt'` |
| `MAX_BATCHES` | `0` |

## Functions

- **`_imp_ch0(x: torch.Tensor)`**
- **`_decode_occ_topk(logits: torch.Tensor, k_per_sample: torch.Tensor)`**
- **`_predict_k(logits: torch.Tensor)`** — K from sigmoid mass (rounded), clamped to [1, 52].
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.impedance_spectrum_loss]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]

## External dependencies

`torch`
