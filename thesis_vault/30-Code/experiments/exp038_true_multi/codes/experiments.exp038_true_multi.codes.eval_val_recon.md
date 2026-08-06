---
title: eval_val_recon
type: code
path: experiments/exp038_true_multi/codes/eval_val_recon.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 163
tags: [code, exp038_true_multi, runnable]
---

# eval_val_recon

> Validation reconstruction metrics for exp038.

**Source:** `experiments/exp038_true_multi/codes/eval_val_recon.py` · 163 lines
**Experiment:** [[exp038_true_multi]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp038_true_multi/codes/eval_val_recon.py`

## Purpose

```text
Validation reconstruction metrics for exp038.

Purpose:
    Report occupancy BCE, slot accuracy, K-match, and impedance peak MSE on the val set.

Run:
    python experiments/exp038_true_multi/codes/eval_val_recon.py

Agent notes:
    - What: Full val-set reconstruction diagnostic (occ + imp heads) for a saved checkpoint.
    - Usage: Set ``CHECKPOINT_NAME`` in CONFIG → run. Prints per-metric summary to stdout.
    - Config keys:
        - ``CHECKPOINT_NAME`` — filename under ``checkpoints/`` (e.g. ``last_model.pt``)
        - ``MAX_BATCHES`` — limit val batches; ``0`` = all
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
