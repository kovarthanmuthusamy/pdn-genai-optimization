---
title: eval_cross_freq
type: code
path: experiments/exp041/codes/codes/eval_cross_freq.py
group: experiments/exp041/codes/codes
experiment: exp041
loc: 237
tags: [code, exp041, runnable]
---

# eval_cross_freq

> Evaluate native vs cross-frequency heatmap reconstruction on the val set.

**Source:** `experiments/exp041/codes/codes/eval_cross_freq.py` · 237 lines
**Experiment:** [[exp041]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp041/codes/codes/eval_cross_freq.py`

## Purpose

```text
Evaluate native vs cross-frequency heatmap reconstruction on the val set.

Run:
    python experiments/exp041/codes/eval_cross_freq.py

Run:
    python experiments/exp041/codes/eval_cross_freq.py
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((p for p in _ROOT.parents if (p / 'src_vae').is_dir() and (p / 'experiments').is_dir…` |
| `EXP_DIR` | `PROJECT_ROOT / 'experiments/exp041'` |
| `DEFAULT_CKPT` | `EXP_DIR / 'checkpoints/last_model.pt'` |
| `OUT_CSV` | `EXP_DIR / 'metrics/cross_freq_eval.csv'` |
| `DEFAULT_OFF_ANCHOR_MHZ` | `(80.0, 250.0)` |
| `CHECKPOINT` | `DEFAULT_CKPT` |
| `MAX_BATCHES` | `0` |
| `OUTPUT_CSV` | `OUT_CSV` |
| `OFF_ANCHOR_MHZ` | `list(DEFAULT_OFF_ANCHOR_MHZ)` |

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float=0.5)`**
- **`run_off_anchor_eval(model: torch.nn.Module, val_loader, *, bg: float, off_anchor_mhz: tuple[float, ...]=DEFAULT_OFF_ANCHOR_MHZ, max_batches: int=30, device: str | torch.device='cuda', out_csv: Path | str | None=None)`** — Layout-z decode at off-anchor MHz; compare to native GT (cross-freq diagnostic).
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.inference_vae]]
- [[pi_freq_utils]]

## External dependencies

`src_vae`, `torch`
