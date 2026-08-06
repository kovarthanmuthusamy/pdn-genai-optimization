---
title: eval_cross_freq
type: code
path: experiments/exp038_true_multi/codes/eval_cross_freq.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 247
tags: [code, exp038_true_multi, runnable]
---

# eval_cross_freq

> Evaluate native vs cross-frequency heatmap reconstruction on val set.

**Source:** `experiments/exp038_true_multi/codes/eval_cross_freq.py` · 247 lines
**Experiment:** [[exp038_true_multi]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp038_true_multi/codes/eval_cross_freq.py`

## Purpose

```text
Evaluate native vs cross-frequency heatmap reconstruction on val set.

Purpose:
    Measure foreground MSE when decoding at native MHz vs cross-anchor MHz (layout-z diagnostic).

Run:
    python experiments/exp038_true_multi/codes/eval_cross_freq.py

Agent notes:
    - What: Cross-frequency generalization eval for exp038 VAE heatmap decoder.
    - Usage: Set ``CHECKPOINT``, ``OFF_ANCHOR_MHZ``, ``MAX_BATCHES`` → run. Writes CSV metrics.
    - Config keys:
        - ``CHECKPOINT`` — ``.pt`` path (default ``checkpoints/last_model.pt``)
        - ``MAX_BATCHES`` — val batches to score; ``0`` = full val set
        - ``OUTPUT_CSV`` — where to write per-MHz MSE rows
        - ``OFF_ANCHOR_MHZ`` — MHz list for off-anchor decode test
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((p for p in _ROOT.parents if (p / 'src_vae').is_dir() and (p / 'experiments').is_dir…` |
| `EXP_DIR` | `PROJECT_ROOT / 'experiments/exp038_true_multi'` |
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

## Imported by

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.evaluate_vae]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.eval_cross_freq]]
- [[experiments.exp040.codes.evaluate_vae]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.evaluate_vae]]
- [[experiments.exp042.codes.evaluate_vae]]
- [[experiments.exp043.codes.evaluate_vae]]
- [[experiments.exp044.codes.evaluate_vae]]
- [[experiments.exp045.codes.evaluate_vae]]
- [[experiments.exp045.codes.train_vae_simple]]
- [[experiments.exp046.codes.train_vae_simple]]
- [[experiments.exp047.codes.train_vae_simple]]
- [[experiments.exp048.codes.train_vae_simple]]
- [[experiments.exp049.codes.train_vae_simple]]
- [[experiments.exp050.codes.train_vae_simple]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
