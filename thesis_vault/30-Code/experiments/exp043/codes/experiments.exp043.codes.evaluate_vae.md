---
title: evaluate_vae
type: code
path: experiments/exp043/codes/evaluate_vae.py
group: experiments/exp043/codes
experiment: exp043
loc: 373
tags: [code, exp043, runnable]
---

# evaluate_vae

> Post-training VAE diagnostics for exp043 (PI_freq-conditioned multifreq model).

**Source:** `experiments/exp043/codes/evaluate_vae.py` · 373 lines
**Experiment:** [[exp043]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp043/codes/evaluate_vae.py`

## Purpose

```text
Post-training VAE diagnostics for exp043 (PI_freq-conditioned multifreq model).

Run:
    python experiments/exp043/codes/evaluate_vae.py
```

## Constants

| Name | Value |
|------|-------|
| `_CODES` | `Path(__file__).resolve().parent` |
| `_ROOT` | `_CODES.parents[2]` |
| `K_TEST_PRESENT` | `26` |
| `K_TEST_RARE` | `2` |
| `TEST_MHZ` | `(63.0, 200.0, 400.0)` |
| `OFF_ANCHOR_MHZ` | `(100.0, 175.0, 350.0)` |
| `N_VARIATION` | `50` |
| `N_NN_TRAIN` | `500` |
| `N_PRIOR` | `200` |
| `INTERP_STEPS` | `8` |
| `TEMPS` | `[0.6, 0.8, 1.0, 1.2, 1.5]` |
| `TEST_MHZ_LIST` | `list(TEST_MHZ)` |
| `OFF_ANCHOR_MHZ_LIST` | `list(OFF_ANCHOR_MHZ)` |
| `MAX_VAL_BATCHES` | `30` |
| `SKIP_OFF_ANCHOR` | `False` |

## Functions

- **`test_variation(model, latent_stats, device, out_dir: Path, mhz_list: tuple[float, ...])`**
- **`test_nn_distance(model, latent_stats, device, data_dir, out_dir: Path, mhz: float=200.0)`**
- **`test_interpolation(model, device, data_dir, paths, mhz: float, out_dir: Path)`**
- **`test_prior_sampling(model, latent_stats, device, paths, mhz: float, out_dir: Path)`**
- **`test_cross_freq_same_z(model, device, data_dir, out_dir: Path, mhz_native: float=200.0, mhz_alt: float=250.0)`** — Decode the same posterior z at two MHz — measures freq conditioning on decode.
- **`run_off_anchor(model, cfg, device, out_dir: Path, mhz_list: tuple[float, ...], max_batches: int)`**
- **`main()`**

## Imports

- [[dataloader]]
- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp043.codes.dataloader_multifreq]]
- [[pi_freq_utils]]

## External dependencies

`exp043_eval_common`, `matplotlib`, `numpy`, `src_vae`, `torch`
