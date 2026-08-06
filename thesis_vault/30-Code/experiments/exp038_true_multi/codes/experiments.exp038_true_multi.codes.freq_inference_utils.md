---
title: freq_inference_utils
type: code
path: experiments/exp038_true_multi/codes/freq_inference_utils.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 121
tags: [code, exp038_true_multi]
---

# freq_inference_utils

> PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors).

**Source:** `experiments/exp038_true_multi/codes/freq_inference_utils.py` · 121 lines
**Experiment:** [[exp038_true_multi]]

## Functions

- **`bracket_anchors_mhz(mhz: float, anchors: Sequence[float]=ANCHOR_MHZ)`** — Return (mhz_lo, mhz_hi, t) with t in [0,1] for log-spaced blend weight on mhz_hi.
- **`is_training_anchor(mhz: float, tol: float=0.5)`**
- **`interp_anchor_value(mhz: float, table: dict[float, float])`**
- **`load_anchor_fg_max_table(path: Path | None=None)`**
- **`_heatmap_plane(hm: np.ndarray)`** — Return (H, W) for masking — accepts (H, W) or (1, H, W) / (C, H, W).
- **`calibrate_heatmap_physical(hm_phys: np.ndarray, mask: np.ndarray, target_fg_max: float, *, also_p95: float | None=None, p95_blend: float=0.35, scale_up_only: bool=False)`** — Scale physical heatmap so foreground max (and optionally p95) match targets.
- **`pi_norm_tensor(mhz: float, batch: int, device: torch.device)`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.vae_multi_input_simple]]
- [[experiments.exp040.codes.codes.vae_multi_input_simple]]
- [[experiments.exp041.codes.codes.vae_multi_input_simple]]
- [[inference_pool]]
- [[robust_stats]]
- [[run_multifreq_heatmap_sweep]]
- [[vae_factorized_freq]]

## External dependencies

`numpy`, `src_vae`, `torch`
