---
title: freq_inference_utils
type: code
path: experiments/exp041/codes/codes/freq_inference_utils.py
group: experiments/exp041/codes/codes
experiment: exp041
loc: 112
tags: [code, exp041]
---

# freq_inference_utils

> PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors).

**Source:** `experiments/exp041/codes/codes/freq_inference_utils.py` · 112 lines
**Experiment:** [[exp041]]

## Functions

- **`bracket_anchors_mhz(mhz: float, anchors: Sequence[float]=ANCHOR_MHZ)`** — Return (mhz_lo, mhz_hi, t) with t in [0,1] for log-spaced blend weight on mhz_hi.
- **`is_training_anchor(mhz: float, tol: float=0.5)`**
- **`interp_anchor_value(mhz: float, table: dict[float, float])`**
- **`load_anchor_fg_max_table(path: Path | None=None)`**
- **`_heatmap_plane(hm: np.ndarray)`** — Return (H, W) for masking — accepts (H, W) or (1, H, W) / (C, H, W).
- **`calibrate_heatmap_physical(hm_phys: np.ndarray, mask: np.ndarray, target_fg_max: float, *, also_p95: float | None=None, p95_blend: float=0.35)`** — Scale physical heatmap so foreground max (and optionally p95) match targets.
- **`pi_norm_tensor(mhz: float, batch: int, device: torch.device)`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[pi_freq_utils]]

## External dependencies

`numpy`, `src_vae`, `torch`
