---
title: freq_inference_utils
type: code
path: experiments/exp043/codes/freq_inference_utils.py
group: experiments/exp043/codes
experiment: exp043
loc: 112
tags: [code, exp043]
---

# freq_inference_utils

> PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors).

**Source:** `experiments/exp043/codes/freq_inference_utils.py` · 112 lines
**Experiment:** [[exp043]]

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

## Imported by

- [[experiments.exp043.codes.vae_multi_input_simple]]
- [[experiments.exp045.codes.vae_multi_input_simple]]
- [[experiments.exp046.codes.vae_multi_input_simple]]
- [[experiments.exp047.codes.vae_multi_input_simple]]
- [[experiments.exp048.codes.vae_multi_input_simple]]
- [[experiments.exp049.codes.vae_multi_input_simple]]
- [[experiments.exp050.codes.vae_multi_input_simple]]
- [[experiments.exp051_new_datas_appended.codes.vae_multi_input_simple]]
- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]
- [[experiments.exp054_K_30.codes.vae_multi_input_simple]]
- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]
- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]
- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`numpy`, `src_vae`, `torch`
