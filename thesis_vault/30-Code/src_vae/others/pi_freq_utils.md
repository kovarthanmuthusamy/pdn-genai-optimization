---
title: pi_freq_utils
type: code
path: src_vae/others/pi_freq_utils.py
group: src_vae/others
loc: 82
tags: [code, src_vae]
---

# pi_freq_utils

> PI frequency conditioning: MHz/Hz → model scalar in [0, 1].

**Source:** `src_vae/others/pi_freq_utils.py` · 82 lines

## Purpose

```text
PI frequency conditioning: MHz/Hz → model scalar in [0, 1].

Run: ``from src_vae.others.pi_freq_utils import pi_freq_to_norm, pi_freq_norm_for_model``.
```

## Constants

| Name | Value |
|------|-------|
| `PI_FREQ_MIN_HZ` | `1000000.0` |
| `PI_FREQ_MAX_HZ` | `600000000.0` |
| `_LOG10_MIN` | `math.log10(PI_FREQ_MIN_HZ)` |
| `_LOG10_RANGE` | `math.log10(PI_FREQ_MAX_HZ) - _LOG10_MIN` |

## Functions

- **`pi_freq_hz_to_norm(freq_hz: float)`**
- **`pi_freq_mhz_to_norm(freq_mhz: float)`** — Convert PI frequency in MHz to model conditioning scalar in [0, 1].
- **`pi_freq_to_norm(freq: Union[float, int, np.ndarray, torch.Tensor], *, unit: PiFreqUnit='mhz', device: torch.device | str | None=None, dtype: torch.dtype=torch.float32)`** — Convert PI frequency to (N,) normalised tensor for the VAE / physics critic.
- **`pi_freq_norm_for_model(freq: Union[float, int, np.ndarray, torch.Tensor] | None, batch_size: int, *, unit: PiFreqUnit='mhz', default_mhz: float=200.0, device: torch.device | None=None, dtype: torch.dtype=torch.float32)`** — Build (B,) PI_freq tensor; default operating point if ``freq`` is None.

## Imported by

- [[dataloader]]
- [[diagnose_exp044_sweep]]
- [[diagnose_sweep_vs_dataset]]
- [[eval_cross_freq_gmax]]
- [[exp041_eval_common]]
- [[exp042_eval_common]]
- [[exp043_eval_common]]
- [[exp044_eval_common]]
- [[exp045_eval_common]]
- [[exp046_eval_common]]
- [[exp047_eval_common]]
- [[exp048_eval_common]]
- [[exp049_eval_common]]
- [[exp050_eval_common]]
- [[exp051_eval_common]]
- [[exp052_eval_common]]
- [[exp053_eval_common]]
- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.eval_cross_freq]]
- [[experiments.exp039_improved_heatmap.codes.codes.freq_inference_utils]]
- [[experiments.exp039_improved_heatmap.codes.codes.physics_loss]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.evaluate_vae]]
- [[experiments.exp039_improved_heatmap.codes.exp039_eval_common]]
- [[experiments.exp040.codes.codes.eval_cross_freq]]
- [[experiments.exp040.codes.codes.freq_inference_utils]]
- [[experiments.exp040.codes.codes.physics_loss]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.codes.vae_multi_input_simple]]
- [[experiments.exp040.codes.evaluate_vae]]
- [[experiments.exp040.codes.exp039_eval_common]]
- [[experiments.exp041.codes.codes.eval_cross_freq]]
- [[experiments.exp041.codes.codes.freq_inference_utils]]
- [[experiments.exp041.codes.codes.physics_loss]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.codes.vae_multi_input_simple]]
- [[experiments.exp041.codes.evaluate_vae]]
- [[experiments.exp042.codes.evaluate_vae]]
- [[experiments.exp043.codes.evaluate_vae]]
- [[experiments.exp043.codes.freq_inference_utils]]
- [[experiments.exp043.codes.physics_loss]]
- [[experiments.exp043.codes.train_vae_simple]]
- [[experiments.exp043.codes.vae_multi_input_simple]]
- [[experiments.exp044.codes.evaluate_vae]]
- [[experiments.exp044.codes.latent_optimization_joint]]
- [[experiments.exp045.codes.eval_spatial_metrics]]
- [[experiments.exp045.codes.evaluate_vae]]
- [[experiments.exp045.codes.latent_optimization_joint]]
- [[experiments.exp045.codes.vae_multi_input_simple]]
- [[experiments.exp046.codes.eval_spatial_metrics]]
- [[experiments.exp046.codes.vae_multi_input_simple]]
- [[experiments.exp047.codes.eval_spatial_metrics]]
- [[experiments.exp047.codes.run_epoch_encode]]
- [[experiments.exp047.codes.vae_multi_input_simple]]
- [[experiments.exp048.codes.eval_spatial_metrics]]
- [[experiments.exp048.codes.run_epoch_encode]]
- [[experiments.exp048.codes.vae_multi_input_simple]]
- [[experiments.exp049.codes.eval_spatial_metrics]]
- [[experiments.exp049.codes.run_epoch_encode]]
- [[experiments.exp049.codes.vae_multi_input_simple]]
- [[experiments.exp050.codes.eval_spatial_metrics]]
- [[experiments.exp050.codes.run_epoch_encode]]
- [[experiments.exp050.codes.vae_multi_input_simple]]
- [[experiments.exp051_new_datas_appended.codes.eval_spatial_metrics]]
- [[experiments.exp051_new_datas_appended.codes.run_epoch_encode]]
- [[experiments.exp051_new_datas_appended.codes.vae_multi_input_simple]]
- [[experiments.exp052_unbounded_pearson.codes.eval_spatial_metrics]]
- [[experiments.exp052_unbounded_pearson.codes.mhz_loss_weight]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_spatial_metrics]]
- [[experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]
- [[experiments.exp054_K_30.codes.eval_spatial_metrics]]
- [[experiments.exp054_K_30.codes.physics_loss]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.vae_multi_input_simple]]
- [[experiments.exp055_hard_occ.codes.eval_spatial_metrics]]
- [[experiments.exp055_hard_occ.codes.physics_loss]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]
- [[experiments.exp056_graph_vae.codes.eval_spatial_metrics]]
- [[experiments.exp056_graph_vae.codes.physics_loss]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]
- [[experiments.exp057_structured_graph.codes.eval_spatial_metrics]]
- [[experiments.exp057_structured_graph.codes.physics_loss]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]
- [[experiments.exp058_asymmetric_kl.codes.eval_spatial_metrics]]
- [[experiments.exp058_asymmetric_kl.codes.physics_loss]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]
- [[experiments.exp059_capacity_freq.codes.eval_spatial_metrics]]
- [[experiments.exp059_capacity_freq.codes.physics_loss]]
- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]
- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]]
- [[experiments.exp060_multitype_occ.codes.physics_loss]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]
- [[gmax_training_patch]]
- [[inference_pool]]
- [[inspect_sweep_artifacts]]
- [[norm_stats]]
- [[run_multifreq_heatmap_sweep]]
- [[sweep_latent_opt_rules]]
- [[sweep_qc_eval]]
- [[vae_factorized_freq]]

## External dependencies

`numpy`, `torch`
