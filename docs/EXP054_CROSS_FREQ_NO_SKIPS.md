# EXP054: Cross-freq decode without native encode skips

## Problem

Off-anchor eval at epoch 600 showed **400 MHz encode_cross** was much worse than **layout_cross** (encode MSE 1.59 vs layout 0.85). Peak position was already good; the failure mode was amplitude/scale mismatch from injecting **native-MHz U-Net skips** when decoding at a different frequency.

## Change

New config flag `cross_freq_use_encode_skips` (default **false**):

- **Training** (`run_epoch_encode._cross_freq_decode`): when `cross_freq_use_encode_z` is true, still use the encode latent `z`, but pass `heatmap_skips=None` into decode unless this flag is true.
- **Off-anchor eval** (`eval_spatial_metrics` / `eval_off_anchor`): `encode_cross` decode matches training — no native skips when the flag is false.

`encode_native_teacher_skips` is unchanged; it only affects the main encode→decode teacher path (output distill), not cross-freq.

## Config

```json
"cross_freq_use_encode_z": true,
"cross_freq_use_encode_skips": false,
"encode_native_teacher_skips": true
```

## Files

- `experiments/exp054_K_30/config.yaml`
- `experiments/exp054_K_30/codes/train_vae_simple.py` — Config field
- `experiments/exp054_K_30/codes/run_epoch_encode.py` — `_cross_freq_decode`
- `experiments/exp054_K_30/codes/eval_off_anchor.py` — pass flag to spatial eval
- `experiments/exp054_K_30/codes/eval_spatial_metrics.py` — `use_encode_skips` param

## Resume training

No checkpoint format change. Resume from latest and continue toward 800 epochs:

```bash
./experiments/exp054_K_30/run_train_extend_800.sh
```

Watch `metrics/off_anchor_eval.csv` at the next interval (epoch 650, …) for improved `encode_cross` at 400 MHz.
