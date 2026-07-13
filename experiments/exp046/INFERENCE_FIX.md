# exp046 inference fix (2026-06-17)

## Problem

`run_multifreq_sweep_pipeline.py` failed at STEP 1 with:

```
FileNotFoundError: normalization_stats.json not found
```

## Cause

- `experiments/exp046/codes/inference_vae.py` did not exist.
- The sweep script fell back to `experiments.exp038_true_multi.codes.inference_vae`.
- exp038 reads stats from `/home/ubuntu/gan/datasets/data_multifreq_norm/` (missing file).
- exp046 was trained on `/home/ubuntu/gan/data_multi_norm/` where `normalization_stats.json` exists.

## Fix

Added `experiments/exp046/codes/inference_vae.py` (based on exp045) with:

- `data_dir` from exp046 `config.yaml` → `/home/ubuntu/gan/data_multi_norm`
- exp046 model class (`MultiInputVAEPoeFreq` with U-Net skips + occ spatial tower)
- `MODEL_LATENT_DIM = 48`

## Verify

```bash
cd ~/gan
python scrap/orchestration/run_multifreq_sweep_pipeline.py
```

Smoke test (optional):

```bash
PYTHONPATH=/home/ubuntu/gan python -c "
from experiments.exp046.codes.inference_vae import VAEInference, _norm_stats_path
import torch
print(_norm_stats_path())
VAEInference('experiments/exp046/checkpoints/last_model.pt', device=torch.device('cuda'))
"
```
