# Experiment: exp051_new_datas_appended

## Goal

Fine-tune the exp050 Tier-A VAE on the expanded multifreq training set
(`datasets/data_multifreq_train_norm_robust`, ~472k manifest rows including
combinations-appended layouts).

## Changes vs exp050

- `balance_k=false`, `balance_freq=false` — dataset already inverse-K sampled and 16-anchor uniform; avoid double reweighting

- `data_dir` → `datasets/data_multifreq_train_norm_robust` (robust per-MHz log norm)
- Resume from `experiments/exp050/checkpoints/checkpoint_epoch_700.pt`
- Train epochs 701–1000 (`recalculate_curriculum_on_resume=true`)
- All training/eval imports point at `experiments.exp051_new_datas_appended.codes`

## Run

```bash
cd /home/ubuntu/genai_pdn
.venv/bin/python -m experiments.exp051_new_datas_appended.codes.train_vae_simple
# or GPU 1:
./experiments/exp051_new_datas_appended/run_train_gpu1.sh
```

## Results

(TBD)

## Decision

(TBD)
