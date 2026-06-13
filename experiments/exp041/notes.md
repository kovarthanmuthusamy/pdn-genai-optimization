# Experiment: exp041

## Goal

Train the exp039-style multifreq PI VAE on the **inverse-K subsampled** dataset
(~195k rows, ~19.5k layouts). Same architecture and training stack as exp039;
dataset generation already upweights low-K diversity.

## Dataset

- Path: `datasets/data_multifreq_norm` (subsampled via `datasets/subsample_multifreq_inverse_k.py`)
- Policy: τ_low=35 for K≤20, τ=12 for K>20, n_min=150

## Changes vs exp039

- Fresh training (no `resume_checkpoint` in config)
- `k_balance_power`: **0.35** (lighter train-time K rebalance — dataset is pre-weighted)
- `train_samples_per_epoch`: **35000** (~175k train rows after 90% split)
- `learning_rate`: **5e-5** from epoch 1

## Train

```bash
python experiments/exp041/codes/train_vae_simple.py
```

## Eval / latent

```bash
python experiments/exp041/codes/evaluate_vae.py
python experiments/exp041/codes/visualize_latent.py
```

## Results


## Decision
