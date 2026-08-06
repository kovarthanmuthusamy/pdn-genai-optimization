---
title: TRAINING_GPU_UTILIZATION (archived)
type: archive
source: docs/_archive/TRAINING_GPU_UTILIZATION.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Training: high CPU / low GPU utilization

## What we observed (exp042, GV100)

| Signal | Value |
|--------|--------|
| GPU util | ~16% (32 GB card, ~11 GB used) |
| Main train process CPU | ~114% (single-thread bound) |
| DataLoader workers | 12 processes, mostly idle |
| Train time | **~750–800 s/epoch** (~3.8 s/batch @ bs=96, 20k draws) |

The GPU is waiting on the training loop, not the other way around.

## Root causes

1. **GPU sync every batch** — `losses[k].item()` for ~15 loss keys forced a device sync after each forward/backward.
2. **`torch.quantile` in heatmap loss** — full sort per batch for p95/p99 dyn-range terms (slow on Volta).
3. **12 `num_workers` with `cache_in_ram: true`** — all ~195k samples already in RAM; extra workers add fork overhead and memory without faster I/O.
4. **`torch.cuda.empty_cache()` every epoch** — extra synchronization.
5. **Small effective GPU work** — batch 96 on a 32 GB GV100 leaves headroom; larger batches improve utilization.
6. **Volta (GV100)** — `compile` and `tf32` are correctly off; `fp16` + GradScaler is appropriate.

## Code changes (2026-06-11)

In `experiments/exp038_true_multi/codes/train_vae_simple.py` (used by exp042):

- Accumulate train losses on GPU; single `.item()` sync at end of epoch.
- Replace `torch.quantile` with fast `kthvalue` percentiles in heatmap losses.
- Auto-cap `num_workers` to **2** when `cache_in_ram=True` (override: `VAE_NUM_WORKERS=8`).
- `empty_cache_interval: 25` instead of every epoch.
- PI-freq jitter without CPU branch via `torch.where`.

In `dataloader_multifreq.py`:

- Faster cross-freq collate (numpy alt-index lookup).

In `experiments/exp042/config.yaml` (for next restart):

- `batch_size`: 96 → **160**
- `num_workers`: 12 → **2**
- `prefetch_factor`: 6 → **2**
- `empty_cache_interval`: **25**

## Apply changes

**Restart training** to pick up code + config (running process does not hot-reload):

```bash
# Stop current run (Ctrl+C), then resume:
cd /home/ubuntu/genai_pdn
source .venv/bin/activate
python experiments/exp042/codes/train_vae_simple.py
```

`resume_checkpoint` in config will continue from epoch 300.

## Further tuning (if GPU still &lt; 50%)

| Knob | Suggestion |
|------|------------|
| `batch_size` | Try **192** or **224** if memory allows (`nvidia-smi` during train) |
| `cross_modal_update_freq` | 4 → **8** (fewer extra forward passes) |
| `train_samples_per_epoch` | 20000 → **15000** for faster epochs (optional) |
| `layout_train_prob` | 0.92 is high (extra encode path); 0.7–0.85 if quality allows |
| `VAE_NUM_WORKERS` | Try **0** or **4** and compare epoch time in `metrics/epoch_timing.csv` |
| Second GPU | Set `CUDA_VISIBLE_DEVICES=0` so only one card is used (avoid idle 2nd process) |

## Monitor

```bash
watch -n1 nvidia-smi
tail -f experiments/exp042/metrics/epoch_timing.csv
```

Target after fixes: **train_sec** per epoch dropping toward **200–400 s** (hardware dependent).
