# exp054: DDP + batch size speedup

---

## 📝 Summary of Changes

- Added `experiments/exp054_K_30/codes/distributed_train.py` — DDP init, model wrap/unwrap, LR scaling, rank-0 helpers.
- Patched `train_core.py` — 2-GPU training loop, val/checkpoint/logging on rank 0, broadcast val loss for LR scheduler sync.
- Patched `dataloader_multifreq.py` — per-rank train draws (`train_samples_per_epoch // world_size`).
- Patched `run_epoch_encode.py` — `unwrap_model()` for DDP + compile.
- Updated `config.yaml`:
  - `batch_size`: 160 → **224** (per GPU)
  - `val_batch_size`: 160 → **320**
  - `cache_in_ram`: false (DDP avoids duplicating host RAM)
  - `num_workers`: 4
  - `use_ddp`, `ddp_base_batch_size`, `ddp_linear_lr_scale`
- Added `experiments/exp054_K_30/run_train_ddp.sh` — `torchrun` on GPUs 0,1.

---

## 🚀 Implementation Details

### Launch

```bash
# 2× GV100 (default GPUs 0,1)
./experiments/exp054_K_30/run_train_ddp.sh

# Or manually
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
  -m experiments.exp054_K_30.codes.train_vae_simple
```

Single-GPU (no torchrun) still works — DDP is skipped when `WORLD_SIZE=1`.

### Effective batch & LR

| Setting | Value |
|---------|-------|
| Per-GPU batch | 224 |
| GPUs | 2 |
| **Global batch** | **448** |
| LR reference batch | 160 |
| LR scale | 2.8× |
| Base LR | 3e-5 → **8.4e-5** |

Train draws: **15,000 per rank** (30,000 global) × 224 batch ≈ **67 steps/epoch** (vs 187 @ batch 160 single-GPU).

### DDP behaviour

- Rank 0: logging, CSV, checkpoints, validation, off-anchor eval.
- All ranks: training forward/backward (gradients synced).
- `find_unused_parameters=True` for layout-path branches.
- Checkpoints save **unwrapped** `state_dict` (no `module.` prefix).

---

## 🛠️ Verification & Execution Results

```bash
python3 -c "from experiments.exp054_K_30.codes import train_core, distributed_train"
torchrun --nproc_per_node=2 --master_port=29502 -m experiments.exp054_K_30.codes.train_vae_simple
```

Smoke run reached **DDP wrap on 2 GPUs** with:
- `draws=15000/508060 rank=0/2` and `rank=1/2`
- `DDP: world_size=2  global_batch=448  lr=8.40e-05`

Full epoch timing not benchmarked in this step.
