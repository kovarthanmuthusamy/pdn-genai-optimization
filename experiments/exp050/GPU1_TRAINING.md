# exp050 GPU 1 Training Resume (epoch 150)

## Changes applied (2026-06-22)

### GPU
- Training runs on **physical GPU 1 only** via `CUDA_VISIBLE_DEVICES=1`.
- PyTorch sees it as `cuda:0`; GPU 0 is free for other work.

### CPU bottleneck reductions
- `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=8`
- `TORCH_CPU_THREADS=8` with `torch.set_num_threads(8)` in `train_vae_simple.main()`
- `compile: false` (already set; Volta GV100 does not benefit from torch.compile)
- `num_workers: 0` with `cache_in_ram: true` (data preloaded in RAM)

### Resume
- `config.yaml`: `"resume_checkpoint": 150`
- Loads `experiments/exp050/checkpoints/checkpoint_epoch_150.pt`
- Continues epochs 151 through 500

## Launch

```bash
cd /home/ubuntu/gan
./experiments/exp050/run_train_gpu1.sh
```

Or manually:

```bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_CPU_THREADS=8
cd /home/ubuntu/gan
.venv/bin/python -m experiments.exp050.codes.train_vae_simple
```

## Monitor

```bash
tail -f experiments/exp050/logs/train_gpu1_resume150_*.log
watch -n2 nvidia-smi
cat experiments/exp050/logs/train_gpu1.pid   # current PID
```

## Stop

```bash
kill $(cat experiments/exp050/logs/train_gpu1.pid)
```