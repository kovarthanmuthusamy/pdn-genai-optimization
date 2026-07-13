#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export TORCH_CPU_THREADS=8
LOG_DIR=experiments/exp050/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/train_gpu1_resume150_${STAMP}.log"
echo "Starting exp050 on GPU 1, resume epoch 150"
echo "Log: $LOG"
nohup .venv/bin/python -m experiments.exp050.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$LOG_DIR/train_gpu1.pid"
echo "PID: $(cat $LOG_DIR/train_gpu1.pid)"
