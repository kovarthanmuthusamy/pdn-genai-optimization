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
ROOT_LOG_DIR=logs
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp051_train_gpu1_${STAMP}.log"
echo "Starting exp051 (expanded train data) on GPU 1"
echo "Log: $LOG"
nohup .venv/bin/python -m experiments.exp051_new_datas_appended.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp051_train_gpu1.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp051_train_gpu1.pid)"
