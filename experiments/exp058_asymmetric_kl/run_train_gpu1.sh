#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 TORCH_CPU_THREADS=8
export VAE_EXPERIMENT_DIR="$ROOT/experiments/exp058_asymmetric_kl"
ROOT_LOG_DIR="$VAE_EXPERIMENT_DIR/logs"
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp058_fresh_${STAMP}.log"
echo "exp058 fresh training on GPU ${CUDA_VISIBLE_DEVICES}"
echo "Log: $LOG"
nohup python -m experiments.exp058_asymmetric_kl.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp058_train.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp058_train.pid)"
