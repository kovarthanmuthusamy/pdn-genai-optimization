#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 TORCH_CPU_THREADS=8
ROOT_LOG_DIR=logs
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp053_resume450_peakloc_${STAMP}.log"
echo "exp053 resume ep450→600 (P1+P3 peak boost) on GPU ${CUDA_VISIBLE_DEVICES}"
echo "Log: $LOG"
nohup .venv/bin/python -m experiments.exp053_peak_log1p_losses.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp053_resume450.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp053_resume450.pid)"
