#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 TORCH_CPU_THREADS=8
ROOT_LOG_DIR=logs
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp057_K30_fresh_${STAMP}.log"
echo "exp057 fresh 400-epoch training on GPU ${CUDA_VISIBLE_DEVICES}"
echo "Log: $LOG"
nohup .venv/bin/python -m experiments.exp057_structured_graph.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp057_K30_train.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp057_K30_train.pid)"
