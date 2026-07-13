#!/usr/bin/env bash
# Resume exp054 training from latest checkpoint → epochs 401–600 (after updating config.yaml).
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ROOT_LOG_DIR=logs
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp054_K30_extend600_${STAMP}.log"
echo "exp054 resume → 600 epochs on GPUs ${CUDA_VISIBLE_DEVICES}"
echo "Log: $LOG"
nohup .venv/bin/torchrun --nproc_per_node="${NPROC:-2}" --master_port="${MASTER_PORT:-29501}" \
  -m experiments.exp054_K_30.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp054_K30_extend600.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp054_K30_extend600.pid)"
