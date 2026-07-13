#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 TORCH_CPU_THREADS=4
NPROC="${NPROC:-2}"
ROOT_LOG_DIR=logs
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp055_K30_ddp_${STAMP}.log"
echo "exp055 DDP training on GPUs ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"
echo "Log: $LOG"
nohup .venv/bin/torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29501}" \
  -m experiments.exp055_hard_occ.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp055_K30_ddp_train.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp055_K30_ddp_train.pid)"
