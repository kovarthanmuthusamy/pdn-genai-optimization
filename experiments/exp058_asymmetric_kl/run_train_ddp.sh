#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
export VAE_EXPERIMENT_DIR="$ROOT/experiments/exp058_asymmetric_kl"
ROOT_LOG_DIR="$VAE_EXPERIMENT_DIR/logs"
mkdir -p "$ROOT_LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_LOG_DIR/exp058_fresh_ddp_${STAMP}.log"
echo "exp058 fresh DDP training"
echo "Log: $LOG"
nohup torchrun --standalone --nproc_per_node="${NPROC:-2}" \
  -m experiments.exp058_asymmetric_kl.codes.train_vae_simple >"$LOG" 2>&1 &
echo $! >"$ROOT_LOG_DIR/exp058_train_ddp.pid"
echo "PID: $(cat $ROOT_LOG_DIR/exp058_train_ddp.pid)"
