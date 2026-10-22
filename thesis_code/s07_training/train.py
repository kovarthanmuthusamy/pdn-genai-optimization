"""Train the VAE (phase 1). Run as a module from the repo root.

What: trains the multi-modal VAE with the shipped config.yaml (700 epochs, 112 placements per GPU,
    2 GPUs in the thesis run). Checkpoints, metric files and the run log go to
    OUTPUT_DIR/s07_training/ (TrainConfig.experiment_dir). A fresh run (resume_checkpoint null)
    first deletes that folder's checkpoints/, logs/ and metrics/.
Usage:
    CUDA_VISIBLE_DEVICES=0 python -m thesis_code.s07_training.train
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m thesis_code.s07_training.train
Config keys: EXTRA_CONFIG (a second config file applied on top), MAX_BATCHES (a short check run).
"""
from __future__ import annotations

from pathlib import Path

from thesis_code.s07_training.loop import train_vae

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
EXTRA_CONFIG: Path | None = None     # e.g. a file with {"num_epochs": 1} for a check run
MAX_BATCHES: int | None = None       # None = full epochs; a number stops every pass after that many batches
# =============================================================================


def main() -> None:
    train_vae(extra_config=EXTRA_CONFIG, max_batches=MAX_BATCHES)


if __name__ == "__main__":
    main()
