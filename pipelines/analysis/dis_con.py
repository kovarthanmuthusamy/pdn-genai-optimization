"""Sigmoid-smooth 3-channel occupancy heatmap batch processor.

Run: python pipelines/analysis/dis_con.py"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import sys
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/analysis/dis_con.py
# =============================================================================

INPUT_DIR = Path("src/data_2/Occ_map")
OUTPUT_DIR = Path("src/data_norm/Occ_map")
SIGMOID_ALPHA = 5.0

# =============================================================================


def process_channels(path: Path, out: Path, *, alpha: float) -> None:
    arr = torch.tensor(np.load(path), dtype=torch.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.permute(2, 0, 1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {arr.shape} in {path}")

    imp, occ, mask = arr[0], arr[1], arr[2]
    occ_s = torch.sigmoid(alpha * (occ - 0.5))
    mask_s = torch.sigmoid(alpha * (mask - 0.5))
    np.save(out, torch.stack([imp, occ_s, mask_s]).numpy())
    print(f"Processed: {path.name} → {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.npy")):
        process_channels(path, OUTPUT_DIR / path.name, alpha=SIGMOID_ALPHA)
    print("✓ Processing complete!")


if __name__ == "__main__":
    main()
