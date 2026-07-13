"""Binary PCB mask shape inspector.

Run: python pipelines/analysis/check_mask.py"""
from __future__ import annotations

import sys
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from repo_paths import repo_path, setup_path

setup_path()

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/analysis/check_mask.py
# =============================================================================

MASK_FILE = repo_path("configs", "binary_mask.npy")
OUTPUT_PATH = Path("mask_debug.png")

# =============================================================================


def main() -> None:
    mask = np.load(MASK_FILE).astype(np.float32)
    print(f"Shape: {mask.shape}")
    print(f"Min: {mask.min()}, Max: {mask.max()}")

    fig, ax = plt.subplots()
    im = ax.imshow(mask, cmap="gray", origin="lower")
    ax.set_title("Mask")
    fig.colorbar(im, ax=ax)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
