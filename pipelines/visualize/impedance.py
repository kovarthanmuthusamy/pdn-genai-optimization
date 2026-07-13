"""Impedance profile log-log comparison plotter.

Run: python pipelines/visualize/impedance.py"""
from __future__ import annotations

import sys
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from repo_paths import repo_path, setup_path

setup_path()

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/visualize/impedance.py
# =============================================================================

FREQ_PATH = repo_path("configs", "Frequency_data_hz.npy")
TARGET_IMP_PATH = repo_path("configs", "target_impedance.npy")
OUTPUT_PATH = repo_path("temp_visuals", "generated_vs_target_impedance_npy.png")

# =============================================================================


def plot_impedance_profile(
    output_path: str | Path,
    *,
    generated_impedance: np.ndarray | None = None,
    freq_path: Path = FREQ_PATH,
    target_path: Path = TARGET_IMP_PATH,
) -> None:
    frequency = np.load(freq_path)
    target = np.load(target_path)

    plt.figure(figsize=(15, 9))
    plt.loglog(frequency, target, linestyle="--", linewidth=3.5, label="Target Impedance (TI)", color="red")
    if generated_impedance is not None:
        plt.loglog(frequency, generated_impedance, linestyle="-", linewidth=3.5, label="Generated Impedance", color="blue")
    plt.ylim(1e-3, 1e2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Impedance (Ohm)")
    plt.title("Impedance Profile Comparison")
    plt.legend(fontsize=22)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{output_path}'")


# Backward-compatible alias
Impedance_profile = plot_impedance_profile


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_impedance_profile(OUTPUT_PATH)
