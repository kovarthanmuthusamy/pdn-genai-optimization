"""Occupancy comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and render occupancy checkbox plots per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real_occupancy``.
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first error
"""
from __future__ import annotations

from scrap.comparison import compare_generated_vs_real_occupancy as occ
from scrap.comparison.batch_over_k import run_over_k

# =============================================================================
# CONFIGURATION — edit these before running: python scrap/comparison/compare_generated_vs_real_occupancy_all_k.py
# =============================================================================

K_MIN, K_MAX = 1, 52
GENERATED_BASE_DIR = "scrap/generated_samples_v2"
FAIL_FAST = False

# =============================================================================


def main() -> None:
    run_over_k(
        occ,
        k_min=K_MIN,
        k_max=K_MAX,
        base_dir=GENERATED_BASE_DIR,
        fail_fast=FAIL_FAST,
        skip_markers=("folder not found", "No data_sample"),
    )


if __name__ == "__main__":
    main()
