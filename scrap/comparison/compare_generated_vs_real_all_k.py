"""Generated vs real comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and run heatmap+impedance comparison per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real`` for K=1..52 (or a subrange).
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first K that errors
"""
from __future__ import annotations

from scrap.comparison import compare_generated_vs_real as compare
from scrap.comparison.batch_over_k import run_over_k

# =============================================================================
# CONFIGURATION — edit these before running: python scrap/comparison/compare_generated_vs_real_all_k.py
# =============================================================================

K_MIN, K_MAX = 1, 52
GENERATED_BASE_DIR = "scrap/generated_samples_v2"
FAIL_FAST = False

# =============================================================================


def main() -> None:
    run_over_k(
        compare,
        k_min=K_MIN,
        k_max=K_MAX,
        base_dir=GENERATED_BASE_DIR,
        fail_fast=FAIL_FAST,
        skip_markers=("folder not found", "No real"),
    )


if __name__ == "__main__":
    main()
