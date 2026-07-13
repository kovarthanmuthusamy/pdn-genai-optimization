"""Batch-regenerate anchor-frequency PEB files from combined_all.peb.

Purpose:
    Write ``combined_all_{MHz}MHz.peb`` for each MHz in ``ANCHORS_MHZ`` by rewriting PI frequency tags.

Run:
    python pipelines/heatmaps/regenerate_mhz_pebs.py

Agent notes:
    - What: Splits one master PEB into per-anchor-frequency PEB files for ECADStar batch runs.
    - Usage: Point ``INPUT_PEB`` at ``combined_all.peb`` → set ``ANCHORS_MHZ`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source combined PEB under ``data/heatmaps/``
        - ``ANCHORS_MHZ`` — list of MHz values to emit
    - Key symbol: ``write_peb_at_mhz`` (from ``libs.peb.frequency``)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import repo_path, setup_path

setup_path()
from libs.peb.frequency import write_peb_at_mhz

SCRIPT_DIR = repo_path("data", "heatmaps")

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/heatmaps/regenerate_mhz_pebs.py
# =============================================================================

INPUT_PEB = SCRIPT_DIR / "combined_all.peb"
ANCHORS_MHZ = [10, 80, 130, 150, 200, 230, 250, 270, 300, 330, 400, 450, 500, 550, 600]

# =============================================================================


def main() -> None:
    for mhz in ANCHORS_MHZ:
        out = SCRIPT_DIR / f"combined_all_{mhz}MHz.peb"
        count = write_peb_at_mhz(INPUT_PEB, out, mhz)
        print(f"{out.name}: {count} replacements")


if __name__ == "__main__":
    main()
