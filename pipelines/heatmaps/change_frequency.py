"""Change PI-Distribution frequency in a single .peb file.

Purpose:
    Replace ``EditPIDistribution Frequency="..."`` in a source PEB and write one output file.

Run:
    python pipelines/heatmaps/change_frequency.py

Agent notes:
    - What: Single-frequency PEB rewrite (one MHz). For all anchors use ``regenerate_mhz_pebs.py``.
    - Usage: Set ``INPUT_PEB``, ``SET_FREQ_MHZ``, ``OUTPUT_PEB`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source PEB path
        - ``SET_FREQ_MHZ`` — target inspection frequency (MHz)
        - ``OUTPUT_PEB`` — destination path
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

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/heatmaps/change_frequency.py
# =============================================================================

SCRIPT_DIR = repo_path("data", "heatmaps")
# Master PEB with all layouts (19k: combined_all.peb; 29k: build from all_combinations_merged.csv first)
INPUT_PEB = SCRIPT_DIR / "combined_merged_63MHz.peb"
SET_FREQ_MHZ = 0
OUTPUT_PEB = SCRIPT_DIR / f"peb_with_29k/combined_all_{SET_FREQ_MHZ}MHz.peb"

# =============================================================================


def main() -> None:
    count = write_peb_at_mhz(INPUT_PEB, OUTPUT_PEB, SET_FREQ_MHZ)
    print(f"Replaced {count} frequency tags → {SET_FREQ_MHZ}e6")
    print(f"Saved: {OUTPUT_PEB}")


if __name__ == "__main__":
    main()
