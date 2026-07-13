#!/usr/bin/env python3
"""Generate ECADStar .peb file(s) from a decap combinations CSV (52-column 0/1 rows).

Each CSV row → one PI job in PEB order (row 0 = PI-1, row N-1 = PI-N).

Run:
    python pipelines/heatmaps/generate_peb_from_csv.py

Typical use after merge:
    INPUT_CSV = data/heatmaps/all_combinations_merged.csv
    → combined_merged_63MHz.peb  (PI-Distribution @ 63 MHz)
    → combined_merged_impedance.peb  (CreatePISpectrum only)

For per-anchor MHz files from an existing master PEB, use ``regenerate_mhz_pebs.py``
or ``change_frequency.py`` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, repo_path, setup_path

setup_path()

from pipelines.dataset_sim.combinations import load_combinations_csv
from pipelines.dataset_sim.peb import build_distribution_peb, build_impedance_peb

# =============================================================================
# CONFIGURATION — edit before running
# =============================================================================

INPUT_CSV = repo_path("data", "heatmaps", "all_combinations_merged.csv")
OUTPUT_DIR = repo_path("data", "heatmaps")

# PI-Distribution (heatmap) frequency for distribution PEB
DIST_MHZ = 63.0
WRITE_DISTRIBUTION = True
WRITE_IMPEDANCE = True

POWERBUS = "Power_GND"
IMPEDANCE_COMPONENTS = "IC1_Port1,IC2_Port2"

# Output basenames (under OUTPUT_DIR)
DIST_PEB_NAME = "combined_merged_{mhz}MHz.peb"
IMPEDANCE_PEB_NAME = "combined_merged_impedance.peb"

# Optional row slice (0-based). None = all rows.
START_ROW = 0
MAX_ROWS: int | None = None

# =============================================================================


def _slice_occupancy(occ, start: int, max_rows: int | None):
    end = occ.shape[0] if max_rows is None else min(occ.shape[0], start + max_rows)
    if start >= end:
        raise SystemExit(f"START_ROW={start} >= end={end}")
    return occ[start:end], start, end


def main() -> None:
    csv_path = Path(INPUT_CSV).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path}")

    occ_all = load_combinations_csv(csv_path)
    occ, start, end = _slice_occupancy(occ_all, START_ROW, MAX_ROWS)
    n = int(occ.shape[0])

    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV       : {csv_path}")
    print(f"Rows      : {start}..{end - 1} ({n:,} layouts)")
    print(f"PI range  : PI-{start + 1}..PI-{end}")
    print(f"Output dir: {out_dir}")

    if WRITE_DISTRIBUTION:
        mhz = int(round(float(DIST_MHZ)))
        dist_path = out_dir / DIST_PEB_NAME.format(mhz=mhz)
        build_distribution_peb(
            occ,
            dist_path,
            repo_root=REPO_ROOT,
            mhz=float(DIST_MHZ),
            powerbus=POWERBUS,
        )
        print(f"  Distribution @ {mhz} MHz : {dist_path}  ({n:,} jobs)")

    if WRITE_IMPEDANCE:
        imp_path = out_dir / IMPEDANCE_PEB_NAME
        build_impedance_peb(
            occ,
            imp_path,
            repo_root=REPO_ROOT,
            powerbus=POWERBUS,
            components=IMPEDANCE_COMPONENTS,
        )
        print(f"  Impedance              : {imp_path}  ({n:,} jobs)")

    if not WRITE_DISTRIBUTION and not WRITE_IMPEDANCE:
        raise SystemExit("Set WRITE_DISTRIBUTION and/or WRITE_IMPEDANCE to True")


if __name__ == "__main__":
    main()
