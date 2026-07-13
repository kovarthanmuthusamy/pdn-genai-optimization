#!/usr/bin/env python3
"""Verify sim → move → append path alignment for the combinations pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import setup_path

setup_path()

from pipelines.dataset_sim.paths import (
    RAW_ROOT_WIN,
    heatmap_map_path,
    heatmap_raw_dir,
    resolve_raw_root,
    verify_heatmap_ready,
)
from pipelines.dataset_sim.move_outputs import heatmaps_dir_name
from pipelines.dataset_sim.run_combinations_sim_pipeline import SWEEP_MHZ


def main() -> None:
    raw = resolve_raw_root()
    print("Pipeline path verification")
    print(f"  RAW_ROOT_WIN : {RAW_ROOT_WIN}")
    print(f"  WSL raw root : {raw}  exists={raw.is_dir()}")

    errors: list[str] = []
    for mhz in SWEEP_MHZ:
        mhz_f = float(mhz)
        sim_name = heatmaps_dir_name(mhz_f)
        sim_path = raw / sim_name
        append_path = heatmap_raw_dir(raw, mhz_f)

        ok = sim_path == append_path and sim_path.is_dir()
        line = f"  {int(round(mhz_f)):>3} MHz  sim={sim_name}  append={append_path.name}  match={sim_path == append_path}"
        if sim_path.is_dir():
            pi_n = sum(1 for c in sim_path.iterdir() if c.name.startswith("PI-"))
            line += f"  PI={pi_n:,}"
            map_ok = heatmap_map_path(sim_path / "PI-1", mhz_f).is_file()
            line += f"  map_ok={map_ok}"
            if not map_ok:
                errors.append(f"{mhz_f} MHz: missing expected .map under PI-1")
        else:
            line += "  (not generated yet)"
        print(line)

        if sim_path.is_dir() and sim_path != append_path:
            errors.append(f"{mhz_f} MHz: sim/append folder mismatch")

    # validate helper on latest completed MHz if any
    for mhz in reversed(SWEEP_MHZ):
        mhz_f = float(mhz)
        if heatmap_raw_dir(raw, mhz_f).is_dir():
            verify_heatmap_ready(mhz_f)
            print(f"\n  verify_heatmap_ready({int(round(mhz_f))} MHz): OK")
            break

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nAll checked paths align.")


if __name__ == "__main__":
    main()
