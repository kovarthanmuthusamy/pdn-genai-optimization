"""Run occupancy checkbox plotting for all K values.

This is a thin wrapper around `scrap/compare_generated_vs_real_occupancy.py`.

It will attempt to run the occupancy plot for each K in [K_MIN..K_MAX].
Plots are written into each `scrap/generated_samples/K{K}/` folder.

Run
    python scrap/compare_generated_vs_real_occupancy_all_k.py
"""

from __future__ import annotations

import traceback
from pathlib import Path

import compare_generated_vs_real_occupancy as occ


# ============================================================
# CONFIGURATION
# ============================================================
K_MIN = 1
K_MAX = 52

# Base directory that contains K1/, K2/, ... sub-folders with generated samples.
# Overrides compare_generated_vs_real_occupancy.BASE_GENERATED_DIR for every K run.
GENERATED_BASE_DIR = "scrap/generated_samples_v2"

# If True, stop at first failure. If False, continue and report failures.
FAIL_FAST = False


def main() -> None:
    ok: list[int] = []
    failed: list[tuple[int, str]] = []
    skipped: list[int] = []

    for k in range(K_MIN, K_MAX + 1):
        occ.K_VALUE = k
        occ.BASE_GENERATED_DIR = Path(GENERATED_BASE_DIR)

        try:
            occ.main()
            ok.append(k)
        except SystemExit as e:
            msg = str(e)
            failed.append((k, msg))
            if "folder not found" in msg or "No data_sample" in msg:
                skipped.append(k)
            if FAIL_FAST:
                raise
        except Exception as e:
            failed.append((k, repr(e)))
            traceback.print_exc()
            if FAIL_FAST:
                raise

    print("\n=== Summary ===")
    print(f"OK:      {len(ok)}")
    print(f"Failed:  {len(failed)}")
    print(f"Skipped: {len(set(skipped))}")
    if failed:
        print("\nFailures:")
        for k, msg in failed:
            print(f"  K{k}: {msg}")


if __name__ == "__main__":
    main()
