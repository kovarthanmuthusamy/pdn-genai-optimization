"""Run a single-K comparison module for K_MIN..K_MAX."""
from __future__ import annotations

import traceback
from pathlib import Path
from types import ModuleType
from typing import Iterable


def run_over_k(
    module: ModuleType,
    *,
    k_min: int,
    k_max: int,
    base_dir: str | Path,
    fail_fast: bool = False,
    skip_markers: Iterable[str] = ("folder not found", "No real", "No data_sample"),
) -> None:
    """Patch ``module.K_VALUE`` / ``module.BASE_GENERATED_DIR`` and call ``module.main()`` per K."""
    ok: list[int] = []
    failed: list[tuple[int, str]] = []
    skipped: list[int] = []
    base = Path(base_dir)

    for k in range(k_min, k_max + 1):
        module.K_VALUE = k
        module.BASE_GENERATED_DIR = base
        try:
            module.main()
            ok.append(k)
        except SystemExit as exc:
            msg = str(exc)
            failed.append((k, msg))
            if any(m in msg for m in skip_markers):
                skipped.append(k)
            if fail_fast:
                raise
        except Exception as exc:
            failed.append((k, repr(exc)))
            traceback.print_exc()
            if fail_fast:
                raise

    print("\n=== Summary ===")
    print(f"OK:      {len(ok)}")
    print(f"Failed:  {len(failed)}")
    print(f"Skipped: {len(set(skipped))}")
    if failed:
        print("\nFailures:")
        for k, msg in failed:
            print(f"  K{k}: {msg}")
