"""Step 1: Export latent-opt solutions to scrap layout and build .peb file.

Reads Latent_opm/runs/<experiment>/<n>/K##/best_*.npy and writes:
  <run_dir>/exported_samples/K##/   (compatible with scrap/comparison/compare.py)
  <run_dir>/latent_run.peb          (PI-Spectrum only — no PI-Distribution)

Run:
  python Latent_opm/latent_run_export_peb.py
  python Latent_opm/latent_run_export_peb.py Latent_opm/runs/exp038_true_multi/0
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Latent_opm.latent_scrap_pipeline import (  # noqa: E402
    export_all,
    generate_peb_for_run,
    load_pi_freq_mhz,
)

RUN_DIR: str | None = None


def _load_lo():
    path = Path(__file__).with_name("latent_optimization_impedance.py")
    spec = importlib.util.spec_from_file_location("latent_optimization_impedance", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    lo = _load_lo()
    repo_root = lo._project_root()
    os.chdir(repo_root)

    run_dir = Path(RUN_DIR) if RUN_DIR else None
    if run_dir is None:
        run_dir = lo.resolve_run_dir(repo_root, None)
    else:
        run_dir = (repo_root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()

    if not (run_dir / "run_config.json").exists():
        raise SystemExit(f"Missing run_config.json: {run_dir}")

    pi_mhz = load_pi_freq_mhz(run_dir)
    print(f"Run: {run_dir}")
    print(f"PI frequency (PEB): {pi_mhz} MHz\n")

    print("Exporting to scrap layout …")
    k_values = export_all(run_dir)

    print("\nGenerating PEB …")
    peb_path = generate_peb_for_run(run_dir, k_values, pi_freq_mhz=pi_mhz)
    print(f"\nDone. PEB → {peb_path}")
    print("Next: python Latent_opm/latent_run_compare_report.py", run_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RUN_DIR = sys.argv[1]
    main()
