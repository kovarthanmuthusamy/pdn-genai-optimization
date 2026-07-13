"""Latent run export PEB — step 1: scrap layout export and PEB generation.

Run: python pipelines/latent/export_peb.py"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import os
import sys
from pathlib import Path


from pipelines.latent.optimization_loader import load_optimization_module, resolve_run_dir
from pipelines.latent.scrap_pipeline import export_all, generate_peb_for_run, load_pi_freq_mhz

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/latent/export_peb.py
# =============================================================================

RUN_DIR: str | Path | None = None

# =============================================================================


def main() -> None:
    lo = load_optimization_module()
    repo_root = lo._project_root()
    os.chdir(repo_root)
    run_dir = resolve_run_dir(RUN_DIR, repo_root)

    if not (run_dir / "run_config.json").exists():
        raise SystemExit(f"Missing run_config.json: {run_dir}")

    pi_mhz = load_pi_freq_mhz(run_dir)
    print(f"Run: {run_dir}\nPI frequency: {pi_mhz} MHz\n")

    print("Exporting to scrap layout …")
    k_values = export_all(run_dir)
    peb_path = generate_peb_for_run(run_dir, k_values, pi_freq_mhz=pi_mhz)
    print(f"\nDone. PEB → {peb_path}")
    print("Next: python pipelines/latent/compare_report.py")


if __name__ == "__main__":
    main()
