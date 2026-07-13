"""Latent run compare report — step 2: PI move, impedance plots, and report.

Run: python pipelines/latent/compare_report.py"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import os
import sys
from pathlib import Path


from pipelines.latent.optimization_loader import load_optimization_module, resolve_run_dir
from pipelines.latent.scrap_pipeline import (
    collect_existing_compare_pngs,
    export_root,
    load_pi_freq_mhz,
    move_pi_from_emc,
    run_impedance_compare_all,
    write_compare_report,
)

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/latent/compare_report.py
# =============================================================================

RUN_DIR: str | Path | None = None
SOURCE_EMC_DIR = os.getenv("LATENT_SOURCE_EMC_DIR", r"C:\Users\muthusamy\Desktop\design\H-shape.emc")
REPORT_ONLY = False
NO_MOVE = False

# =============================================================================


def _k_list(run_dir: Path) -> list[int]:
    root = export_root(run_dir)
    if not root.is_dir():
        raise SystemExit(f"Missing {root}. Run latent_run_export_peb.py first.")
    return sorted(int(p.name[1:]) for p in root.iterdir() if p.is_dir() and p.name.startswith("K") and p.name[1:].isdigit())


def main() -> None:
    if REPORT_ONLY and NO_MOVE:
        raise SystemExit("Use only one of REPORT_ONLY or NO_MOVE")

    lo = load_optimization_module()
    repo_root = lo._project_root()
    os.chdir(repo_root)
    run_dir = resolve_run_dir(RUN_DIR, repo_root)
    k_values = _k_list(run_dir)
    pi_mhz = load_pi_freq_mhz(run_dir)
    peb_path = run_dir / "latent_run.peb"
    if not peb_path.exists():
        peb_path = None

    mode = "report-only" if REPORT_ONLY else ("no-move" if NO_MOVE else "full")
    print(f"Run: {run_dir}\nK: {k_values}\nMode: {mode}\n")

    if REPORT_ONLY:
        compare_pngs, heatmap_pngs = collect_existing_compare_pngs(run_dir, k_values)
        if not compare_pngs:
            raise SystemExit(f"No comparison PNGs under {export_root(run_dir)}")
        print(f"  found {len(compare_pngs)} impedance plot(s)")
    else:
        if not NO_MOVE:
            print("--- move_pi ---")
            try:
                move_pi_from_emc(run_dir, k_values, source_emc_dir=SOURCE_EMC_DIR, peb_path=peb_path)
            except SystemExit as exc:
                print(f"move_pi: {exc}\nContinuing with compare/report.\n")
        print("--- impedance compare ---")
        compare_pngs, heatmap_pngs = run_impedance_compare_all(run_dir, repo_root, k_values, pi_freq_mhz=pi_mhz)

    report_path = write_compare_report(run_dir, peb_path=peb_path, compare_pngs=compare_pngs, heatmap_pngs=heatmap_pngs or None)
    print(f"\nReport → {report_path}")
    html = run_dir / "run_report.html"
    if html.is_file():
        print(f"Open: {html}")


if __name__ == "__main__":
    main()
