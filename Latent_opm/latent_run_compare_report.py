"""Step 2: Move PI results → Real/, impedance compare plots, run_report.md.

Requires step 1 (latent_run_export_peb.py).

Run:
  python Latent_opm/latent_run_compare_report.py
  python Latent_opm/latent_run_compare_report.py Latent_opm/runs/exp038_true_multi/0
  python Latent_opm/latent_run_compare_report.py Latent_opm/runs/exp038_true_multi/4 --report-only
  python Latent_opm/latent_run_compare_report.py Latent_opm/runs/exp038_true_multi/4 --no-move

Options:
  --report-only   Rebuild run_report.md from existing plot PNGs only (no move_pi, no compare).
  --no-move       Run compare + report using existing Real/ data (skip move_pi).

Environment:
  LATENT_SOURCE_EMC_DIR  — path to .emc folder after running latent_run.peb (default: scrap config)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Latent_opm.latent_scrap_pipeline import (  # noqa: E402
    collect_existing_compare_pngs,
    export_root,
    load_pi_freq_mhz,
    move_pi_from_emc,
    run_impedance_compare_all,
    write_compare_report,
)

# ECADStar output folder (after batch PI run on latent_run.peb)
SOURCE_EMC_DIR = os.getenv(
    "LATENT_SOURCE_EMC_DIR",
    r"C:\Users\muthusamy\Desktop\design\H-shape.emc",
)


def _load_lo():
    path = Path(__file__).with_name("latent_optimization_impedance.py")
    spec = importlib.util.spec_from_file_location("latent_optimization_impedance", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _k_list_from_export(run_dir: Path) -> list[int]:
    root = export_root(run_dir)
    if not root.is_dir():
        raise SystemExit(f"Missing {root}. Run latent_run_export_peb.py first.")
    ks = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("K") and p.name[1:].isdigit():
            ks.append(int(p.name[1:]))
    return sorted(ks)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move PI outputs, compare impedance, write run_report.md")
    p.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run folder (default: latest under Latent_opm/runs)",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild report from existing plot PNGs only; skip move_pi and compare",
    )
    p.add_argument(
        "--no-move",
        action="store_true",
        help="Skip move_pi; run compare + report (Real/ must already exist)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.report_only and args.no_move:
        raise SystemExit("Use only one of --report-only or --no-move")

    lo = _load_lo()
    repo_root = lo._project_root()
    os.chdir(repo_root)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir = (repo_root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    else:
        run_dir = lo.resolve_run_dir(repo_root, None)

    pi_mhz = load_pi_freq_mhz(run_dir)
    k_values = _k_list_from_export(run_dir)

    print(f"Run: {run_dir}")
    print(f"K values: {k_values}")
    if args.report_only:
        print("Mode: report-only (skip move_pi and compare)\n")
    elif args.no_move:
        print("Mode: no-move (compare + report only)\n")
    else:
        print("Mode: full (move_pi + compare + report)\n")

    peb_path = run_dir / "latent_run.peb"
    if not peb_path.exists():
        peb_path = None

    if args.report_only:
        compare_pngs, heatmap_pngs = collect_existing_compare_pngs(run_dir, k_values)
        if not compare_pngs:
            raise SystemExit(
                f"No comparison PNGs under {export_root(run_dir)}. "
                "Run without --report-only after ECADStar + compare, or use --no-move."
            )
        print(f"  found {len(compare_pngs)} impedance plot(s) for report")
    else:
        if not args.no_move:
            print("--- move_pi_to_real (PI-* → exported_samples/K{n}/Real/) ---")
            try:
                move_pi_from_emc(
                    run_dir, k_values, source_emc_dir=SOURCE_EMC_DIR, peb_path=peb_path,
                )
            except SystemExit as e:
                print(f"\nmove_pi_to_real: {e}")
                print("Continuing with compare/report (use --report-only to skip move next time).\n")
        else:
            print("--- move_pi_to_real: skipped (--no-move) ---\n")

        print("--- impedance compare (Target + Real + Generated, scrap/compare.py) ---")
        compare_pngs, heatmap_pngs = run_impedance_compare_all(
            run_dir, repo_root, k_values, pi_freq_mhz=pi_mhz,
        )

    report_path = write_compare_report(
        run_dir,
        peb_path=peb_path,
        compare_pngs=compare_pngs,
        heatmap_pngs=heatmap_pngs or None,
    )
    print(f"\nReport → {report_path}")
    html_path = run_dir / "run_report.html"
    if html_path.is_file():
        print(f"Open in browser for images: {html_path}")


if __name__ == "__main__":
    main()
