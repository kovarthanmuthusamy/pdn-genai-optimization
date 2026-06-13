"""Deprecated — use the two-step latent run scripts instead.

  1. python Latent_opm/latent_run_export_peb.py [run_dir]
     Export best latent solutions → exported_samples/ + latent_run.peb

  2. python Latent_opm/latent_run_compare_report.py [run_dir]
     Impedance gen-vs-real plots (scrap/compare.py style, no heatmap) + run_report.md

Requires Real/ under exported_samples/K{n}/ for compare (see scrap move_pi_to_real).
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    main()
