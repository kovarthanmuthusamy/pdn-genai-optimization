"""Compare Multifreq Dataset Layouts.

Run: python scrap/_compare_multifreq_datasets.py"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from repo_paths import REPO_ROOT as ROOT, setup_path
setup_path()


def main() -> None:
    names = ["data_multifreq", "data_multifreq_norm", "data_multifreq_gmax"]
    designs_by: dict[str, set[str]] = {}

    for name in names:
        p = ROOT / "datasets" / name
        mf = p / "manifest.csv"
        if not mf.is_file():
            print(f"{name}: no manifest")
            continue
        rows = list(csv.DictReader(mf.open()))
        designs = {r["design_id"] for r in rows}
        designs_by[name] = designs
        freqs = Counter(float(r["freq_mhz"]) for r in rows)
        layouts_dir = p / "layouts"
        n_ld = sum(1 for x in layouts_dir.iterdir() if x.is_dir()) if layouts_dir.is_dir() else 0
        hm_dir = p / "heatmap"
        hm = len(list(hm_dir.glob("sample_*.npy"))) if hm_dir.is_dir() else 0
        print("=" * 50)
        print(name)
        print(f"  manifest rows:     {len(rows):,}")
        print(f"  unique design_id:  {len(designs):,}")
        print(f"  layouts/ dirs:     {n_ld:,}")
        print(f"  heatmap files:     {hm:,}")
        print(f"  MHz anchors:       {sorted(freqs.keys())}")

    if "data_multifreq" in designs_by and "data_multifreq_norm" in designs_by:
        d_raw = designs_by["data_multifreq"]
        d_norm = designs_by["data_multifreq_norm"]
        print("=" * 50)
        print("DIFF data_multifreq vs data_multifreq_norm")
        print(f"  only in data_multifreq:      {len(d_raw - d_norm):,}")
        print(f"  only in data_multifreq_norm: {len(d_norm - d_raw):,}")
        print(f"  in both:                     {len(d_raw & d_norm):,}")
        if d_raw - d_norm:
            print(f"  example extra raw: {sorted(d_raw - d_norm)[:3]}")


if __name__ == "__main__":
    main()
