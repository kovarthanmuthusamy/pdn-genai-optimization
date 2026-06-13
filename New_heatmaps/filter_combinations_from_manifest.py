#!/usr/bin/env python3
"""Filter ``all_combinations.csv`` to layouts retained after inverse-K subsampling.

After ``datasets/subsample_multifreq_inverse_k.py --execute``, the multifreq dataset
keeps ~19.5k layouts (``manifest.csv`` + ``layouts/*/occ.npy``) but
``New_heatmaps/all_combinations.csv`` still has the original 49,106 decap rows.

This script:
  1. Collects unique ``decap_index`` values from ``datasets/data_multifreq_norm/manifest.csv``
  2. Verifies each row matches ``layouts/{design_id}/occ.npy`` (Occ_map successor)
  3. Backs up the full CSV and writes a filtered CSV (sorted by original decap_index)
  4. Writes ``decap_index_map.csv`` (peb_row → original_decap_index → design_id)
  5. Regenerates ``combined_all.peb`` (PI-Distribution @ 63 MHz default)

Examples::

    python New_heatmaps/filter_combinations_from_manifest.py --dry-run
    python New_heatmaps/filter_combinations_from_manifest.py --execute
    python New_heatmaps/filter_combinations_from_manifest.py --execute --freq-mhz 200
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src_vae.others.multifreq_layout_store import layouts_root, manifest_path  # noqa: E402
from scrap.generation.generate_peb import generate_peb  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _ROOT / "datasets" / "data_multifreq_norm"
DEFAULT_CSV = SCRIPT_DIR / "all_combinations.csv"
DEFAULT_PEB = SCRIPT_DIR / "combined_all.peb"
DEFAULT_FREQ_MHZ = 63


def load_decap_design_map(data_dir: Path) -> dict[int, str]:
    """original decap_index → design_id (one per retained layout)."""
    mapping: dict[int, str] = {}
    with manifest_path(data_dir).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            di = int(row["decap_index"])
            did = row["design_id"]
            if di in mapping and mapping[di] != did:
                raise ValueError(f"Duplicate decap_index {di}: {mapping[di]} vs {did}")
            mapping[di] = did
    return mapping


def validate_rows(
    df: pd.DataFrame,
    decap_to_design: dict[int, str],
    data_dir: Path,
) -> list[int]:
    """Return sorted decap indices; raise if any occ.npy disagrees with CSV row."""
    bad_occ = 0
    missing_layout = 0
    for di, did in decap_to_design.items():
        occ_path = layouts_root(data_dir) / did / "occ.npy"
        if not occ_path.is_file():
            missing_layout += 1
            continue
        occ = np.load(occ_path, mmap_mode="r").reshape(-1)
        row = df.iloc[di].to_numpy(dtype=np.float32)
        if not np.allclose(occ, row):
            bad_occ += 1

    if missing_layout:
        raise FileNotFoundError(
            f"{missing_layout} manifest design_id(s) missing layouts/*/occ.npy under {data_dir}",
        )
    if bad_occ:
        raise ValueError(
            f"{bad_occ} decap row(s) disagree with layouts/*/occ.npy — "
            "all_combinations.csv may be out of sync with the dataset",
        )

    if df.shape[1] != 52:
        raise ValueError(f"Expected 52 decap columns, got {df.shape[1]}")

    max_index = max(decap_to_design)
    if max_index >= len(df):
        raise IndexError(
            f"manifest decap_index max={max_index} but CSV has only {len(df)} rows",
        )

    return sorted(decap_to_design.keys())


def filter_combinations(
    *,
    data_dir: Path,
    csv_path: Path,
    execute: bool,
    peb_path: Path,
    freq_mhz: int,
    map_path: Path | None,
) -> dict[str, int | str]:
    decap_to_design = load_decap_design_map(data_dir)
    df = pd.read_csv(csv_path, header=None)
    n_before = len(df)

    kept_indices = validate_rows(df, decap_to_design, data_dir)
    filtered = df.iloc[kept_indices].reset_index(drop=True)

    summary: dict[str, int | str] = {
        "csv_rows_before": n_before,
        "csv_rows_after": len(filtered),
        "layouts": len(kept_indices),
        "data_dir": str(data_dir),
        "csv_path": str(csv_path),
    }

    if not execute:
        print(f"[DRY-RUN] {n_before:,} → {len(filtered):,} rows ({len(kept_indices):,} layouts)")
        return summary

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = csv_path.with_name(f"{csv_path.stem}.csv.bak_{n_before}_{ts}")
    shutil.copy2(csv_path, backup)
    print(f"  backup → {backup.name}")

    filtered.to_csv(csv_path, header=False, index=False)
    print(f"  wrote {csv_path} ({len(filtered):,} rows)")

    map_out = map_path or (SCRIPT_DIR / "decap_index_map.csv")
    with map_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["peb_row", "original_decap_index", "design_id"],
        )
        w.writeheader()
        for peb_row, di in enumerate(kept_indices):
            w.writerow(
                {
                    "peb_row": peb_row,
                    "original_decap_index": di,
                    "design_id": decap_to_design[di],
                },
            )
    print(f"  wrote {map_out} ({len(kept_indices):,} rows)")

    freq_hz = f"{int(freq_mhz)}e6"
    generate_peb(
        occupancy=filtered.to_numpy(dtype=np.float32),
        output_path=str(peb_path),
        powerbus="Power_GND",
        freq=freq_hz,
        components="IC1_Port1,IC2_Port2",
        include_distribution=True,
        include_spectrum=False,
    )
    summary["peb_path"] = str(peb_path)
    summary["peb_freq"] = freq_hz
    summary["map_path"] = str(map_out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter all_combinations.csv to manifest / Occ_map layouts.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Multifreq dataset root (manifest + layouts/*/occ.npy)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to all_combinations.csv",
    )
    ap.add_argument(
        "--peb",
        type=Path,
        default=DEFAULT_PEB,
        help="Output combined PI-Distribution .peb",
    )
    ap.add_argument(
        "--map",
        type=Path,
        default=None,
        help="decap_index_map.csv path (default: beside csv)",
    )
    ap.add_argument(
        "--freq-mhz",
        type=int,
        default=DEFAULT_FREQ_MHZ,
        help="PI-Distribution frequency for regenerated combined_all.peb",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Write filtered CSV, map, and PEB (default: dry-run)",
    )
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    if not manifest_path(data_dir).is_file():
        raise SystemExit(f"Missing manifest: {manifest_path(data_dir)}")

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path}")

    print(f"Dataset : {data_dir}")
    print(f"CSV     : {csv_path}")
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode    : {mode}")

    summary = filter_combinations(
        data_dir=data_dir,
        csv_path=csv_path,
        execute=args.execute,
        peb_path=args.peb.resolve(),
        freq_mhz=args.freq_mhz,
        map_path=args.map.resolve() if args.map else None,
    )

    print(f"\nLayouts kept: {summary['layouts']:,}")
    print(f"CSV rows    : {summary['csv_rows_before']:,} → {summary['csv_rows_after']:,}")
    if not args.execute:
        print("\nDry-run only. Re-run with --execute to apply.")
    else:
        print(f"\nPEB         : {summary.get('peb_path')} @ {summary.get('peb_freq')}")
        print(f"Index map   : {summary.get('map_path')}")
        print("\nFor other anchor MHz, run: python New_heatmaps/change_frequency.py")


if __name__ == "__main__":
    main()
