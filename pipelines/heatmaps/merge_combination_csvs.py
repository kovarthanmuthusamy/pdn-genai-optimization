#!/usr/bin/env python3
"""Merge old + new decap layout CSVs for unified PEB / multifreq simulation.

Order (required for PI-* alignment):
  1. ``all_combinations.csv``  — 19,499 filtered legacy layouts (peb_row 0..N_old-1)
  2. ``combinations.csv``      — 10,000 new layouts (local row 0..N_new-1)

Merged row ``i`` → ECADStar ``PI-(i+1)``.

Legacy rows use ``decap_index_map.csv`` for ``design_id`` / ``decap_index``.
New rows use ``combinations_pi{pi}_d{local_row}`` with ``pi = local_row + 1``.

Run:
    python pipelines/heatmaps/merge_combination_csvs.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, setup_path

setup_path()

N_DECAPS = 52

# =============================================================================
# CONFIGURATION
# =============================================================================

OLD_CSV = REPO_ROOT / "data" / "heatmaps" / "all_combinations.csv"
NEW_CSV = REPO_ROOT / "data" / "heatmaps" / "combinations.csv"
DECAP_INDEX_MAP = REPO_ROOT / "data" / "heatmaps" / "decap_index_map.csv"

OUTPUT_CSV = REPO_ROOT / "data" / "heatmaps" / "all_combinations_merged.csv"
OUTPUT_MAP = REPO_ROOT / "data" / "heatmaps" / "merged_combinations_index_map.csv"
OUTPUT_REPORT = REPO_ROOT / "data" / "heatmaps" / "merged_combinations_report.json"

EXECUTE = True  # False = dry-run validation only

# =============================================================================


def load_rows(path: Path) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != N_DECAPS:
                raise ValueError(f"{path}:{line_no}: expected {N_DECAPS} columns, got {len(parts)}")
            rows.append(tuple(int(x) for x in parts))
    return rows


def load_legacy_map(path: Path) -> list[dict[str, str | int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy map: {path}")
    rows: list[dict[str, str | int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "peb_row": int(row["peb_row"]),
                    "original_decap_index": int(row["original_decap_index"]),
                    "design_id": str(row["design_id"]),
                }
            )
    rows.sort(key=lambda r: int(r["peb_row"]))
    return rows


def merge_combination_csvs(
    *,
    old_csv: Path,
    new_csv: Path,
    legacy_map_csv: Path,
    output_csv: Path,
    output_map: Path,
    execute: bool,
) -> dict:
    old_rows = load_rows(old_csv)
    new_rows = load_rows(new_csv)
    legacy_map = load_legacy_map(legacy_map_csv)

    if len(old_rows) != len(legacy_map):
        raise ValueError(
            f"Row count mismatch: {old_csv.name} has {len(old_rows):,} rows but "
            f"{legacy_map_csv.name} has {len(legacy_map):,} entries"
        )

    old_set = set(old_rows)
    new_set = set(new_rows)
    overlap = old_set & new_set
    if overlap:
        raise ValueError(f"{len(overlap)} layout(s) appear in both old and new CSV")

    for peb_row, entry in enumerate(legacy_map):
        if int(entry["peb_row"]) != peb_row:
            raise ValueError(f"Legacy map peb_row not contiguous at {peb_row}")

    merged: list[tuple[int, ...]] = old_rows + new_rows
    index_rows: list[dict[str, str | int]] = []

    for peb_row, entry in enumerate(legacy_map):
        index_rows.append(
            {
                "global_peb_row": peb_row,
                "pi_number": peb_row + 1,
                "segment": "legacy",
                "local_row": peb_row,
                "decap_index": int(entry["original_decap_index"]),
                "design_id": str(entry["design_id"]),
            }
        )

    n_old = len(old_rows)
    for local_row, _ in enumerate(new_rows):
        global_row = n_old + local_row
        pi = local_row + 1
        index_rows.append(
            {
                "global_peb_row": global_row,
                "pi_number": global_row + 1,
                "segment": "combinations",
                "local_row": local_row,
                "decap_index": local_row,
                "design_id": f"combinations_pi{pi}_d{local_row}",
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
        "legacy_map_csv": str(legacy_map_csv),
        "n_old": n_old,
        "n_new": len(new_rows),
        "n_merged": len(merged),
        "overlap_rows": 0,
        "output_csv": str(output_csv),
        "output_map": str(output_map),
        "pi_range_legacy": f"PI-1..PI-{n_old}",
        "pi_range_combinations": f"PI-{n_old + 1}..PI-{len(merged)}",
        "append_scripts": {
            "legacy": "pipelines/data/append_peb_batch_raw.py (PI 1..N_old, decap_index_map)",
            "combinations": "pipelines/data/append_combinations_multifreq.py (PI 1..N_new in combinations Raw)",
        },
    }

    if not execute:
        print(f"[DRY-RUN] Would write {len(merged):,} rows → {output_csv.name}")
        print(f"  legacy PI-1..PI-{n_old:,}  +  combinations PI-{n_old + 1}..PI-{len(merged):,}")
        return report

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in merged:
            w.writerow(list(row))

    with output_map.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "global_peb_row",
            "pi_number",
            "segment",
            "local_row",
            "decap_index",
            "design_id",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(index_rows)

    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_csv} ({len(merged):,} rows)")
    print(f"Wrote {output_map} ({len(index_rows):,} rows)")
    print(f"Report  {OUTPUT_REPORT}")
    return report


def main() -> None:
    print("Merge decap combination CSVs (legacy first, then new)")
    print(f"  Old : {OLD_CSV}")
    print(f"  New : {NEW_CSV}")
    print(f"  Map : {DECAP_INDEX_MAP}")
    mode = "EXECUTE" if EXECUTE else "DRY-RUN"
    print(f"  Mode: {mode}\n")

    report = merge_combination_csvs(
        old_csv=OLD_CSV,
        new_csv=NEW_CSV,
        legacy_map_csv=DECAP_INDEX_MAP,
        output_csv=OUTPUT_CSV,
        output_map=OUTPUT_MAP,
        execute=EXECUTE,
    )
    print(
        f"\nMerged {report['n_merged']:,} layouts "
        f"({report['n_old']:,} legacy + {report['n_new']:,} combinations)"
    )
    print(f"  {report['pi_range_legacy']}")
    print(f"  {report['pi_range_combinations']}")


if __name__ == "__main__":
    main()
