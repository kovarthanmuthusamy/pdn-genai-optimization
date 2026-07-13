#!/usr/bin/env python3
"""Remove duplicate manifest rows for one MHz (keep newest sample_N per design_id)."""
from __future__ import annotations

import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from libs.dataset_meta import write_dataset_meta
from src_vae.others.multifreq_layout_store import invalidate_training_caches, manifest_path

DATA_DIR = REPO_ROOT / "datasets" / "data_multifreq_train"
FREQ_MHZ = 470.0
EXECUTE = True


def _sample_index(name: str) -> int:
    m = re.search(r"sample_(\d+)\.npy", name)
    return int(m.group(1)) if m else 0


def _matches_mhz(row: dict[str, str], freq_mhz: float) -> bool:
    label = f"{int(freq_mhz) if freq_mhz == int(freq_mhz) else freq_mhz}MHz"
    if row.get("freq_label") == label:
        return True
    try:
        return abs(float(row.get("freq_mhz", -1)) - freq_mhz) < 1e-6
    except (TypeError, ValueError):
        return False


def dedupe_mhz(data_dir: Path, freq_mhz: float, *, execute: bool) -> dict[str, int]:
    data_dir = data_dir.resolve()
    mf = manifest_path(data_dir)
    hm_dir = data_dir / "heatmap"
    pf_dir = data_dir / "PI_freq"

    with mf.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    by_design: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for row in rows:
        if not _matches_mhz(row, freq_mhz):
            continue
        by_design.setdefault(row["design_id"], []).append((_sample_index(row["sample_name"]), row))

    drop: list[dict[str, str]] = []
    for entries in by_design.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda x: x[0])
        for _, row in entries[:-1]:
            drop.append(row)

    drop_set = {id(r) for r in drop}
    keep = [r for r in rows if id(r) not in drop_set]

    counts = {
        "manifest_before": len(rows),
        "manifest_after": len(keep),
        "rows_removed": len(drop),
        "design_ids_deduped": sum(1 for v in by_design.values() if len(v) > 1),
        "heatmap_removed": 0,
        "pi_freq_removed": 0,
        "errors": 0,
    }

    if not execute:
        return counts

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = data_dir / f"manifest.csv.bak_dedupe_{int(freq_mhz)}MHz_{ts}"
    shutil.copy2(mf, backup)
    print(f"  manifest backup → {backup.name}")

    for row in drop:
        stem = Path(row["sample_name"]).name
        for sub, key in ((hm_dir, "heatmap_removed"), (pf_dir, "pi_freq_removed")):
            if not sub.is_dir():
                continue
            p = sub / stem
            if p.is_file():
                counts[key] += 1
                try:
                    p.unlink()
                except OSError:
                    counts["errors"] += 1

    with mf.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    invalidate_training_caches(data_dir)
    write_dataset_meta(
        data_dir,
        stage="raw",
        source_script="pipelines/dataset/dedupe_mhz_manifest.py",
        extra={"deduped_mhz": freq_mhz, "rows_removed": counts["rows_removed"]},
    )
    return counts


def main() -> None:
    mode = "EXECUTE" if EXECUTE else "DRY-RUN"
    print(f"Dedupe {FREQ_MHZ:g} MHz duplicates in {DATA_DIR} [{mode}]\n")

    counts = dedupe_mhz(DATA_DIR, FREQ_MHZ, execute=EXECUTE)
    print(f"  design_ids deduped: {counts['design_ids_deduped']:,}")
    print(
        f"  manifest: {counts['manifest_before']:,} → {counts['manifest_after']:,} "
        f"(-{counts['rows_removed']:,})",
    )
    if EXECUTE:
        print(
            f"  deleted: {counts['heatmap_removed']:,} heatmap, "
            f"{counts['pi_freq_removed']:,} PI_freq (older duplicate copies)",
        )
        if counts["errors"]:
            print(f"  WARNING: {counts['errors']} delete errors")
    else:
        print("\nDry-run only. Set EXECUTE = True to apply.")


if __name__ == "__main__":
    main()
