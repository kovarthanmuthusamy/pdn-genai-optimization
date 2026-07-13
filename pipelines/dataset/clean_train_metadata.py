#!/usr/bin/env python3
"""Remove append progress, batch registries, and manifest backups from data_multifreq_train.

Keeps only training essentials:
  manifest.csv, dataset_meta.json, heatmap/, PI_freq/, layouts/
  (+ runtime caches multifreq_meta.json, k_values_cache.npy if present)

Run:
    python pipelines/dataset/clean_train_metadata.py
    python pipelines/dataset/clean_train_metadata.py --execute
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

TRAIN_ROOT = REPO_ROOT / "datasets" / "data_multifreq_train"

# Append / restore bookkeeping — not used by training or normalization
REMOVE_FILES = (
    "append_combinations_progress.json",
    "append_merged_progress.json",
    "combinations_append_index.json",
    "merged_append_index.json",
    "subset_meta.json",
    "restore_49k_stepwise_progress.json",
)

REMOVE_DIRS = ("append_batches",)

KEEP_FILES = frozenset(
    {
        "manifest.csv",
        "dataset_meta.json",
        "multifreq_meta.json",
        "k_values_cache.npy",
        "subsample_inverse_k_report.json",
    }
)


def clean_train_metadata(root: Path, *, execute: bool) -> dict[str, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    stats = {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0}

    for name in REMOVE_FILES:
        p = root / name
        if not p.is_file():
            continue
        size = p.stat().st_size
        if execute:
            p.unlink()
        print(f"  {'remove' if execute else 'would remove'} file: {p.name} ({size:,} B)")
        stats["files_removed"] += 1
        stats["bytes_freed"] += size

    for name in REMOVE_DIRS:
        p = root / name
        if not p.is_dir():
            continue
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if execute:
            shutil.rmtree(p)
        print(f"  {'remove' if execute else 'would remove'} dir:  {p.name}/ ({size:,} B)")
        stats["dirs_removed"] += 1
        stats["bytes_freed"] += size

    for p in sorted(root.glob("manifest.csv.bak_*")):
        size = p.stat().st_size
        if execute:
            p.unlink()
        print(f"  {'remove' if execute else 'would remove'} backup: {p.name} ({size:,} B)")
        stats["files_removed"] += 1
        stats["bytes_freed"] += size

    # Report unexpected json/csv at root
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix in (".json", ".csv") and p.name not in KEEP_FILES:
            print(f"  note: leftover {p.name} (not in remove list)")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Clean data_multifreq_train metadata [{mode}]\n  {TRAIN_ROOT}\n")

    stats = clean_train_metadata(TRAIN_ROOT, execute=args.execute)
    mb = stats["bytes_freed"] / (1024 * 1024)
    print(
        f"\n{'Removed' if args.execute else 'Would remove'}: "
        f"{stats['files_removed']} files, {stats['dirs_removed']} dirs (~{mb:.1f} MB)"
    )
    print("\nKept: manifest.csv, dataset_meta.json, heatmap/, PI_freq/, layouts/")


if __name__ == "__main__":
    main()
