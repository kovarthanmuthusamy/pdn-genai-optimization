#!/usr/bin/env python3
"""Remove broken symlinks and materialize valid ones as real files (no symlinks in datasets)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from libs.dataset_meta import write_dataset_meta
from src_vae.others.multifreq_layout_store import invalidate_training_caches, repair_multifreq_dataset

PRIMARY_RAW = REPO_ROOT / "datasets" / "data_multifreq_train"
PRIMARY_NORM = REPO_ROOT / "datasets" / "data_multifreq_train_norm_unbounded"
LEGACY_NORM = REPO_ROOT / "datasets" / "data_multifreq_train_norm_robust"
BROKEN_STAGING = REPO_ROOT / "datasets" / "data_multifreq_train_expanded"
OLD_NORM = REPO_ROOT / "data_multi_norm_robust"


def _is_broken(path: Path) -> bool:
    return path.is_symlink() and not path.exists()


def _materialize(path: Path, *, dry_run: bool) -> str:
    """Replace symlink with a copy of its target. Returns action label."""
    if not path.is_symlink():
        return "skip"
    try:
        target = path.resolve()
    except OSError:
        if dry_run:
            return "remove_broken"
        path.unlink(missing_ok=True)
        return "removed_broken"

    if not target.exists():
        if dry_run:
            return "remove_broken"
        path.unlink(missing_ok=True)
        return "removed_broken"

    if dry_run:
        return "materialize"

    path.unlink()
    if target.is_dir():
        shutil.copytree(target, path, symlinks=False, dirs_exist_ok=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, path)
    return "materialized"


def clean_tree(root: Path, *, dry_run: bool) -> dict[str, int]:
    """Walk bottom-up so directory symlinks are handled after contents."""
    stats = {
        "materialized": 0,
        "removed_broken": 0,
        "skipped": 0,
    }
    if not root.is_dir():
        return stats

    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        base = Path(dirpath)
        for name in dirnames + filenames:
            p = base / name
            if p.is_symlink():
                paths.append(p)

    for p in paths:
        action = _materialize(p, dry_run=dry_run)
        if action in ("materialized", "materialize"):
            stats["materialized"] += 1
        elif action in ("removed_broken", "remove_broken"):
            stats["removed_broken"] += 1
        else:
            stats["skipped"] += 1
    return stats


def remove_staging_expanded(*, dry_run: bool) -> bool:
    if not BROKEN_STAGING.exists():
        return False
    if dry_run:
        print(f"  DRY-RUN: would remove broken staging tree {BROKEN_STAGING}")
        return True
    print(f"  Removing broken staging tree {BROKEN_STAGING}", flush=True)
    shutil.rmtree(BROKEN_STAGING)
    return True


def repair_raw(*, dry_run: bool) -> None:
    if dry_run or not PRIMARY_RAW.is_dir():
        return
    stats = repair_multifreq_dataset(PRIMARY_RAW, dry_run=False)
    if any(stats.get(k, 0) for k in ("orphan_heatmaps", "orphan_manifest_stems", "manifest_dupes_dropped")):
        print(f"  repair_multifreq_dataset: {stats}")
    invalidate_training_caches(PRIMARY_RAW)
    write_dataset_meta(
        PRIMARY_RAW,
        stage="raw",
        source_script="pipelines/dataset/clean_dataset_symlinks.py",
        extra={"action": "symlink_cleanup"},
    )


def count_symlinks(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        return 0, 0
    total = broken = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                total += 1
                if not p.exists():
                    broken += 1
    return total, broken


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dry_run = not args.execute
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"Dataset symlink cleanup [{mode}]\n")

    targets = [PRIMARY_RAW, PRIMARY_NORM, LEGACY_NORM, OLD_NORM]
    for root in targets:
        n, br = count_symlinks(root)
        print(f"Before {root.name}: symlinks={n:,} broken={br:,}")

    if remove_staging_expanded(dry_run=dry_run):
        print()

    for root in targets:
        if not root.is_dir():
            continue
        print(f"Cleaning {root}...")
        stats = clean_tree(root, dry_run=dry_run)
        print(f"  materialized={stats['materialized']:,} removed_broken={stats['removed_broken']:,}")

    repair_raw(dry_run=dry_run)

    print("\nAfter:")
    for root in targets:
        n, br = count_symlinks(root)
        print(f"  {root.name}: symlinks={n:,} broken={br:,}")
    if BROKEN_STAGING.exists():
        n, br = count_symlinks(BROKEN_STAGING)
        print(f"  data_multifreq_train_expanded: symlinks={n:,} broken={br:,}")
    else:
        print("  data_multifreq_train_expanded: removed")


if __name__ == "__main__":
    main()
