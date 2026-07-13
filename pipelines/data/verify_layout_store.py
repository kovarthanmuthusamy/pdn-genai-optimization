"""Verify, migrate, and prune layout-centric multifreq dataset storage.

Run: python pipelines/data/verify_layout_store.py"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


from repo_paths import repo_path, setup_path

setup_path()
from src_vae.others.multifreq_layout_store import migrate_to_layout_store

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/data/verify_layout_store.py
# =============================================================================

ROOTS = [repo_path("datasets", "data_multifreq"), repo_path("datasets", "data_multifreq_norm")]
MIGRATE = False
PRUNE = False

# =============================================================================


def verify(root: Path) -> bool:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        print(f"SKIP {root}: no manifest")
        return False

    with manifest.open(newline="", encoding="utf-8") as f:
        designs = {r["design_id"] for r in csv.DictReader(f) if r.get("design_id")}

    missing = [
        did for did in designs
        if not (root / "layouts" / did / "imp.npy").is_file()
        or not (root / "layouts" / did / "occ.npy").is_file()
    ]
    n_legacy = len(list((root / "Imp").glob("*.npy"))) if (root / "Imp").is_dir() else 0
    print(f"{root.name}: designs={len(designs)} ok={len(designs)-len(missing)} missing={len(missing)} legacy_Imp={n_legacy}")
    if missing[:5]:
        print(f"  examples: {missing[:5]}")
    return not missing


def prune_legacy(root: Path) -> int:
    if not verify(root):
        raise RuntimeError(f"Layout store incomplete under {root}")
    removed = 0
    for sub in ("Imp", "Occ_map"):
        d = root / sub
        if d.is_dir():
            for p in d.glob("*.npy"):
                p.unlink(missing_ok=True)
                removed += 1
    return removed


def main() -> None:
    ok = True
    for root in ROOTS:
        if not root.is_dir():
            continue
        if MIGRATE:
            print(f"Migrating {root} … → {migrate_to_layout_store(root, prune_legacy=False)}")
        if not verify(root):
            ok = False
        elif PRUNE:
            print(f"  pruned {prune_legacy(root)} legacy files under {root.name}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
