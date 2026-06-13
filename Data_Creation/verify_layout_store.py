"""Verify / migrate / prune layout-centric multifreq storage.

  python Data_Creation/verify_layout_store.py              # check
  python Data_Creation/verify_layout_store.py --migrate  # legacy Imp/Occ → layouts/
  python Data_Creation/verify_layout_store.py --prune      # delete duplicate Imp/Occ_map
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO))

from src_vae.others.multifreq_layout_store import migrate_to_layout_store  # noqa: E402


def verify(root: Path) -> bool:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        print(f"SKIP {root}: no manifest")
        return False
    designs: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("design_id"):
                designs.add(row["design_id"])
    missing: list[str] = []
    for did in designs:
        sub = root / "layouts" / did
        if not (sub / "imp.npy").is_file() or not (sub / "occ.npy").is_file():
            missing.append(did)
    n_imp_legacy = len(list((root / "Imp").glob("*.npy"))) if (root / "Imp").is_dir() else 0
    print(
        f"{root.name}: designs={len(designs)} layouts_ok={len(designs) - len(missing)} "
        f"missing={len(missing)} legacy_Imp={n_imp_legacy}",
    )
    if missing[:5]:
        print(f"  examples: {missing[:5]}")
    return len(missing) == 0


def prune_legacy(root: Path) -> int:
    """Remove per-sample Imp/ and Occ_map/ after layout store is complete."""
    if not verify(root):
        raise RuntimeError(f"Layout store incomplete under {root} — not pruning")
    removed = 0
    for sub in ("Imp", "Occ_map"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.npy"):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Verify or prune legacy Imp/Occ_map duplicates")
    ap.add_argument(
        "--migrate",
        action="store_true",
        help="Copy legacy per-sample Imp/Occ_map into layouts/{design_id}/",
    )
    ap.add_argument("--prune", action="store_true", help="Delete Imp/*.npy and Occ_map/*.npy after verify")
    ap.add_argument("roots", nargs="*", type=Path)
    args = ap.parse_args()
    roots = args.roots or [
        _REPO / "datasets" / "data_multifreq",
        _REPO / "datasets" / "data_multifreq_norm",
    ]
    ok = True
    for r in roots:
        if not r.is_dir():
            continue
        if args.migrate:
            print(f"Migrating {r} …")
            print(f"  {migrate_to_layout_store(r, prune_legacy=False)}")
        if not verify(r):
            ok = False
            continue
        if args.prune:
            n = prune_legacy(r)
            print(f"  pruned {n} legacy files under {r.name}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
