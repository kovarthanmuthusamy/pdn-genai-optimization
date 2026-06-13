#!/usr/bin/env python3
"""Rename folders PI-19500..PI-38998 to PI-1..PI-19499 (rename only)."""

import argparse
import re
import sys
from pathlib import Path

SOURCE_START = 19500
SOURCE_END = 38998
PREFIX = "PI-"
FOLDER_PATTERN = re.compile(rf"^{re.escape(PREFIX)}(\d+)$")


def collect_renames(root: Path) -> list[tuple[Path, Path]]:
    """Build ordered list of (old_path, new_path) for matching folders."""
    renames: list[tuple[Path, Path]] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = FOLDER_PATTERN.match(entry.name)
        if not match:
            continue
        old_num = int(match.group(1))
        if not (SOURCE_START <= old_num <= SOURCE_END):
            continue
        new_num = old_num - SOURCE_START + 1
        new_name = f"{PREFIX}{new_num}"
        new_path = root / new_name
        renames.append((entry, new_path))

    renames.sort(key=lambda pair: pair[0].name)
    return renames


def apply_renames(renames: list[tuple[Path, Path]], dry_run: bool) -> int:
    """Rename folders via a temporary name to avoid collisions."""
    if not renames:
        print("No matching folders found.")
        return 0

    expected = SOURCE_END - SOURCE_START + 1
    if len(renames) != expected:
        print(
            f"Warning: expected {expected} folders, found {len(renames)}.",
            file=sys.stderr,
        )

    temp_pairs: list[tuple[Path, Path]] = []
    for index, (old_path, new_path) in enumerate(renames, start=1):
        temp_path = old_path.parent / f"__rename_tmp_{index:05d}__"
        temp_pairs.append((old_path, temp_path))

    final_pairs = [(temp, new) for (_, new), (_, temp) in zip(renames, temp_pairs)]

    for old_path, temp_path in temp_pairs:
        print(f"{old_path.name} -> {temp_path.name}")
        if not dry_run:
            old_path.rename(temp_path)

    for temp_path, new_path in final_pairs:
        print(f"{temp_path.name} -> {new_path.name}")
        if not dry_run:
            temp_path.rename(new_path)

    return len(renames)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Rename {PREFIX}{SOURCE_START}..{PREFIX}{SOURCE_END} "
            f"to {PREFIX}1..{PREFIX}{SOURCE_END - SOURCE_START + 1}."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=r"C:\Users\muthusamy\Desktop\Raw\heatmap_450MHz",
        help="Directory containing the folders to rename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing anything",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    renames = collect_renames(root)
    count = apply_renames(renames, dry_run=args.dry_run)

    action = "Would rename" if args.dry_run else "Renamed"
    print(f"\n{action} {count} folder(s) in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
