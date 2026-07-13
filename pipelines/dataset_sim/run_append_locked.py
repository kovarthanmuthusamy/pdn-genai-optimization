#!/usr/bin/env python3
"""Run one append job under an exclusive lock; refresh dataset_meta on success."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from libs.dataset_meta import write_dataset_meta
from pipelines.dataset_sim.paths import RAW_ROOT_WIN

APPEND_SCRIPT = REPO_ROOT / "pipelines" / "data" / "append_merged_combinations_multifreq.py"
LOCK_FILE = REPO_ROOT / "logs" / ".append_merged.lock"
OUTPUT_ROOT = REPO_ROOT / "datasets" / "data_multifreq_train"


def _refresh_meta() -> None:
    write_dataset_meta(
        OUTPUT_ROOT,
        stage="raw",
        source_script="pipelines/data/append_merged_combinations_multifreq.py",
    )
    print(f"  ✓ Refreshed {OUTPUT_ROOT / 'dataset_meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhz", type=float, required=True)
    parser.add_argument("--append-tag", required=True)
    parser.add_argument("--raw-root", default=RAW_ROOT_WIN)
    args = parser.parse_args()

    mhz_tag = int(round(args.mhz))
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"[append-lock] waiting for lock ({mhz_tag} MHz, tag={args.append_tag})…")
    with LOCK_FILE.open("w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        print(f"[append-lock] acquired — running append for {mhz_tag} MHz")

        cmd = [
            sys.executable,
            str(APPEND_SCRIPT),
            "--mhz",
            str(mhz_tag),
            "--append-tag",
            args.append_tag,
            "--raw-root",
            args.raw_root,
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        _refresh_meta()

        # quick sanity check
        import csv
        from collections import Counter

        counts: Counter[str] = Counter()
        manifest = OUTPUT_ROOT / "manifest.csv"
        with manifest.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("freq_label") == f"{mhz_tag}MHz":
                    counts[row.get("append_tag") or "(none)"] += 1
        total = sum(counts.values())
        print(f"[append-lock] {mhz_tag} MHz manifest rows: {total:,} ({dict(counts)})")
        if total < 29499:
            print(
                f"[append-lock] WARNING: expected 29,499 rows at {mhz_tag} MHz — "
                "re-run append or check logs/append_merged_{mhz}.log"
            )


if __name__ == "__main__":
    main()
