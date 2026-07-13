#!/usr/bin/env python3
"""Process append_merged queue jobs one at a time (survives sim pipeline continuing).

Run as worker (default):
    python pipelines/dataset_sim/run_append_worker.py

Enqueue recovery / manual jobs:
    python pipelines/dataset_sim/run_append_worker.py --enqueue 470 490
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from libs.dataset_meta import write_dataset_meta
from pipelines.dataset_sim.append_queue import (
    WORKER_LOG,
    clear_worker_pid,
    complete_job,
    enqueue_append_job,
    ensure_append_worker_running,
    get_next_job,
    write_worker_pid,
)
from pipelines.dataset_sim.paths import RAW_ROOT_WIN
from pipelines.dataset_sim.trigger_append import append_tag_for_mhz

APPEND_SCRIPT = REPO_ROOT / "pipelines" / "data" / "append_merged_combinations_multifreq.py"
APPEND_LOCK_FILE = REPO_ROOT / "logs" / ".append_merged.lock"
OUTPUT_ROOT = REPO_ROOT / "datasets" / "data_multifreq_train"

_STOP_REQUESTED = False


def _handle_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[append-worker] signal {signum} — finish current job then stop", flush=True)


def _refresh_meta() -> None:
    write_dataset_meta(
        OUTPUT_ROOT,
        stage="raw",
        source_script="pipelines/data/append_merged_combinations_multifreq.py",
    )


def _sanity_check(mhz_tag: int, append_tag: str) -> None:
    counts: Counter[str] = Counter()
    manifest = OUTPUT_ROOT / "manifest.csv"
    if not manifest.is_file():
        print(f"[append-worker] WARNING: no manifest after {mhz_tag} MHz", flush=True)
        return
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("freq_label") == f"{mhz_tag}MHz":
                counts[row.get("append_tag") or "(none)"] += 1
    total = sum(counts.values())
    print(f"[append-worker] {mhz_tag} MHz manifest rows: {total:,} ({dict(counts)})", flush=True)
    if total < 29499:
        print(
            f"[append-worker] WARNING: expected 29,499 rows at {mhz_tag} MHz "
            f"(tag={append_tag})",
            flush=True,
        )


def _run_one_job(job: dict) -> int:
    mhz_tag = int(round(float(job["mhz"])))
    append_tag = str(job["append_tag"])
    raw_root = str(job.get("raw_root") or "")

    print(
        f"[append-worker] start {mhz_tag} MHz tag={append_tag} raw={raw_root}",
        flush=True,
    )

    APPEND_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with APPEND_LOCK_FILE.open("w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

        cmd = [
            sys.executable,
            str(APPEND_SCRIPT),
            "--mhz",
            str(mhz_tag),
            "--append-tag",
            append_tag,
            "--raw-root",
            raw_root,
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        rc = int(proc.returncode)

    if rc == 0:
        _refresh_meta()
        _sanity_check(mhz_tag, append_tag)
    return rc


def _enqueue_mhz_list(mhz_values: list[float], *, raw_root: str) -> None:
    for mhz in mhz_values:
        tag = append_tag_for_mhz(mhz)
        if enqueue_append_job(mhz=mhz, append_tag=tag, raw_root=raw_root):
            print(f"  enqueued {int(round(mhz))} MHz (tag={tag})")
        else:
            print(f"  skip {int(round(mhz))} MHz (already pending or successfully done)")
    ensure_append_worker_running()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enqueue",
        type=float,
        nargs="+",
        metavar="MHZ",
        help="Enqueue MHz job(s) and start worker if needed",
    )
    parser.add_argument("--raw-root", default=RAW_ROOT_WIN)
    args = parser.parse_args()

    if args.enqueue:
        _enqueue_mhz_list(args.enqueue, raw_root=args.raw_root)
        return

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    write_worker_pid()
    print(f"[append-worker] pid={os.getpid()} log={WORKER_LOG}", flush=True)

    try:
        while not _STOP_REQUESTED:
            job = get_next_job()
            if job is None:
                for _ in range(6):
                    if _STOP_REQUESTED:
                        break
                    time.sleep(5)
                    job = get_next_job()
                    if job is not None:
                        break
            if job is None:
                print("[append-worker] queue empty — exiting", flush=True)
                break

            job["started_at"] = job.get("started_at") or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            rc = _run_one_job(job)
            complete_job(job, returncode=rc)
            if rc == 0:
                print(f"[append-worker] done {job['mhz']} MHz", flush=True)
            else:
                print(
                    f"[append-worker] failed {job['mhz']} MHz (rc={rc}) — "
                    "re-enqueue with: python pipelines/dataset_sim/run_append_worker.py --enqueue 470",
                    flush=True,
                )
    finally:
        clear_worker_pid()


if __name__ == "__main__":
    main()
