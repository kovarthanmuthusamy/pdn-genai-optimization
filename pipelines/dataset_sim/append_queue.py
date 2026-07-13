"""Persistent append job queue — sim enqueues MHz; one worker processes serially."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import REPO_ROOT

QUEUE_DIR = REPO_ROOT / "logs" / "append_queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"
DONE_FILE = QUEUE_DIR / "done.jsonl"
QUEUE_LOCK_FILE = QUEUE_DIR / ".queue.lock"
WORKER_PID_FILE = QUEUE_DIR / "worker.pid"
WORKER_LOG = REPO_ROOT / "logs" / "append_worker.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _pending_mhz_tags() -> set[int]:
    tags: set[int] = set()
    for row in _read_jsonl(PENDING_FILE):
        tags.add(int(round(float(row["mhz"]))))
    return tags


def _done_mhz_tags(*, success_only: bool = True) -> set[int]:
    tags: set[int] = set()
    for row in _read_jsonl(DONE_FILE):
        if success_only and int(row.get("returncode", 0)) != 0:
            continue
        tags.add(int(round(float(row["mhz"]))))
    return tags


def enqueue_append_job(*, mhz: float, append_tag: str, raw_root: str) -> bool:
    """Add one MHz append job. Returns False if already pending or done."""
    mhz_tag = int(round(float(mhz)))
    if mhz_tag in _pending_mhz_tags() or mhz_tag in _done_mhz_tags():
        return False

    _append_jsonl(
        PENDING_FILE,
        {
            "mhz": mhz_tag,
            "append_tag": append_tag,
            "raw_root": raw_root,
            "enqueued_at": _now_iso(),
        },
    )
    return True


def get_next_job() -> dict | None:
    """Return oldest pending job without removing it (safe if worker is killed mid-run)."""
    import fcntl

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK_FILE.open("w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(PENDING_FILE)
        if not rows:
            return None
        return dict(rows[0])


def complete_job(job: dict, *, returncode: int) -> None:
    """Remove job from pending; record in done log."""
    import fcntl

    mhz_tag = int(round(float(job["mhz"])))
    with QUEUE_LOCK_FILE.open("w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(PENDING_FILE)
        rest = [r for r in rows if int(round(float(r["mhz"]))) != mhz_tag]
        if rest:
            PENDING_FILE.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in rest) + "\n",
                encoding="utf-8",
            )
        else:
            PENDING_FILE.unlink(missing_ok=True)

    mark_job_done({**job, "finished_at": _now_iso()}, returncode=returncode)


def mark_job_done(job: dict, *, returncode: int) -> None:
    row = {
        "mhz": int(round(float(job["mhz"]))),
        "append_tag": job["append_tag"],
        "raw_root": job.get("raw_root"),
        "enqueued_at": job.get("enqueued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at") or _now_iso(),
        "returncode": int(returncode),
    }
    _append_jsonl(DONE_FILE, row)


def worker_is_running() -> bool:
    if not WORKER_PID_FILE.is_file():
        return False
    try:
        pid = int(WORKER_PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        WORKER_PID_FILE.unlink(missing_ok=True)
        return False


def write_worker_pid() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def clear_worker_pid() -> None:
    WORKER_PID_FILE.unlink(missing_ok=True)


def pending_count() -> int:
    return len(_read_jsonl(PENDING_FILE))


APPEND_WORKER_SCRIPT = REPO_ROOT / "pipelines" / "dataset_sim" / "run_append_worker.py"


def spawn_append_worker() -> subprocess.Popen:
    """Start append worker detached from sim pipeline / terminal session."""
    WORKER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(WORKER_LOG, "a", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(
        [sys.executable, str(APPEND_WORKER_SCRIPT)],
        cwd=str(REPO_ROOT),
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=False,
    )


def ensure_append_worker_running() -> subprocess.Popen | None:
    if worker_is_running():
        return None
    return spawn_append_worker()
