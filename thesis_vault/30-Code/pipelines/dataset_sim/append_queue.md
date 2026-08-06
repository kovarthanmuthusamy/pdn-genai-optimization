---
title: append_queue
type: code
path: pipelines/dataset_sim/append_queue.py
group: pipelines/dataset_sim
loc: 172
tags: [code, pipelines]
---

# append_queue

> Persistent append job queue — sim enqueues MHz; one worker processes serially.

**Source:** `pipelines/dataset_sim/append_queue.py` · 172 lines

## Constants

| Name | Value |
|------|-------|
| `QUEUE_DIR` | `REPO_ROOT / 'logs' / 'append_queue'` |
| `PENDING_FILE` | `QUEUE_DIR / 'pending.jsonl'` |
| `DONE_FILE` | `QUEUE_DIR / 'done.jsonl'` |
| `QUEUE_LOCK_FILE` | `QUEUE_DIR / '.queue.lock'` |
| `WORKER_PID_FILE` | `QUEUE_DIR / 'worker.pid'` |
| `WORKER_LOG` | `REPO_ROOT / 'logs' / 'append_worker.log'` |
| `APPEND_WORKER_SCRIPT` | `REPO_ROOT / 'pipelines' / 'dataset_sim' / 'run_append_worker.py'` |

## Functions

- **`_now_iso()`**
- **`_read_jsonl(path: Path)`**
- **`_append_jsonl(path: Path, row: dict)`**
- **`_pending_mhz_tags()`**
- **`_done_mhz_tags(*, success_only: bool=True)`**
- **`enqueue_append_job(*, mhz: float, append_tag: str, raw_root: str)`** — Add one MHz append job. Returns False if already pending or done.
- **`get_next_job()`** — Return oldest pending job without removing it (safe if worker is killed mid-run).
- **`complete_job(job: dict, *, returncode: int)`** — Remove job from pending; record in done log.
- **`mark_job_done(job: dict, *, returncode: int)`**
- **`worker_is_running()`**
- **`write_worker_pid()`**
- **`clear_worker_pid()`**
- **`pending_count()`**
- **`spawn_append_worker()`** — Start append worker detached from sim pipeline / terminal session.
- **`ensure_append_worker_running()`**

## Imports

- [[repo_paths]]

## Imported by

- [[run_append_worker]]
- [[trigger_append]]

## External dependencies

`fcntl`, `repo_paths`
