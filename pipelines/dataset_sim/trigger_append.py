"""Enqueue append jobs and ensure a detached worker is running."""
from __future__ import annotations

from pathlib import Path

from repo_paths import REPO_ROOT

from pipelines.dataset_sim.append_queue import (
    WORKER_LOG,
    ensure_append_worker_running,
    enqueue_append_job,
    pending_count,
)
from pipelines.dataset_sim.paths import RAW_ROOT_WIN, verify_heatmap_ready


def append_tag_for_mhz(mhz: float) -> str:
    return f"merged_{int(round(float(mhz)))}"


def trigger_append_after_move(
    mhz: float,
    *,
    raw_root_win: str = RAW_ROOT_WIN,
) -> Path:
    """Validate moved heatmaps, enqueue append, start worker if needed (no sim wait)."""
    mhz_val = float(mhz)
    mhz_tag = int(round(mhz_val))
    hm_dir = verify_heatmap_ready(mhz_val, raw_root_win=raw_root_win)
    tag = append_tag_for_mhz(mhz_val)

    added = enqueue_append_job(mhz=mhz_tag, append_tag=tag, raw_root=raw_root_win)
    proc = ensure_append_worker_running()

    per_mhz_log = REPO_ROOT / "logs" / f"append_merged_{mhz_tag}.log"
    if proc is not None:
        print(f"  → Append worker started (pid {proc.pid}, log: {WORKER_LOG})")
    if added:
        print(
            f"  → Queued append for {mhz_tag} MHz "
            f"(raw: {hm_dir.name}/, pending={pending_count()})"
        )
    else:
        print(
            f"  → Append for {mhz_tag} MHz already queued or done "
            f"(pending={pending_count()})"
        )
    return per_mhz_log
