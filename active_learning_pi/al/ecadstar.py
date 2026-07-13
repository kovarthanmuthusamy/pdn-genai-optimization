"""ECADSTAR batch simulation bridge (Windows PowerShell + AutoHotkey).

Uses the same path resolution and wait logic as
``pipelines/dataset_sim/ecadstar.py`` and ``run_multifreq_sweep_pipeline.py``.
"""
from __future__ import annotations

import time
from pathlib import Path

from pipelines.dataset_sim.ecadstar import (
    clear_ecadstar_lock,
    resolve_windows_path,
    run_ecadstar_batch as _run_batch,
    stage_peb_for_ecadstar,
    wait_for_batch_started,
    wait_for_pi_outputs,
    windows_path_str,
)


def run_ecadstar_batch(
    cfg: dict,
    peb_path: Path,
    groot: Path,
    *,
    pi_count: int = 1,
) -> int:
    """
    Stage PEB → launch ECADStar Load Batch → optionally wait for PI outputs.

    ``ecadstar`` config keys (Windows paths, same as sweep pipeline):
      erf_path, emc_output_dir, peb_copy_dest (optional),
      skip_open_erf, ahk_exe, clear_lock_file,
      wait_for_pi, wait_timeout_sec, wait_poll_sec
    """
    ec = cfg.get("ecadstar", {})
    erf_path = str(ec["erf_path"])
    emc_output_dir = str(ec["emc_output_dir"])

    erf = resolve_windows_path(erf_path)
    if not erf.is_file():
        raise FileNotFoundError(
            f"ERF not found: {erf_path}\n"
            f"  Resolved: {erf}\n"
            "  Edit active_learning_pi/config/exp057.json → ecadstar.erf_path "
            "(use the same paths as scrap/orchestration/run_multifreq_sweep_pipeline.py)."
        )
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    if bool(ec.get("clear_lock_file", True)):
        clear_ecadstar_lock(erf_path)

    staged_peb = stage_peb_for_ecadstar(
        peb_path.resolve(),
        erf_path=erf_path,
        peb_copy_dest=ec.get("peb_copy_dest"),
    )

    print(f"  ERF: {windows_path_str(erf)}")
    print(f"  EMC: {windows_path_str(resolve_windows_path(emc_output_dir))}")
    print(f"  PEB: {windows_path_str(staged_peb)}")

    batch_start = time.time()
    rc = _run_batch(
        staged_peb,
        erf_path=erf_path,
        repo_root=groot,
        ahk_exe=ec.get("ahk_exe"),
        skip_open_erf=bool(ec.get("skip_open_erf", False)),
    )
    if rc != 0:
        return rc

    if bool(ec.get("wait_for_batch_started", True)):
        wait_for_batch_started(
            emc_output_dir,
            peb_path=staged_peb,
            batch_start_ts=batch_start,
            timeout_sec=int(ec.get("wait_batch_start_timeout_sec", 600)),
            poll_sec=int(ec.get("wait_poll_sec", 40)),
        )

    if bool(ec.get("wait_for_pi", True)) and pi_count > 0:
        wait_for_pi_outputs(
            emc_output_dir,
            pi_count=pi_count,
            batch_start_ts=batch_start,
            mode="distribution",
            timeout_sec=int(ec.get("wait_timeout_sec", 7200)),
            poll_sec=int(ec.get("wait_poll_sec", 40)),
            expected_mhz=None,
        )

    return 0
