#!/usr/bin/env python3
"""Combinations.csv → ECADStar simulation pipeline (impedance, then PI-Distribution per MHz).

One .peb file per phase containing **all** layouts in the selected CSV slice (default 10,000).
ECADStar creates PI-1, PI-2, … in PEB order; outputs are moved with those names unchanged.

Workflow:
  1. ``combinations_impedance.peb`` (all rows) → simulate → ``Impedance/PI-1 … PI-N``
  2. Per MHz: ``combinations_dist_{MHz}MHz.peb`` → simulate → ``heatmaps_{MHz}MHz/PI-1 … PI-N``

Reference: ``scrap/orchestration/run_multifreq_sweep_pipeline.py``

Run:
    python pipelines/dataset_sim/run_combinations_sim_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from pipelines.dataset_sim.combinations import load_combinations_csv
from pipelines.dataset_sim.ecadstar import (
    clear_ecadstar_lock,
    resolve_windows_path,
    run_ecadstar_batch_headless,
    stage_peb_for_ecadstar,
    wait_for_pi_outputs,
)
from pipelines.dataset_sim.paths import PEB_DIR_WIN, RAW_ROOT_WIN, resolve_raw_root
from pipelines.dataset_sim.move_outputs import heatmaps_dir_name, move_pi_outputs
from pipelines.dataset_sim.peb import build_distribution_peb, build_impedance_peb
from pipelines.dataset_sim.trigger_append import trigger_append_after_move
from src_vae.others.multifreq_anchors import load_anchors_mhz

# =============================================================================
# CONFIGURATION — edit before running
# =============================================================================

COMBINATIONS_CSV = REPO_ROOT / "data" / "heatmaps" / "peb_with_29k"/"all_combinations_merged.csv"

# CSV slice (0-based). PI-k in output = row (START_LAYOUT + k - 1) when START_LAYOUT=0.
START_LAYOUT = 0
MAX_LAYOUTS: int | None = None  # None = all rows after START_LAYOUT

# PI-Distribution frequencies (MHz)
USE_ANCHOR_MHZ = False  # True = use anchors from src_vae.others.multifreq_anchors.load_anchors_mhz()
SWEEP_MHZ: list[float] = [170,180,280,430,470,490,550,590]
SKIP_MHZ: set[float] = set()

POWERBUS = "Power_GND"
IMPEDANCE_COMPONENTS = "IC1_Port1,IC2_Port2"

# --- Windows / ECADStar (native headless CLI; see docs/ecadstar_headless_cli.md) ---
# engineer.exe <design.erf> --batch <file.peb> --batch-auto-exit : opens the .erf,
# runs the .peb, writes PI-1..N, and QUITS ITSELF. No GUI window / no focus / no AHK.
PEB_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\design\PEB"
ECADSTAR_ERF_PATH = r"C:\Users\muthusamy\Desktop\design\H-shape.emc\H-shape.erf"
ECADSTAR_EMC_OUTPUT_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"
ENGINEER_EXE = r"C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe"
ECADSTAR_IMPULSE_PORT: int | None = None
ECADSTAR_CLEAR_LOCK_FILE = True
# Full 10k run may take many hours — raise if needed
ECADSTAR_WAIT_TIMEOUT_SEC = 172_800  # 48 h per phase
ECADSTAR_WAIT_POLL_SEC = 120

# --- Step control ---
SKIP_IMPEDANCE = True  # True = skip impedance phase
SKIP_DISTRIBUTION = False
SKIP_SIMULATE = False  # True = only write .peb files
SKIP_MOVE = False
CLEAN_DEST_BEFORE_PASTE = True
RESUME_FROM_PROGRESS = True
# After moving heatmaps for an MHz: launch append script in background (CLI args)
TRIGGER_APPEND_AFTER_MOVE = True

OUTPUT_ROOT = resolve_raw_root()
PEB_DIR = resolve_windows_path(PEB_DIR_WIN)

# =============================================================================


def _mhz_list() -> list[float]:
    if USE_ANCHOR_MHZ:
        return [float(m) for m in load_anchors_mhz()]
    return [float(m) for m in SWEEP_MHZ]


def _progress_path() -> Path:
    return OUTPUT_ROOT / "sim_progress.json"


def _load_progress() -> dict:
    path = _progress_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"impedance_done": False, "distribution_mhz_done": []}


def _save_progress(progress: dict) -> None:
    path = _progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _simulate_peb(
    peb_path: Path,
    *,
    pi_count: int,
    mode: str,
    expected_mhz: float | None = None,
) -> None:
    if SKIP_SIMULATE:
        print("  SKIP_SIMULATE: PEB written, ECADStar not invoked.")
        return

    # Native headless CLI: engineer.exe opens the .erf, runs the .peb, writes
    # PI-1..N, and quits itself. No GUI window / no focus / background-safe.
    print("  ECADStar mode: headless engineer.exe --batch --batch-auto-exit")
    if ECADSTAR_CLEAR_LOCK_FILE:
        clear_ecadstar_lock(ECADSTAR_ERF_PATH)
    stage_peb_for_ecadstar(
        peb_path,
        erf_path=ECADSTAR_ERF_PATH,
        peb_copy_dest=PEB_COPY_DEST,
    )
    batch_start = time.time()
    rc = run_ecadstar_batch_headless(
        peb_path,
        erf_path=ECADSTAR_ERF_PATH,
        engineer_exe=ENGINEER_EXE,
        impulse_port=ECADSTAR_IMPULSE_PORT,
        timeout_sec=ECADSTAR_WAIT_TIMEOUT_SEC,
    )
    if rc != 0:
        raise SystemExit(
            f"ECADStar headless batch failed (rc={rc}) for {peb_path.name}. "
            f"Check {ECADSTAR_EMC_OUTPUT_DIR}\\PI-1\\log.txt on Windows."
        )
    wait_for_pi_outputs(
        ECADSTAR_EMC_OUTPUT_DIR,
        pi_count=pi_count,
        batch_start_ts=batch_start,
        mode=mode,
        timeout_sec=ECADSTAR_WAIT_TIMEOUT_SEC,
        poll_sec=ECADSTAR_WAIT_POLL_SEC,
        expected_mhz=expected_mhz,
    )


def run() -> None:
    if not COMBINATIONS_CSV.is_file():
        raise SystemExit(f"Missing combinations CSV: {COMBINATIONS_CSV}")

    occ_all = load_combinations_csv(COMBINATIONS_CSV)
    end = len(occ_all) if MAX_LAYOUTS is None else min(len(occ_all), START_LAYOUT + MAX_LAYOUTS)
    if START_LAYOUT >= end:
        raise SystemExit(f"START_LAYOUT={START_LAYOUT} >= end={end}")

    occ = occ_all[START_LAYOUT:end]
    n_layouts = int(occ.shape[0])
    mhz_list = [m for m in _mhz_list() if float(m) not in SKIP_MHZ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PEB_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress() if RESUME_FROM_PROGRESS else {
        "impedance_done": False,
        "distribution_mhz_done": [],
    }

    print("Combinations simulation pipeline (single PEB per phase)")
    print(f"  CSV          : {COMBINATIONS_CSV} ({len(occ_all):,} rows)")
    print(f"  This run     : rows {START_LAYOUT}..{end - 1} ({n_layouts:,} layouts)")
    print(f"  PI folders   : PI-1 .. PI-{n_layouts} (PEB row order = CSV order)")
    print(f"  Output       : {OUTPUT_ROOT} ({RAW_ROOT_WIN})")
    print(f"  MHz sweep    : {mhz_list}")
    print(f"  Simulate     : {not SKIP_SIMULATE}")
    if not SKIP_SIMULATE:
        print("  ECADStar     : headless engineer.exe --batch --batch-auto-exit")

    # --- Impedance: one PEB for all layouts ---
    if not SKIP_IMPEDANCE and not progress.get("impedance_done"):
        imp_peb = PEB_DIR / "combinations_impedance.peb"
        print("\n" + "=" * 72)
        print(f"[Impedance] one PEB, {n_layouts:,} layouts → {imp_peb.name}")
        print("=" * 72)
        build_impedance_peb(
            occ,
            imp_peb,
            repo_root=REPO_ROOT,
            powerbus=POWERBUS,
            components=IMPEDANCE_COMPONENTS,
        )
        _simulate_peb(imp_peb, pi_count=n_layouts, mode="impedance")

        if not SKIP_MOVE and not SKIP_SIMULATE:
            n_moved = move_pi_outputs(
                source_emc_dir=ECADSTAR_EMC_OUTPUT_DIR,
                dest_dir=OUTPUT_ROOT / "Impedance",
                pi_count=n_layouts,
                clean_dest=CLEAN_DEST_BEFORE_PASTE,
            )
            print(f"  ✓ Moved {n_moved} folder(s) → Impedance/PI-1..PI-{n_layouts}")

        progress["impedance_done"] = True
        _save_progress(progress)
    elif SKIP_IMPEDANCE:
        print("\n[Impedance] skipped (SKIP_IMPEDANCE)")
    else:
        print("\n[Impedance] already done (resume)")

    # --- PI-Distribution: one PEB per MHz ---
    if SKIP_DISTRIBUTION:
        print("\n[Distribution] skipped (SKIP_DISTRIBUTION)")
    else:
        done_mhz = {int(x) for x in progress.get("distribution_mhz_done", [])}
        for mhz in mhz_list:
            mhz_tag = int(round(mhz))
            if mhz_tag in done_mhz:
                print(f"\n[{mhz_tag} MHz] already done (resume)")
                continue

            dist_peb = PEB_DIR / f"combinations_dist_{mhz_tag}MHz.peb"
            print("\n" + "=" * 72)
            print(f"[{mhz_tag} MHz] one PEB, {n_layouts:,} layouts → {dist_peb.name}")
            print("=" * 72)
            build_distribution_peb(
                occ,
                dist_peb,
                repo_root=REPO_ROOT,
                mhz=mhz,
                powerbus=POWERBUS,
            )
            _simulate_peb(
                dist_peb,
                pi_count=n_layouts,
                mode="distribution",
                expected_mhz=mhz,
            )
            dist_pending_index += 1

            if not SKIP_MOVE and not SKIP_SIMULATE:
                heatmaps_dir = OUTPUT_ROOT / heatmaps_dir_name(mhz)
                n_moved = move_pi_outputs(
                    source_emc_dir=ECADSTAR_EMC_OUTPUT_DIR,
                    dest_dir=heatmaps_dir,
                    pi_count=n_layouts,
                    clean_dest=CLEAN_DEST_BEFORE_PASTE,
                )
                print(f"  ✓ Moved {n_moved} folder(s) → {heatmaps_dir.name}/PI-1..PI-{n_layouts}")

                if TRIGGER_APPEND_AFTER_MOVE:
                    trigger_append_after_move(mhz, raw_root_win=RAW_ROOT_WIN)

            done_mhz.add(mhz_tag)
            progress["distribution_mhz_done"] = sorted(done_mhz)
            _save_progress(progress)

    print("\n" + "=" * 72)
    print("✓ Pipeline complete")
    print(f"  Progress: {_progress_path()}")
    print("=" * 72)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
