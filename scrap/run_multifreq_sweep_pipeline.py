"""End-to-end multifreq heatmap sweep: generate → ECADStar batch → move/compare/report.

All tunables live in the CONFIGURATION block below. CLI flags override config when passed.

Run
---
    python scrap/run_multifreq_sweep_pipeline.py
    python scrap/run_multifreq_sweep_pipeline.py --mhz 80 250
    python scrap/run_multifreq_sweep_pipeline.py --skip-generate --skip-simulate
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

# --- Model / VAE generate (exp043: global-max heatmaps) ---
EXPERIMENT_DIR = "experiments/exp043"
CHECKPOINT_PATH = f"{EXPERIMENT_DIR}/checkpoints/last_model.pt"
DATA_DIR = "datasets/data_multifreq_gmax"

# PI sweep: explicit MHz list, or "anchors" | "dense"
SWEEP: list[float] | str = [10, 70, 120, 270, 400]
DENSE_N_POINTS = 24

K_VALUE = 30
OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep_{K_VALUE}"  # auto-synced when K changes
NUM_SAMPLES = 2
SHARED_TEMP = 1.5
SEED = 42

# PEB (heatmap-only = 1 PI per sample in ECADStar)
HEATMAP_ONLY_PEB = True
POWERBUS = "Power_GND"
FORCE_CPU = False
BACKGROUND_MARGIN = 0.5

# Inference: marginal | layout | anchor_blend | encode
INFERENCE_MODE = "anchor_blend"
PI_REF_MHZ = 200.0
CALIBRATE_FG_MAX = False
INFERENCE_FACTORIZED_ONLY = False

# --- Windows copy destinations ---
PEB_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\design\PEB"
REPORT_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\reports"

# --- ECADStar automation (Windows paths) ---
ECADSTAR_ERF_PATH = r"C:\Users\muthusamy\Desktop\design\H-shape.emc\H-shape.erf"
ECADSTAR_EMC_OUTPUT_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"
ECADSTAR_SKIP_OPEN_ERF = False
ECADSTAR_AHK_EXE: str | None = None  # None → AutoHotkey v2 default in .ps1
# Wait for batch sim to finish (AHK only *starts* Load Batch, then exits)
ECADSTAR_WAIT_FOR_PI = True
ECADSTAR_WAIT_TIMEOUT_SEC = 7200
ECADSTAR_WAIT_POLL_SEC = 40
ECADSTAR_CLOSE_AFTER_BATCH = True

# --- Move PI outputs (after ECADStar) ---
SOURCE_EMC_DIR = ECADSTAR_EMC_OUTPUT_DIR
MOVE = True
CLEAN_DEST_BEFORE_PASTE = True

# --- Pipeline step control (True = skip that step) ---
SKIP_GENERATE = False
SKIP_SIMULATE = False
SKIP_MOVE = False
SKIP_COMPARE = False
SKIP_REPORT = False
SALVAGE_MISPLACED = False

# =============================================================================


def _sync_output_root() -> None:
    """Keep OUTPUT_ROOT aligned with K_VALUE unless CLI --out overrides."""
    global OUTPUT_ROOT
    OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep_{K_VALUE}"


def _peb_out_file() -> str:
    return f"{OUTPUT_ROOT}/pi_distribution_K{K_VALUE}_freq_sweep.peb"


def _expected_peb_name() -> str:
    return f"pi_distribution_K{K_VALUE}_freq_sweep.peb"


def _peb_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_stage_peb_for_ecadstar() -> Path:
    """Ensure ECADStar loads the same PEB that generate wrote (name + content).

    AHK types **filename only** in Load Batch (see scripts/ecadstar_piemi_batch.ahk),
    so we copy the generated PEB beside the .erf as well as PEB_COPY_DEST.
    """
    expected_name = _expected_peb_name()
    repo_peb = (_PROJECT_ROOT / _peb_out_file()).resolve()

    if not repo_peb.is_file():
        raise SystemExit(
            f"Generated PEB not found: {repo_peb}\n"
            f"  Expected name: {expected_name} (K={K_VALUE})\n"
            "  Run without --skip-generate first."
        )
    if repo_peb.name != expected_name:
        raise SystemExit(
            f"PEB filename mismatch: {repo_peb.name!r} != {expected_name!r} (K={K_VALUE})"
        )

    repo_fp = _peb_md5(repo_peb)
    print("  PEB verification:")
    print(f"    Generated : {repo_peb}")
    print(f"    Name      : {expected_name}")
    print(f"    MD5       : {repo_fp}")

    staged: list[Path] = []
    if PEB_COPY_DEST:
        dest = _resolve_windows_path(PEB_COPY_DEST) / expected_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_peb, dest)
        staged.append(dest)

    erf_dir = _resolve_windows_path(ECADSTAR_ERF_PATH).parent
    erf_peb = erf_dir / expected_name
    erf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_peb, erf_peb)
    staged.append(erf_peb)

    for copy_path in staged:
        if copy_path.name != expected_name:
            raise SystemExit(f"Staged PEB name mismatch: {copy_path.name!r}")
        if _peb_md5(copy_path) != repo_fp:
            raise SystemExit(f"Staged PEB content mismatch vs generated: {copy_path}")
        print(f"    Staged    : {copy_path}")

    # Warn if other K PEBs sit in the same folders (easy to load wrong batch manually)
    for folder in {p.parent for p in staged}:
        for other in folder.glob("pi_distribution_K*_freq_sweep.peb"):
            if other.name == expected_name:
                continue
            print(f"    Note: other PEB in folder: {other.name} (ECADStar will type {expected_name!r})")

    simulate_peb = staged[0] if PEB_COPY_DEST else repo_peb
    print(f"    AHK types : {expected_name!r}  (filename only in Load Batch dialog)")
    print(f"    -PebPath  : {_windows_path_str(simulate_peb)}")
    return simulate_peb


def _parse_pi_number(name: str) -> int | None:
    m = re.match(r"^PI-(\d+)(?:\..+)?$", name)
    return int(m.group(1)) if m else None


def _expected_pi_count(freq_list: list[int]) -> int:
    pps = 1 if HEATMAP_ONLY_PEB else 2
    return len(freq_list) * NUM_SAMPLES * pps


def _pi_folder(emc_dir: Path, pi_num: int) -> Path | None:
    for item in emc_dir.iterdir():
        if _parse_pi_number(item.name) == pi_num:
            return item
    return None


def _pi_map_mtime(pi_dir: Path) -> float | None:
    maps = list(pi_dir.rglob("Z_*MHz.map"))
    if not maps:
        return None
    return max(m.stat().st_mtime for m in maps)


def _pi_ready_after(emc_dir: Path, pi_num: int, since_ts: float) -> bool:
    folder = _pi_folder(emc_dir, pi_num)
    if folder is None or not folder.is_dir():
        return False
    mtime = _pi_map_mtime(folder)
    return mtime is not None and mtime >= since_ts - 2.0


def wait_for_pi_outputs(freq_list: list[int], *, batch_start_ts: float) -> None:
    """Poll EMC until PI-1..PI-N have fresh .map files (batch simulation finished)."""
    if not ECADSTAR_WAIT_FOR_PI:
        return
    emc = _resolve_windows_path(SOURCE_EMC_DIR)
    need = list(range(1, _expected_pi_count(freq_list) + 1))
    print(f"\nWaiting for {len(need)} PI output(s) in {emc} (timeout {ECADSTAR_WAIT_TIMEOUT_SEC}s)…")
    deadline = time.time() + ECADSTAR_WAIT_TIMEOUT_SEC
    while time.time() < deadline:
        ready = [n for n in need if _pi_ready_after(emc, n, batch_start_ts)]
        if len(ready) >= len(need):
            print(f"✓ All {len(need)} PI outputs present with fresh .map files.")
            return
        print(
            f"  … {len(ready)}/{len(need)} PI outputs ready "
            f"(need new .map after batch start) — retry in {ECADSTAR_WAIT_POLL_SEC}s"
        )
        time.sleep(ECADSTAR_WAIT_POLL_SEC)
    raise SystemExit(
        f"Timeout waiting for PI outputs in {emc}\n"
        f"  Expected PI numbers: {need}\n"
        "  Increase ECADSTAR_WAIT_TIMEOUT_SEC or re-run with --skip-generate --skip-simulate "
        "after batch finishes."
    )


def close_ecadstar_piemi() -> None:
    """Close eCADSTAR PI/EMI window after batch completes."""
    if not ECADSTAR_CLOSE_AFTER_BATCH:
        return
    ps = (
        "Get-Process | Where-Object { $_.MainWindowTitle -like '*PI/EMI*' } "
        "| Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    print("Closing eCADSTAR PI/EMI window…")
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        cwd=str(_PROJECT_ROOT),
    )


def _apply_sweep_config(mod) -> None:
    """Push pipeline CONFIG into run_multifreq_heatmap_sweep module globals."""
    mod.EXPERIMENT_DIR = EXPERIMENT_DIR
    mod.CHECKPOINT_PATH = CHECKPOINT_PATH
    mod.DATA_DIR = DATA_DIR
    mod.OUTPUT_ROOT = OUTPUT_ROOT
    mod.SWEEP = SWEEP
    mod.DENSE_N_POINTS = DENSE_N_POINTS
    mod.K_VALUE = K_VALUE
    mod.NUM_SAMPLES = NUM_SAMPLES
    mod.SHARED_TEMP = SHARED_TEMP
    mod.SEED = SEED
    mod.HEATMAP_ONLY_PEB = HEATMAP_ONLY_PEB
    mod.PEB_OUT_FILE = _peb_out_file()
    mod.POWERBUS = POWERBUS
    mod.PEB_COPY_DEST = PEB_COPY_DEST
    mod.REPORT_COPY_DEST = REPORT_COPY_DEST
    mod.FORCE_CPU = FORCE_CPU
    mod.BACKGROUND_MARGIN = BACKGROUND_MARGIN
    mod.INFERENCE_MODE = INFERENCE_MODE
    mod.PI_REF_MHZ = PI_REF_MHZ
    mod.CALIBRATE_FG_MAX = CALIBRATE_FG_MAX
    mod.INFERENCE_FACTORIZED_ONLY = INFERENCE_FACTORIZED_ONLY


def _apply_move_config(mod) -> None:
    mod.SOURCE_EMC_DIR = SOURCE_EMC_DIR
    mod.MOVE = MOVE
    mod.CLEAN_DEST_BEFORE_PASTE = CLEAN_DEST_BEFORE_PASTE


def _apply_all_config() -> None:
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402
    import scrap.multifreq_move_and_compare as mc  # noqa: E402

    _apply_sweep_config(sweep)
    _apply_move_config(mc)


def _windows_path_str(path: Path | str) -> str:
    p = Path(path).resolve()
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    if re.match(r"^[A-Za-z]:\\", str(p)):
        return str(p)
    return str(p)


def _resolve_windows_path(path_str: str) -> Path:
    if re.match(r"^[A-Za-z]:\\", path_str) or re.match(r"^[A-Za-z]:/", path_str):
        win = PureWindowsPath(path_str.replace("/", "\\"))
        drive = win.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*win.parts[1:])
        if wsl_path.exists():
            return wsl_path
        return Path(path_str)
    return Path(path_str)


# ---------------------------------------------------------------------------
# ECADStar automation (copied from active_learning_pi/al/ecadstar.py)
# ---------------------------------------------------------------------------

def run_ecadstar_batch(peb_path: Path, groot: Path) -> int:
    """Invoke Windows PowerShell + AutoHotkey to Load Batch in PI/EMI."""
    erf = Path(ECADSTAR_ERF_PATH)
    if not erf.is_file():
        wsl_erf = _resolve_windows_path(ECADSTAR_ERF_PATH)
        if wsl_erf.is_file():
            erf = wsl_erf
        else:
            raise FileNotFoundError(f"ERF not found: {erf}")
    if not peb_path.is_file():
        raise FileNotFoundError(f"PEB not found: {peb_path}")

    typed_name = peb_path.name
    expected_name = _expected_peb_name()
    if typed_name != expected_name:
        raise SystemExit(
            f"ECADStar PEB name mismatch: will type {typed_name!r}, "
            f"expected generated {expected_name!r} (K={K_VALUE})"
        )

    ps_script = groot / "scripts" / "run_ecadstar_piemi_batch.ps1"
    if not ps_script.is_file():
        raise FileNotFoundError(f"AHK runner not found: {ps_script}")

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _windows_path_str(ps_script),
        "-ErfPath",
        _windows_path_str(erf),
        "-PebPath",
        _windows_path_str(peb_path),
    ]
    if ECADSTAR_AHK_EXE:
        cmd.extend(["-AhkExe", str(ECADSTAR_AHK_EXE)])
    if ECADSTAR_SKIP_OPEN_ERF:
        cmd.append("-SkipOpenErf")

    log_hint = Path(os.environ.get("TEMP", "/tmp")) / "ecadstar_piemi_batch.log"
    print(f"Running ECADSTAR automation: {' '.join(cmd)}")
    print(f"  Load Batch will type: {typed_name!r} (must match generated PEB)")
    print(f"  Log (Windows): {log_hint}")
    proc = subprocess.run(cmd, cwd=str(groot), env=os.environ.copy())
    return int(proc.returncode)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _patch_argv(script_name: str, extra: list[str]) -> list[str]:
    old = sys.argv
    sys.argv = [script_name, *extra]
    return old


def _effective_sweep(args: argparse.Namespace) -> list[float] | str:
    if args.mhz is not None:
        return [float(m) for m in args.mhz]
    if args.sweep is not None:
        return args.sweep
    return SWEEP


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """CLI flags override CONFIGURATION block when provided."""
    global SWEEP, K_VALUE, OUTPUT_ROOT, INFERENCE_MODE, PI_REF_MHZ, CALIBRATE_FG_MAX, SOURCE_EMC_DIR

    if args.mhz is not None or args.sweep is not None:
        SWEEP = _effective_sweep(args)
    if args.k is not None:
        K_VALUE = args.k
    if args.out is not None:
        OUTPUT_ROOT = args.out
    else:
        _sync_output_root()
    if args.inference_mode is not None:
        INFERENCE_MODE = args.inference_mode
    if args.pi_ref_mhz is not None:
        PI_REF_MHZ = args.pi_ref_mhz
    if args.no_calibrate_fg:
        CALIBRATE_FG_MAX = False
    if args.source_emc is not None:
        SOURCE_EMC_DIR = args.source_emc


def step_generate(args: argparse.Namespace) -> None:
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    _apply_cli_overrides(args)
    _apply_all_config()

    print("\n" + "=" * 72)
    print("STEP 1/3 — Generate samples + PEB")
    print("=" * 72)
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Output     : {OUTPUT_ROOT}")
    print(f"  K={K_VALUE}  samples={NUM_SAMPLES}  inference={INFERENCE_MODE}")

    gen_args = ["--mode", "generate"]
    if isinstance(SWEEP, list):
        gen_args.extend(["--mhz", *[str(m) for m in SWEEP]])
    else:
        gen_args.extend(["--sweep", str(SWEEP)])
    gen_args.extend(["--k", str(K_VALUE), "--out", OUTPUT_ROOT])
    gen_args.extend(["--inference-mode", INFERENCE_MODE])
    gen_args.extend(["--pi-ref-mhz", str(PI_REF_MHZ)])
    if not CALIBRATE_FG_MAX:
        gen_args.append("--no-calibrate-fg")

    old_argv = _patch_argv("run_multifreq_heatmap_sweep.py", gen_args)
    try:
        sweep.main()
    finally:
        sys.argv = old_argv


def step_simulate(args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print("STEP 2/3 — ECADStar batch simulate (wait + close)")
    print("=" * 72)
    _apply_cli_overrides(args)
    _apply_all_config()
    peb = verify_and_stage_peb_for_ecadstar()
    print(f"  Using PEB: {peb}")
    print(f"  EMC outputs → {ECADSTAR_EMC_OUTPUT_DIR}")

    batch_start = time.time()
    rc = run_ecadstar_batch(peb, _PROJECT_ROOT)
    if rc != 0:
        raise SystemExit(
            f"ECADStar automation failed (exit code {rc}). "
            "Check %TEMP%\\ecadstar_piemi_batch.log on Windows."
        )

    eff = _effective_sweep(args)
    freq_list = [int(round(m)) for m in eff] if isinstance(eff, list) else []
    if freq_list:
        wait_for_pi_outputs(freq_list, batch_start_ts=batch_start)
    close_ecadstar_piemi()


def step_move_compare_report(args: argparse.Namespace) -> None:
    import scrap.multifreq_move_and_compare as mc  # noqa: E402
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    _apply_cli_overrides(args)
    _apply_all_config()
    os.chdir(_PROJECT_ROOT)

    mc.SOURCE_EMC_DIR_OVERRIDE = SOURCE_EMC_DIR

    freq_override: list[int] | None = None
    eff = _effective_sweep(args)
    if isinstance(eff, list):
        freq_override = [int(round(m)) for m in eff]
        sweep.write_sweep_freq_manifest([float(m) for m in freq_override])

    salvage = args.salvage_misplaced or SALVAGE_MISPLACED
    skip_move = args.skip_move or SKIP_MOVE

    print("\n" + "=" * 72)
    print("STEP 3/3 — Move PI outputs, compare heatmaps, build report")
    print("=" * 72)

    if salvage:
        fl = freq_override if freq_override is not None else mc.exported_freq_mhz_list()
        print(f"\n=== Salvage misplaced heatmaps ({len(fl)} freq(s), K={K_VALUE}) ===")
        mc.salvage_sweep_heatmap_outputs(
            freq_list=fl,
            base_generated_dir=_PROJECT_ROOT / OUTPUT_ROOT,
            k_value=K_VALUE,
            num_samples=NUM_SAMPLES,
        )
    elif not skip_move:
        mc.run_move(freq_override)
    else:
        print("Skipping move step.")

    if not (args.skip_compare or SKIP_COMPARE):
        mc.run_compare()
    else:
        print("Skipping compare step.")

    if not (args.skip_report or SKIP_REPORT):
        mc.run_report()
    else:
        print("Skipping report step.")


def _print_config_summary() -> None:
    sweep_desc = (
        ", ".join(str(int(m)) if m == round(m) else f"{m:g}" for m in SWEEP)
        if isinstance(SWEEP, list)
        else str(SWEEP)
    )
    print("Pipeline configuration:")
    print(f"  experiment   : {EXPERIMENT_DIR}")
    print(f"  checkpoint   : {CHECKPOINT_PATH}")
    print(f"  output       : {OUTPUT_ROOT}")
    print(f"  K            : {K_VALUE}  samples={NUM_SAMPLES}")
    print(f"  sweep MHz    : {sweep_desc}")
    print(f"  inference    : {INFERENCE_MODE}  pi_ref={PI_REF_MHZ} MHz")
    print(f"  PEB copy     : {PEB_COPY_DEST}")
    print(f"  report copy  : {REPORT_COPY_DEST}")
    print(f"  ECADStar erf : {ECADSTAR_ERF_PATH}")
    print(f"  EMC / move   : {SOURCE_EMC_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multifreq sweep pipeline: generate → ECADStar → move/compare/report",
    )
    ap.add_argument("--skip-generate", action="store_true", help="Skip sample/PEB generation")
    ap.add_argument("--skip-simulate", action="store_true", help="Skip ECADStar batch automation")
    ap.add_argument("--skip-move", action="store_true", help="Skip move PI outputs step")
    ap.add_argument("--skip-compare", action="store_true", help="Skip heatmap compare plots")
    ap.add_argument("--skip-report", action="store_true", help="Skip HTML/markdown report")
    ap.add_argument(
        "--salvage-misplaced",
        action="store_true",
        help="Salvage misplaced Real/ folders by .map MHz (move step only)",
    )
    ap.add_argument("--mhz", nargs="+", type=float, default=None, metavar="MHZ")
    ap.add_argument("--sweep", default=None, help="anchors | dense (overrides CONFIG SWEEP)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--inference-mode",
        choices=("marginal", "layout", "anchor_blend", "encode"),
        default=None,
    )
    ap.add_argument("--pi-ref-mhz", type=float, default=None)
    ap.add_argument("--no-calibrate-fg", action="store_true")
    ap.add_argument("--source-emc", default=None, help="Override SOURCE_EMC_DIR / EMC output dir")

    args = ap.parse_args()
    os.chdir(_PROJECT_ROOT)
    _sync_output_root()
    _apply_cli_overrides(args)
    _apply_all_config()
    _print_config_summary()

    if not (args.skip_generate or SKIP_GENERATE):
        step_generate(args)
    else:
        print("\nSkipping generate step.")

    if not (args.skip_simulate or SKIP_SIMULATE):
        step_simulate(args)
    else:
        print("\nSkipping ECADStar simulate step.")

    step_move_compare_report(args)

    print("\n" + "=" * 72)
    print("✓ Pipeline complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
