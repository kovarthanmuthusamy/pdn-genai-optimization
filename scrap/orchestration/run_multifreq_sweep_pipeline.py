"""Multifreq heatmap sweep pipeline (generate → simulate → compare).

Purpose:
    End-to-end workflow: VAE sample at one or more K values across a MHz sweep, build PEB,
    optional ECADStar PI simulation, move outputs, and comparison report.

Run:
    python scrap/orchestration/run_multifreq_sweep_pipeline.py

Agent notes:
    - What: Orchestrates multifreq heatmap generation and optional Windows ECADStar automation.
    - Usage: Edit CONFIG (model paths, ``K_VALUE``, ``SWEEP``, ECADStar paths) → run on machine with repo + ECADStar access.
    - Config keys:
        - ``EXPERIMENT_DIR`` / ``CHECKPOINT_PATH`` / ``DATA_DIR`` — exp052 + ``data_multifreq_train_norm_unbounded``
        - ``K_VALUE`` — single int (e.g. ``30``) or list (e.g. ``[10, 20, 30]``)
        - ``INFERENCE_MODE`` / ``QC_SWEEP`` / ``LAYOUT_SOURCE`` — QC sweep aligned with latent optimize
        - ``LATENT_RUN_DIR`` — for ``latent_z`` / ``layout_hybrid`` post-opt QC
        - ``SKIP_GENERATE`` … ``SKIP_REPORT`` — skip individual pipeline steps
        - ``RUN_MODE`` — ``heatmap`` | ``impedance`` (mutually exclusive). Drives PEB kind
          (PI-Distribution vs PI-Spectrum), simulate wait pattern, move rename
          (Heatmap_real_ vs Imp_Real), compare, and the *separate* HTML report.
        - ``COMPONENTS`` — CreatePISpectrum IC port(s) used in impedance-mode PEB
        - ``ECADSTAR_ERF_PATH``, ``ECADSTAR_EMC_OUTPUT_DIR`` — Windows simulation paths
    - Key symbols: ``verify_and_stage_peb_for_ecadstar``, ``step_generate``
"""
from __future__ import annotations
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path
setup_path()

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# --- Run mode: EXACTLY ONE of "heatmap" | "impedance" (mutually exclusive) ---
# The whole pipeline (generate → simulate → move → compare → report) follows this mode:
#   "heatmap"   → PEB = PI-Distribution; compare heatmaps; heatmap HTML report
#   "impedance" → PEB = PI-Spectrum;     compare impedance; impedance HTML report
#                 SWEEP / PI frequencies are ignored (decode uses PI_REF_MHZ only).
# The move step uses the same K{n}/ layout in impedance mode (no freq_*MHz/ subfolders).
RUN_MODE = "heatmap"  # "heatmap" | "impedance"

# --- Model / VAE generate (exp052: Pearson+grad heatmap, unbounded per-MHz z) ---
EXPERIMENT_DIR = "experiments/exp057_structured_graph"
CHECKPOINT_PATH = f"{EXPERIMENT_DIR}/checkpoints/last_model.pt"
DATA_DIR = "datasets/data_multifreq_train_norm_unbounded"  

# PI sweep: explicit MHz list, or "anchors" | "dense" (heatmap mode only; ignored when RUN_MODE="impedance")
SWEEP: list[float] | str = [10, 63, 150, 270, 450,500]
DENSE_N_POINTS = 24

K_VALUE: int | list[int] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # min K=2 in unbounded dataset
OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep"  # overwritten by _sync_output_root() from K_VALUE
NUM_SAMPLES = 1
SHARED_TEMP = 1.0
SEED = 42

# PEB (1 PI per sample in ECADStar; both run modes emit a single PI kind)
HEATMAP_ONLY_PEB = True
POWERBUS = "Power_GND"
COMPONENTS = "IC1_Port1"  # CreatePISpectrum IC port(s) used when RUN_MODE == "impedance"
FORCE_CPU = False
BACKGROUND_MARGIN = 0.5

# Inference (QC modes — see scrap/generation/sweep_latent_opt_rules.py)
QC_SWEEP = True  # enforce fixed-layout QC rules (required for layout_qc / latent_z)
INFERENCE_MODE = "layout_qc"  # layout_qc | latent_z | layout_hybrid
LAYOUT_SOURCE = "val"  # val | latent_run | npy
LATENT_RUN_DIR: str | None = None  # e.g. data/latent_runs/exp050/0
EXPLICIT_Z_NPY: str | None = None
EXPLICIT_OCC_NPY: str | None = None
EXPLICIT_IMP_NPY: str | None = None
ALLOW_RANDOM_LAYOUT = False

RUN_SWEEP_QC_EVAL = True

PI_REF_MHZ = 200.0
CALIBRATE_FG_MAX = False  # exp052 unbounded — use true denorm levels in QC/sweep (no anchor fg_max rescale)
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
ECADSTAR_CLOSE_AFTER_BATCH = False
# Delete stale .rlk lock before opening (previous crash / force-close)
ECADSTAR_CLEAR_LOCK_FILE = True

# --- Move PI outputs (after ECADStar) ---
SOURCE_EMC_DIR = ECADSTAR_EMC_OUTPUT_DIR
MOVE = True
CLEAN_DEST_BEFORE_PASTE = True

# --- Pipeline step control (True = skip that step) ---
SKIP_GENERATE = False
SKIP_SIMULATE = False
SKIP_MOVE =  False
SKIP_COMPARE = False
SKIP_REPORT = False
SALVAGE_MISPLACED = False

# True → skip generate/simulate/move/compare plots/report; only write sim_compare_metrics.*
METRICS_ONLY = False

# --- Sim compare reporting (CAD sweep QC) ---
# Primary metrics for MD tables / agent summaries: spatial shape + peak scale.
# Do NOT lead with mae_ohm (often 0 at 10 MHz) or mape_pct (near-zero divide).
# Full Ω-error columns remain in sim_compare_metrics.csv for spreadsheets.
SIM_METRICS_PRIMARY_COLUMNS = (
    "mhz",
    "k",
    "sample",
    "pearson_r",
    "max_diff_ohm",
    "pattern_mae",
    "max_ratio",
    "peak_loc_err_px",
    "real_max_ohm",
    "gen_max_ohm",
)

# --- Comparison / report (derived from RUN_MODE; mutually exclusive) ---
# Each mode produces its own PNGs and a *separate* HTML report:
#   heatmap  → generated_vs_real_heatmap.png            + comparison_report_heatmap_*.html
#   impedance→ generated_vs_real_impedance_profile.png  + comparison_report_impedance_*.html
if str(RUN_MODE).strip().lower() not in ("heatmap", "impedance"):
    raise SystemExit(f"RUN_MODE must be 'heatmap' or 'impedance', got {RUN_MODE!r}")
RUN_HEATMAP_COMPARE = str(RUN_MODE).strip().lower() == "heatmap"
RUN_IMPEDANCE_COMPARE = str(RUN_MODE).strip().lower() == "impedance"

# --- Heatmap compare (Real vs Generated panels) ---
# Per-panel color scale uses foreground max (not percentile).
HEATMAP_LEVELS = 35
HEATMAP_VMAX_MODE = "max"
HEATMAP_COLORBAR_TICKS = 11
HEATMAP_SHARED_COLOR_SCALE = False  # gen_vmax = real_vmax
HEATMAP_REAL_VMAX_MATCH_GENERATED = False  # real_vmax = gen_vmax

# =============================================================================


def _k_values() -> list[int]:
    from scrap.generation.run_multifreq_heatmap_sweep import exported_k_values, normalize_k_values

    _sync_output_root()
    manifest_root = _PROJECT_ROOT / OUTPUT_ROOT
    if (manifest_root / "sweep_frequencies_mhz.json").is_file():
        return exported_k_values(manifest_root)
    return normalize_k_values(K_VALUE)


def _k_tag() -> str:
    from scrap.generation.run_multifreq_heatmap_sweep import k_output_tag

    return k_output_tag(K_VALUE)


def _sync_output_root() -> None:
    """Keep OUTPUT_ROOT aligned with K_VALUE."""
    global OUTPUT_ROOT
    OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep_{_k_tag()}"


def _peb_out_file() -> str:
    from scrap.generation.run_multifreq_heatmap_sweep import peb_basename_for_k

    return f"{OUTPUT_ROOT}/{peb_basename_for_k(K_VALUE)}"


def _expected_peb_name() -> str:
    from scrap.generation.run_multifreq_heatmap_sweep import peb_basename_for_k

    return peb_basename_for_k(K_VALUE)


def _peb_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_stage_peb_for_ecadstar() -> Path:
    """Ensure ECADStar loads the same PEB that generate wrote (name + content).

    AHK pastes the **full Windows path** in Load Batch (see tools/ecadstar/ecadstar_piemi_batch.ahk).
    We still copy the generated PEB to PEB_COPY_DEST and beside the .erf for convenience.
    """
    expected_name = _expected_peb_name()
    repo_peb = (_PROJECT_ROOT / _peb_out_file()).resolve()

    if not repo_peb.is_file():
        raise SystemExit(
            f"Generated PEB not found: {repo_peb}\n"
            f"  Expected name: {expected_name} (K={_k_tag()})\n"
            "  Run generate step first (SKIP_GENERATE = False)."
        )
    if repo_peb.name != expected_name:
        raise SystemExit(
            f"PEB filename mismatch: {repo_peb.name!r} != {expected_name!r} (K={_k_tag()})"
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
        for other in folder.glob("pi_*_K*_freq_sweep.peb"):
            if other.name == expected_name:
                continue
            print(f"    Note: other PEB in folder: {other.name} (ECADStar will load {expected_name!r})")

    simulate_peb = staged[0] if PEB_COPY_DEST else repo_peb
    peb_win_path = _windows_path_str(simulate_peb)
    print(f"    AHK pastes: {peb_win_path!r}  (full path in Load Batch dialog)")
    print(f"    -PebPath  : {peb_win_path}")
    return simulate_peb


def _parse_pi_number(name: str) -> int | None:
    m = re.match(r"^PI-(\d+)(?:\..+)?$", name)
    return int(m.group(1)) if m else None


def _peb_group_count(peb_path: Path) -> int | None:
    """Count ``<Group>`` entries in a PEB (authoritative expected PI count)."""
    if not peb_path.is_file():
        return None
    try:
        text = peb_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    n = text.count("<Group>")
    return n if n > 0 else None


def _expected_pi_count(freq_list: list[int] | None) -> int:
    """Expected PI folders = PEB group count, else effective K × freq × samples."""
    peb = (_PROJECT_ROOT / _peb_out_file()).resolve()
    n_peb = _peb_group_count(peb)
    if n_peb is not None:
        return n_peb

    pps = 1 if HEATMAP_ONLY_PEB else 2
    if str(RUN_MODE).strip().lower() == "impedance":
        return len(_k_values()) * NUM_SAMPLES * pps
    return len(freq_list or []) * len(_k_values()) * NUM_SAMPLES * pps


def _pi_folder(emc_dir: Path, pi_num: int) -> Path | None:
    for item in emc_dir.iterdir():
        if _parse_pi_number(item.name) == pi_num:
            return item
    return None


def _pi_ready_patterns() -> tuple[str, ...]:
    """Files that signal a finished PI output for the active run mode.

    heatmap → ``Z_*MHz.map``; impedance (PI-Spectrum) → ``*PIPinZ*.csv`` / ``*.csv``.
    """
    if str(RUN_MODE).strip().lower() == "impedance":
        return ("*PIPinZ*.csv", "*.csv")
    return ("Z_*MHz.map",)


def _pi_map_mtime(pi_dir: Path) -> float | None:
    files: list[Path] = []
    for pattern in _pi_ready_patterns():
        files = list(pi_dir.rglob(pattern))
        if files:
            break
    if not files:
        return None
    return max(f.stat().st_mtime for f in files)


def _pi_ready_after(emc_dir: Path, pi_num: int, since_ts: float) -> bool:
    folder = _pi_folder(emc_dir, pi_num)
    if folder is None or not folder.is_dir():
        return False
    mtime = _pi_map_mtime(folder)
    return mtime is not None and mtime >= since_ts - 2.0


def wait_for_pi_outputs(freq_list: list[int], *, batch_start_ts: float) -> None:
    """Poll EMC until PI-1..PI-N have fresh output files (batch simulation finished)."""
    if not ECADSTAR_WAIT_FOR_PI:
        return
    emc = _resolve_windows_path(SOURCE_EMC_DIR)
    need = list(range(1, _expected_pi_count(freq_list) + 1))
    peb_n = _peb_group_count((_PROJECT_ROOT / _peb_out_file()).resolve())
    from scrap.generation.run_multifreq_heatmap_sweep import normalize_k_values

    cfg_n = len(freq_list or []) * len(normalize_k_values(K_VALUE)) * NUM_SAMPLES
    if peb_n is not None and peb_n != cfg_n:
        print(
            f"  PI count from PEB: {peb_n} "
            f"(config would expect {cfg_n}; using PEB as source of truth)"
        )
    print(f"\nWaiting for {len(need)} PI output(s) in {emc} (timeout {ECADSTAR_WAIT_TIMEOUT_SEC}s)…")
    deadline = time.time() + ECADSTAR_WAIT_TIMEOUT_SEC
    last_ready = -1
    while time.time() < deadline:
        ready = [n for n in need if _pi_ready_after(emc, n, batch_start_ts)]
        if len(ready) >= len(need):
            print(f"✓ All {len(need)} PI outputs present with fresh files.")
            return
        if len(ready) != last_ready:
            missing = [n for n in need if n not in ready]
            preview = ", ".join(str(n) for n in missing[:12])
            suffix = f" … +{len(missing) - 12} more" if len(missing) > 12 else ""
            print(
                f"  … {len(ready)}/{len(need)} PI outputs ready "
                f"(missing: {preview}{suffix}) — retry in {ECADSTAR_WAIT_POLL_SEC}s"
            )
            last_ready = len(ready)
        else:
            print(
                f"  … {len(ready)}/{len(need)} PI outputs ready "
                f"— retry in {ECADSTAR_WAIT_POLL_SEC}s"
            )
        time.sleep(ECADSTAR_WAIT_POLL_SEC)
    ready = [n for n in need if _pi_ready_after(emc, n, batch_start_ts)]
    missing = [n for n in need if n not in ready]
    raise SystemExit(
        f"Timeout waiting for PI outputs in {emc}\n"
        f"  Ready: {len(ready)}/{len(need)}\n"
        f"  Missing PI numbers: {missing}\n"
        "  Increase ECADSTAR_WAIT_TIMEOUT_SEC or re-run with SKIP_GENERATE and SKIP_SIMULATE "
        "after batch finishes."
    )


def _ecadstar_lock_path() -> Path:
    erf = _resolve_windows_path(ECADSTAR_ERF_PATH)
    return erf.parent / f"{erf.stem}.rlk"


def clear_ecadstar_lock() -> bool:
    """Remove stale eCADSTAR design lock (.rlk) left after a forced close."""
    if not ECADSTAR_CLEAR_LOCK_FILE:
        return False
    lock = _ecadstar_lock_path()
    if not lock.is_file():
        return False
    lock.unlink()
    print(f"  Removed ECADStar lock: {_windows_path_str(lock)}")
    return True


def close_ecadstar_piemi() -> None:
    """Close eCADSTAR PI/EMI after batch completes without leaving a stale .rlk lock."""
    if not ECADSTAR_CLOSE_AFTER_BATCH:
        return

    ps_script = _PROJECT_ROOT / "tools" / "ecadstar" / "ecadstar_piemi_close.ps1"
    print("Closing eCADSTAR PI/EMI window (graceful, then clear lock)…")
    if ps_script.is_file():
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _windows_path_str(ps_script),
            "-ErfPath",
            ECADSTAR_ERF_PATH.replace("/", "\\"),
            "-GraceSec",
            str(ECADSTAR_CLOSE_GRACE_SEC), # type: ignore
            "-ForceIfNeeded",
        ]
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    else:
        # Fallback: old behavior + manual lock cleanup
        ps = (
            "Get-Process | Where-Object { $_.MainWindowTitle -like '*PI/EMI*' } "
            "| Stop-Process -Force -ErrorAction SilentlyContinue"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            cwd=str(_PROJECT_ROOT),
        )
        clear_ecadstar_lock()


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
    mod.RUN_MODE = RUN_MODE
    mod.HEATMAP_ONLY_PEB = HEATMAP_ONLY_PEB
    mod.PEB_OUT_FILE = _peb_out_file()
    mod.POWERBUS = POWERBUS
    mod.COMPONENTS = COMPONENTS
    mod.PEB_COPY_DEST = PEB_COPY_DEST
    mod.REPORT_COPY_DEST = REPORT_COPY_DEST
    mod.FORCE_CPU = FORCE_CPU
    mod.BACKGROUND_MARGIN = BACKGROUND_MARGIN
    mod.INFERENCE_MODE = INFERENCE_MODE
    mod.QC_SWEEP = QC_SWEEP
    mod.LAYOUT_SOURCE = LAYOUT_SOURCE
    mod.LATENT_RUN_DIR = LATENT_RUN_DIR
    mod.EXPLICIT_Z_NPY = EXPLICIT_Z_NPY
    mod.EXPLICIT_OCC_NPY = EXPLICIT_OCC_NPY
    mod.EXPLICIT_IMP_NPY = EXPLICIT_IMP_NPY
    mod.ALLOW_RANDOM_LAYOUT = ALLOW_RANDOM_LAYOUT
    mod.RUN_SWEEP_QC_EVAL = RUN_SWEEP_QC_EVAL
    mod.PI_REF_MHZ = PI_REF_MHZ
    mod.CALIBRATE_FG_MAX = CALIBRATE_FG_MAX
    mod.INFERENCE_FACTORIZED_ONLY = INFERENCE_FACTORIZED_ONLY
    mod.MODE = "generate"
    mod.VAEInference = mod._resolve_vae_inference()


def _apply_move_config(mod) -> None:
    mod.SOURCE_EMC_DIR = SOURCE_EMC_DIR
    mod.MOVE = MOVE
    mod.CLEAN_DEST_BEFORE_PASTE = CLEAN_DEST_BEFORE_PASTE


def _apply_compare_config(mc) -> None:
    """Push heatmap compare visualization settings into compare.py."""
    import scrap.comparison.heatmap_sim_metrics as hsm  # noqa: E402

    hsm.PRIMARY_METRIC_COLUMNS = SIM_METRICS_PRIMARY_COLUMNS
    mc.set_compare_overrides(
        RUN_HEATMAP=RUN_HEATMAP_COMPARE,
        RUN_IMPEDANCE=RUN_IMPEDANCE_COMPARE,
        HEATMAP_LEVELS=HEATMAP_LEVELS,
        HEATMAP_VMAX_MODE=HEATMAP_VMAX_MODE,
        HEATMAP_COLORBAR_TICKS=HEATMAP_COLORBAR_TICKS,
        HEATMAP_SHARED_COLOR_SCALE=HEATMAP_SHARED_COLOR_SCALE,
        HEATMAP_REAL_VMAX_MATCH_GENERATED=HEATMAP_REAL_VMAX_MATCH_GENERATED,
        WRITE_SIM_METRICS=True,
        METRICS_ONLY=METRICS_ONLY,
    )


def _apply_all_config() -> None:
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402
    import scrap.orchestration.multifreq_move_and_compare as mc  # noqa: E402

    _apply_sweep_config(sweep)
    _apply_move_config(mc)
    mc.METRICS_ONLY = METRICS_ONLY
    mc.RUN_MODE = RUN_MODE
    mc.RUN_HEATMAP_COMPARE = RUN_HEATMAP_COMPARE
    mc.RUN_IMPEDANCE_COMPARE = RUN_IMPEDANCE_COMPARE
    _apply_compare_config(mc)


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

    peb_win_path = _windows_path_str(peb_path)
    expected_name = _expected_peb_name()
    if peb_path.name != expected_name:
        raise SystemExit(
            f"ECADStar PEB name mismatch: will paste {peb_path.name!r}, "
            f"expected generated {expected_name!r} (K={_k_tag()})"
        )

    ps_script = groot / "tools" / "ecadstar" / "run_ecadstar_piemi_batch.ps1"
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
    print(f"  Load Batch will paste: {peb_win_path!r} (full path to generated PEB)")
    print(f"  Log (Windows): {log_hint}")
    proc = subprocess.run(cmd, cwd=str(groot), env=os.environ.copy())
    return int(proc.returncode)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_generate() -> None:
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    _sync_output_root()
    _apply_all_config()

    print("\n[1/3] Generate + PEB")
    print(f"  {CHECKPOINT_PATH}  →  {OUTPUT_ROOT}")

    sweep.run_from_config()


def step_simulate() -> None:
    print("\n[2/3] ECADStar simulate")
    _sync_output_root()
    _apply_all_config()
    clear_ecadstar_lock()
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

    if isinstance(SWEEP, list) and not RUN_IMPEDANCE_COMPARE:
        freq_list = [int(round(m)) for m in SWEEP]
    else:
        import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

        freq_list = sweep.exported_freq_mhz_list()
    wait_for_pi_outputs(freq_list if not RUN_IMPEDANCE_COMPARE else [], batch_start_ts=batch_start)


def step_sim_metrics_only() -> None:
    """Write sim_compare_metrics.* only (gen vs ECADStar Real/ — no plots)."""
    import scrap.orchestration.multifreq_move_and_compare as mc  # noqa: E402

    _sync_output_root()
    _apply_all_config()
    os.chdir(_PROJECT_ROOT)

    print("\n" + "=" * 72)
    print("METRICS ONLY — simulated real vs generated heatmap evaluation")
    print("=" * 72)
    json_path = mc.run_sim_metrics_only()
    print(f"\nMetrics written: {json_path}")


def step_move_compare_report() -> None:
    import scrap.orchestration.multifreq_move_and_compare as mc  # noqa: E402
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    _sync_output_root()
    _apply_all_config()
    os.chdir(_PROJECT_ROOT)

    mc.SOURCE_EMC_DIR_OVERRIDE = SOURCE_EMC_DIR

    freq_override: list[int] | None = None
    if RUN_IMPEDANCE_COMPARE:
        freq_override = []
    elif isinstance(SWEEP, list):
        freq_override = [int(round(m)) for m in SWEEP]
        sweep.write_sweep_freq_manifest([float(m) for m in freq_override])

    print("\n[3/3] Move + compare + report")

    if SALVAGE_MISPLACED:
        fl = freq_override if freq_override is not None else mc.exported_freq_mhz_list()
        ks = _k_values()
        print(f"\n=== Salvage misplaced PI outputs ({len(fl)} freq(s), K={ks[0]}–{ks[-1]}) ===")
        mc.salvage_pi_by_global_index(
            freq_list=fl,
            base_generated_dir=_PROJECT_ROOT / OUTPUT_ROOT,
            k_values=ks,
            num_samples=NUM_SAMPLES,
        )
    elif not SKIP_MOVE:
        mc.run_move(freq_override)
    else:
        print("Skipping move step.")

    if not SKIP_COMPARE:
        mc.run_compare()
    else:
        print("Skipping compare step.")

    if not SKIP_REPORT:
        mc.run_report()
    else:
        print("Skipping report step.")


def _print_config_summary() -> None:
    sweep_desc = (
        ", ".join(str(int(m)) if m == round(m) else f"{m:g}" for m in SWEEP)
        if isinstance(SWEEP, list)
        else str(SWEEP)
    )
    print("Config:")
    print(f"  {EXPERIMENT_DIR}  |  {Path(CHECKPOINT_PATH).name}")
    print(f"  out={OUTPUT_ROOT}  K={_k_tag()}  n={NUM_SAMPLES}", end="")
    if RUN_IMPEDANCE_COMPARE:
        print(f"  decode@{PI_REF_MHZ:g}MHz  (SWEEP ignored)")
    else:
        print(f"  MHz=[{sweep_desc}]")
    print(f"  inference={INFERENCE_MODE}/{LAYOUT_SOURCE}  pi_ref={PI_REF_MHZ} MHz")
    peb_kind = "PI-Spectrum" if RUN_IMPEDANCE_COMPARE else "PI-Distribution"
    print(f"  run_mode={str(RUN_MODE).strip().lower()}  PEB={peb_kind}  (compare + report follow mode)")
    if METRICS_ONLY:
        print("  mode=METRICS_ONLY")


def main() -> None:
    os.chdir(_PROJECT_ROOT)
    _sync_output_root()
    print("Loading modules…")
    _apply_all_config()
    _print_config_summary()

    if METRICS_ONLY:
        step_sim_metrics_only()
        print("\n" + "=" * 72)
        print("✓ Metrics-only evaluation complete")
        print("=" * 72)
        return

    if not SKIP_GENERATE:
        step_generate()
    else:
        print("\nSkipping generate step.")

    if not SKIP_SIMULATE:
        step_simulate()
    else:
        print("\nSkipping ECADStar simulate step.")

    step_move_compare_report()

    print("\n" + "=" * 72)
    print("✓ Pipeline complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
