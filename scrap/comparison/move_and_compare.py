"""Move PI Outputs and Compare (run_all_k).

Run: python scrap/comparison/move_and_compare.py"""
from __future__ import annotations

import os
import re
import shutil
import sys
import base64
import traceback
from datetime import datetime
from bisect import bisect_right
from pathlib import Path, PureWindowsPath
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator

# -- Bootstrap project root --------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -- Import run parameters from run_all_k ------------------------------------
from scrap.generation.run_all_k import (  # noqa: E402
    K_MIN,
    K_MAX,
    NUM_SAMPLES as _RUN_NUM_SAMPLES,
    OUTPUT_ROOT as _OUTPUT_ROOT,
    PI_FREQ_MHZ as _RUN_PI_FREQ_MHZ,
)

BASE_GENERATED_DIR = _OUTPUT_ROOT

# ============================================================
# CONFIGURATION — edit these before running
# ============================================================
SOURCE_EMC_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"

# Override which frequencies to process (None = use run_all_k list)
PI_FREQ_OVERRIDE: int | list[int] | None = None

# -- Move options ------------------------------------------------------------
MOVE                       = True   # True = cut/move, False = copy
OVERWRITE                  = False  # False = auto-rename on collision
DRY_RUN                    = False
CLEAN_DEST_BEFORE_PASTE    = True   # wipe Real/ before pasting new PI-* items
SEARCH_RECURSIVE           = False  # search .emc dir recursively for PI-*
LIMIT_TO_EXPECTED_PI       = True   # ignore PI-* numbers outside expected range
RENAME_PI_TO_MATCH_SAMPLES = True   # rename PI-* -> Heatmap_real_* / Imp_Real*

# -- Compare options ---------------------------------------------------------
RUN_HEATMAP   = True
RUN_IMPEDANCE = True
RUN_OCCUPANCY = True
FAIL_FAST     = False  # if True, stop at first K failure

# -- Assets ------------------------------------------------------------------
FREQUENCY_PATH        = Path("configs/Frequency_data_hz.npy")
TARGET_IMPEDANCE_PATH = Path("configs/target_impedance.npy")
MASK_PATH             = Path("configs/binary_mask.npy")

# -- Output file names (written inside each K{n}/ folder) --------------------
HEATMAP_OUT_NAME   = "generated_vs_real_heatmap.png"
IMPEDANCE_OUT_NAME = "generated_vs_real_impedance_profile.png"
OCCUPANCY_OUT_NAME = "generated_occupancy.png"

# -- Heatmap plot settings ---------------------------------------------------
HEATMAP_CMAP                   = "jet"
HEATMAP_LEVELS                 = 22
HEATMAP_DIFF_TOLERANCE         = 0.25
HEATMAP_PATTERN_DIFF_TOLERANCE = 0.05
HEATMAP_VMAX_PERCENTILE_FG     = 99.9

# -- Real output file choosers -----------------------------------------------
MAP_GLOB_PREFERENCE           = ("Z_*.map", "*.map")
IMPEDANCE_CSV_GLOB_PREFERENCE = ("*PIPinZ*.csv", "*.csv")

# -- Occupancy settings ------------------------------------------------------
ACTIVE_POLICY = "topk"  # "topk" or "threshold"
THRESHOLD     = 0.5

# REPORT_ONLY = True  # skip move and compare; only rebuild HTML report
REPORT_ONLY = False
# =============================================================================

# Resolved frequency list from PI_FREQ_OVERRIDE or run_all_k
_freq_src = PI_FREQ_OVERRIDE if PI_FREQ_OVERRIDE is not None else _RUN_PI_FREQ_MHZ
if _freq_src is None:
    _FREQ_LIST: list[int | None] = [None]
elif isinstance(_freq_src, list):
    _FREQ_LIST = list(_freq_src)
else:
    _FREQ_LIST = [int(_freq_src)]

# Human-readable frequency label (updated per-frequency at runtime)
_FREQ_LABEL = ""

PI_NAME_REGEX = r"^PI-\d+(?:\..+)?$"


# ============================================================
# Shared helpers
# ============================================================

def _base_dir_for_freq(mhz: int | None) -> str:
    if mhz is None:
        return BASE_GENERATED_DIR
    return f"{BASE_GENERATED_DIR}/freq_{mhz}MHz"


def _parse_pi_number(name: str) -> int | None:
    m = re.match(r"^PI-(\d+)(?:\..+)?$", name)
    return int(m.group(1)) if m else None


def _suffix(name: str) -> str:
    return Path(name).suffix


def _infer_num_samples(k_dir: Path) -> int:
    sample_dirs = [p for p in k_dir.glob("data_sample_*") if p.is_dir()]
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* folders found in: {k_dir}")

    def _idx(p: Path) -> int | None:
        m = re.match(r"^data_sample_(\d+)$", p.name)
        return int(m.group(1)) if m else None

    indices = sorted(i for i in (_idx(p) for p in sample_dirs) if i is not None)
    if not indices:
        raise SystemExit(f"Found data_sample_* entries but none matched expected naming in: {k_dir}")
    expected = list(range(0, max(indices) + 1))
    if indices != expected:
        raise SystemExit(
            "data_sample_* folders are not consecutive starting at 0.\n"
            f"  Found:    {indices[:20]}{' ...' if len(indices) > 20 else ''}\n"
            f"  Expected: {expected[:20]}{' ...' if len(expected) > 20 else ''}"
        )
    return len(expected)


# ============================================================
# Move helpers
# ============================================================

_PIS_PER_SAMPLE_OVERRIDE: int | None = None
_PI_OUTPUT_KIND_OVERRIDE: str | None = None  # "heatmap" | "impedance"


def _pis_per_sample() -> int:
    """run_all_k always produces 2 PI outputs per sample (heatmap + impedance)."""
    return int(_PIS_PER_SAMPLE_OVERRIDE) if _PIS_PER_SAMPLE_OVERRIDE is not None else 2


def _single_pi_output_kind() -> str:
    if _PI_OUTPUT_KIND_OVERRIDE in ("heatmap", "impedance"):
        return _PI_OUTPUT_KIND_OVERRIDE
    return "impedance"


def _resolve_source_dir(path_str: str) -> Path:
    if re.match(r"^[A-Za-z]:\\", path_str):
        win = PureWindowsPath(path_str)
        drive = win.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*win.parts[1:])
        if wsl_path.exists():
            return wsl_path
        return Path(path_str)
    return Path(path_str)


def _count_pi_names(directory: Path, *, recursive: bool) -> int:
    pattern = re.compile(PI_NAME_REGEX)
    gen = directory.rglob("*") if recursive else directory.iterdir()
    return sum(1 for p in gen if pattern.match(p.name))


def resolve_pi_source_dir(path_str: str, *, recursive: bool = True) -> Path:
    root = _resolve_source_dir(path_str)
    if not root.exists():
        raise SystemExit(f"Source path does not exist:\n  {root}")
    if not root.is_dir():
        raise SystemExit(f"SOURCE path must be a directory: {root}")
    if _count_pi_names(root, recursive=False) > 0:
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _count_pi_names(child, recursive=recursive) > 0:
            print(f"Using PI subfolder: {child}")
            return child
    if recursive and _count_pi_names(root, recursive=True) > 0:
        print(f"Using recursive PI search under: {root}")
        return root
    sample = sorted(root.iterdir())[:15]
    listing = "\n  ".join(p.name for p in sample)
    raise SystemExit(
        "No PI-* items found under source path.\n"
        f"  Path: {root}\n"
        f"  Top-level entries:\n  {listing}\n\n"
        "Run ECADStar batch on the .peb first."
    )


def _iter_candidates(source_dir: Path):
    if SEARCH_RECURSIVE:
        yield from source_dir.rglob("*")
    else:
        yield from source_dir.iterdir()


def _unique_dest_path(dest_dir: Path, name: str) -> Path:
    base = dest_dir / name
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for i in range(1, 10_000):
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free destination name for {name}")


def _clean_dest(dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    if str(dest_resolved) in ("/", ""):
        raise SystemExit(f"Refusing to clean unsafe DEST_DIR: {dest_dir}")
    for child in dest_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _do_move_item(item: Path, target: Path) -> None:
    if MOVE:
        if OVERWRITE and target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
    else:
        if item.is_dir():
            if OVERWRITE and target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if OVERWRITE and target.exists():
                target.unlink()
            shutil.copy2(item, target)


def _rename_pi_outputs_with_offset(dest_dir: Path, *, num_samples: int, sample_offset: int) -> None:
    by_num: dict[int, list[Path]] = {}
    for item in dest_dir.iterdir():
        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue
        by_num.setdefault(pi_num, []).append(item)

    pps = _pis_per_sample()
    expected: list[int] = []
    for i in range(num_samples):
        j = sample_offset + i
        if pps == 1:
            expected.append(j + 1)
        else:
            expected.extend([2 * j + 1, 2 * j + 2])

    missing = [n for n in expected if n not in by_num]
    if missing:
        raise SystemExit(
            "Missing expected PI outputs in DEST_DIR.\n"
            f"  Missing PI numbers: {missing[:20]}{' ...' if len(missing) > 20 else ''}\n"
            f"  DEST_DIR: {dest_dir}"
        )
    for i in range(num_samples):
        j = sample_offset + i
        if pps == 1:
            prefix = "Heatmap_real_" if _single_pi_output_kind() == "heatmap" else "Imp_Real"
            for item in sorted(by_num[j + 1], key=lambda p: p.name):
                new_name = f"{prefix}{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
        else:
            pi_heat = 2 * j + 1
            pi_imp  = 2 * j + 2
            for item in sorted(by_num[pi_heat], key=lambda p: p.name):
                new_name = f"Heatmap_real_{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)
            for item in sorted(by_num[pi_imp], key=lambda p: p.name):
                new_name = f"Imp_Real{i}{_suffix(item.name)}"
                target = dest_dir / new_name
                if target.exists():
                    raise SystemExit(f"Target already exists while renaming: {target}")
                item.rename(target)


def _fallback_rename_pi_to_impedance(dest_dir: Path, num_samples: int) -> None:
    if any(dest_dir.glob("Imp_Real*")):
        return
    pi_items = sorted(
        (p for p in dest_dir.iterdir() if _parse_pi_number(p.name) is not None),
        key=lambda p: _parse_pi_number(p.name) or 0,
    )
    for i, src in enumerate(pi_items[:num_samples]):
        dst = dest_dir / f"Imp_Real{i}{_suffix(src.name)}"
        if dst.exists():
            continue
        src.rename(dst)
        print(f"  fallback rename: {src.name} -> {dst.name}")


def move_pi_outputs_for_k_range(
    k_min: int,
    k_max: int,
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
) -> list[Path]:
    """Split PI-* outputs across K{k_min}..K{k_max} for a single-frequency run."""
    if not (0 <= k_min <= 52 and 0 <= k_max <= 52 and k_min <= k_max):
        raise SystemExit("Expected 0 <= k_min <= k_max <= 52")

    ks = list(range(k_min, k_max + 1))
    source_dir = resolve_pi_source_dir(source_emc_dir, recursive=SEARCH_RECURSIVE)
    base_dir = Path(base_generated_dir)
    per_k_num_samples: list[int] = []
    per_k_dest: list[Path] = []

    for k in ks:
        generated_k_dir = base_dir / f"K{k}"
        if not generated_k_dir.exists() or not generated_k_dir.is_dir():
            raise SystemExit(f"Generated K directory not found: {generated_k_dir}")
        per_k_num_samples.append(_infer_num_samples(generated_k_dir))
        dest_dir = generated_k_dir / "Real"
        if dest_dir.exists() and not dest_dir.is_dir():
            raise SystemExit(f"DEST_DIR exists but is not a directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        per_k_dest.append(dest_dir)

    if CLEAN_DEST_BEFORE_PASTE and not DRY_RUN:
        for dest_dir in per_k_dest:
            _clean_dest(dest_dir)

    pps = _pis_per_sample()
    cumulative = [0]
    for n in per_k_num_samples:
        cumulative.append(cumulative[-1] + n)

    pattern = re.compile(PI_NAME_REGEX)
    moved = 0

    for item in _iter_candidates(source_dir):
        if not item.exists():
            continue
        if not pattern.match(item.name):
            continue
        pi_num = _parse_pi_number(item.name)
        if pi_num is None:
            continue

        sample_index = (pi_num - 1) // pps
        idx = bisect_right(cumulative, sample_index) - 1
        if idx < 0 or idx >= len(ks):
            continue

        block_start = cumulative[idx]
        local_i = sample_index - block_start
        if not (0 <= local_i < per_k_num_samples[idx]):
            continue

        if LIMIT_TO_EXPECTED_PI:
            start_pi = pps * block_start + 1
            end_pi   = pps * (block_start + per_k_num_samples[idx])
            if not (start_pi <= pi_num <= end_pi):
                continue

        dest_dir = per_k_dest[idx]
        target = dest_dir / item.name
        if target.exists() and not OVERWRITE:
            target = _unique_dest_path(dest_dir, item.name)

        print(f"{'MOVE' if MOVE else 'COPY'}: {item} -> {target}   (K={ks[idx]}, sample={local_i})")
        if not DRY_RUN:
            _do_move_item(item, target)
        moved += 1

    if moved == 0:
        print(f"\n!  No PI-* files found in source -- skipping rename for K={ks}.")
    elif RENAME_PI_TO_MATCH_SAMPLES and not DRY_RUN:
        print(f"\nRenaming PI-* in Real/ (pps={pps}) ...")
        for idx, dest_dir in enumerate(per_k_dest):
            try:
                _rename_pi_outputs_with_offset(
                    dest_dir,
                    num_samples=per_k_num_samples[idx],
                    sample_offset=cumulative[idx],
                )
            except SystemExit as e:
                print(f"  !  K{ks[idx]} rename: {e}")
                _fallback_rename_pi_to_impedance(dest_dir, per_k_num_samples[idx])

    print(f"\n+ Moved {moved} item(s) across {len(per_k_dest)} Real/ folders (K{k_min}..K{k_max}).")
    return per_k_dest


def move_pi_outputs_multi_freq(
    *,
    source_emc_dir: str = SOURCE_EMC_DIR,
    freq_list: list[int],
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    num_samples: int = _RUN_NUM_SAMPLES,
    base_generated_dir: str | Path = BASE_GENERATED_DIR,
) -> None:
    """Move PI-* outputs when the PEB was generated with multiple PI frequencies.

    PEB order mirrors run_all_k.py: outer loop = frequency, inner loop = K.
    e.g. freqs=[10,100,200], K=10..11, num_samples=1:
      PI-1,2  -> freq_10MHz/K10/Real
      PI-3,4  -> freq_10MHz/K11/Real
      PI-5,6  -> freq_100MHz/K10/Real  ...
    """
    source_dir = _resolve_source_dir(source_emc_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source .emc directory does not exist:\n  {source_dir}")

    base_dir = Path(base_generated_dir)
    ks = list(range(k_min, k_max + 1))
    pps = _pis_per_sample()

    slots: list[tuple[str, int, int, int]] = []  # (freq_tag, k, local_i, global_idx)
    global_idx = 0
    for mhz in freq_list:
        freq_tag = f"freq_{mhz}MHz"
        for k in ks:
            for local_i in range(num_samples):
                slots.append((freq_tag, k, local_i, global_idx))
                global_idx += 1

    pi_to_dest: dict[int, tuple[Path, int]] = {}
    dest_dirs_seen: set[Path] = set()
    for freq_tag, k, local_i, g_idx in slots:
        dest_dir = base_dir / freq_tag / f"K{k}" / "Real"
        pi_nums = [2 * g_idx + 1, 2 * g_idx + 2] if pps == 2 else [g_idx + 1]
        for pi_num in pi_nums:
            pi_to_dest[pi_num] = (dest_dir, local_i)
        dest_dirs_seen.add(dest_dir)

    for d in dest_dirs_seen:
        d.mkdir(parents=True, exist_ok=True)

    if CLEAN_DEST_BEFORE_PASTE and not DRY_RUN:
        for d in dest_dirs_seen:
            _clean_dest(d)

    pattern = re.compile(PI_NAME_REGEX)
    moved = 0

    for item in _iter_candidates(source_dir):
        if not item.exists():
            continue
        if not pattern.match(item.name):
            continue
        pi_num = _parse_pi_number(item.name)
        if pi_num is None or pi_num not in pi_to_dest:
            continue

        dest_dir, local_i = pi_to_dest[pi_num]
        target = dest_dir / item.name
        if target.exists() and not OVERWRITE:
            target = _unique_dest_path(dest_dir, item.name)

        freq_tag = target.parent.parent.parent.name
        k_tag    = target.parent.parent.name
        print(f"{'MOVE' if MOVE else 'COPY'}: {item} -> {target}   ({freq_tag}/{k_tag}, sample={local_i})")
        if not DRY_RUN:
            _do_move_item(item, target)
        moved += 1

    if moved == 0:
        print("\n!  No PI-* files found in source -- skipping rename.")
    elif RENAME_PI_TO_MATCH_SAMPLES and not DRY_RUN:
        dir_to_slots: dict[Path, list[tuple[int, int]]] = {}
        for freq_tag, k, local_i, g_idx in slots:
            d = base_dir / freq_tag / f"K{k}" / "Real"
            dir_to_slots.setdefault(d, []).append((local_i, g_idx))
        for dest_dir, slot_list in dir_to_slots.items():
            min_g_idx = min(g_idx for _, g_idx in slot_list)
            n = len(set(local_i for local_i, _ in slot_list))
            _rename_pi_outputs_with_offset(dest_dir, num_samples=n, sample_offset=min_g_idx)

    print(
        f"\n+ Moved {moved} item(s) across {len(dest_dirs_seen)} Real/ folders"
        f" ({len(freq_list)} freq(s), K{k_min}..K{k_max})."
    )


# ============================================================
# Compare helpers
# ============================================================

def _load_generated_heatmap(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected heatmap to be 2D, got shape {data.shape} from {path}")
    return data


def _load_map_file(file_path: Path, *, resolution: int) -> np.ndarray:
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "3":
            try:
                for k in range(1, 4):
                    px, py, pz = map(float, lines[i + k].split())
                    x.append(px); y.append(py); z.append(max(pz, 0.01))
                i += 3
            except Exception:
                pass
        i += 1
    if not x:
        raise RuntimeError(f"No data loaded from MAP file: {file_path}")
    x_np = np.asarray(x); y_np = np.asarray(y); z_np = np.asarray(z)
    scale = 1e-5
    x_np = (x_np - x_np.min()) * scale
    y_np = (y_np - y_np.min()) * scale
    xi = np.linspace(x_np.min(), x_np.max(), resolution)
    yi = np.linspace(y_np.min(), y_np.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    points = np.column_stack((x_np, y_np))
    Zi = griddata(points, z_np, (Xi, Yi), method="linear")
    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        rbf = RBFInterpolator(points, z_np, smoothing=0.15)
        Zi[nan_mask] = rbf(np.column_stack((Xi[nan_mask], Yi[nan_mask])))
    return Zi


def _choose_real_heatmap_mapfile(item: Path) -> Path:
    if item.is_file() and item.suffix.lower() == ".map":
        return item
    if item.is_dir():
        for pattern in MAP_GLOB_PREFERENCE:
            candidates = sorted(item.rglob(pattern))
            if candidates:
                return candidates[0]
    raise FileNotFoundError(f"Could not locate a .map file under: {item}")


def _real_impedance_candidates(real_dir: Path, sample_i: int) -> list[Path]:
    imp = sorted(real_dir.glob(f"Imp_Real{sample_i}*"))
    if imp:
        return imp
    pi_items = sorted(
        (p for p in real_dir.iterdir() if _parse_pi_number(p.name) is not None),
        key=lambda p: _parse_pi_number(p.name) or 0,
    )
    if not pi_items:
        return []
    if len(pi_items) == 1:
        return [pi_items[0]]
    if sample_i < len(pi_items):
        return [pi_items[sample_i]]
    return []


def _choose_real_impedance_csv(item: Path) -> Path:
    if item.is_file() and item.suffix.lower() == ".csv":
        return item
    if item.is_dir():
        for pattern in IMPEDANCE_CSV_GLOB_PREFERENCE:
            candidates = sorted(item.rglob(pattern))
            if candidates:
                return candidates[0]
    raise FileNotFoundError(f"Could not locate a .csv file under: {item}")


def _iter_numeric_rows(path: Path) -> Iterable[tuple[float, float]]:
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ").replace(";", " ")
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                yield x, y


def _load_real_impedance(csv_path: Path, *, frequency_hz: np.ndarray) -> np.ndarray:
    rows = list(_iter_numeric_rows(csv_path))
    if not rows:
        raise RuntimeError(f"No numeric data found in: {csv_path}")
    x = np.asarray([r[0] for r in rows], dtype=float)
    z = np.asarray([r[1] for r in rows], dtype=float)
    if len(z) == len(frequency_hz):
        return z
    order = np.argsort(x)
    x_s = x[order]; z_s = z[order]
    good = (x_s > 0) & (z_s > 0)
    x_s = x_s[good]; z_s = z_s[good]
    if len(x_s) < 10:
        return z
    fx = np.log10(frequency_hz.astype(float))
    return 10 ** np.interp(fx, np.log10(x_s.astype(float)), np.log10(z_s.astype(float)))


def _load_generated_impedance_log(path: Path) -> np.ndarray:
    data = np.load(path).squeeze()
    return data.reshape(-1) if data.ndim != 1 else data


def _maybe_load(path: Path) -> np.ndarray | None:
    return np.load(path).squeeze() if path.exists() else None


def _plot_heatmap_comparisons(
    *, comparisons: list[dict], mask: np.ndarray, out_path: Path
) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = np.array([axes])
    cmap = mpl.colormaps[HEATMAP_CMAP].resampled(HEATMAP_LEVELS).copy()
    cmap.set_bad("white")

    for row_idx, item in enumerate(comparisons):
        real = item["real"]; gen = item["generated"]; label = item["label"]
        real_m = np.ma.masked_where(~mask, real)
        gen_m  = np.ma.masked_where(~mask, gen)
        diff   = np.where(np.abs(real - gen) < HEATMAP_DIFF_TOLERANCE, 0.0, np.abs(real - gen))
        diff_m = np.ma.masked_where(~mask, diff)

        real_vals = real[mask]; gen_vals = gen[mask]
        real_norm = np.zeros_like(real, dtype=float); gen_norm = np.zeros_like(gen, dtype=float)
        real_norm[mask] = (real_vals - real_vals.min()) / max(real_vals.max() - real_vals.min(), 1e-12)
        gen_norm[mask]  = (gen_vals  - gen_vals.min())  / max(gen_vals.max()  - gen_vals.min(),  1e-12)
        pattern_diff = np.abs(real_norm - gen_norm)
        pattern_diff[mask] = np.where(
            pattern_diff[mask] < HEATMAP_PATTERN_DIFF_TOLERANCE, 0.0, pattern_diff[mask]
        )
        pattern_diff_m = np.ma.masked_where(~mask, pattern_diff)

        vmax_real = float(max(real_m.max(), 1e-12)); vmax_gen = float(max(gen_m.max(), 1e-12))
        real_fg = real_m.compressed(); gen_fg = gen_m.compressed()
        pctl = float(HEATMAP_VMAX_PERCENTILE_FG)
        vmaxp_real = float(np.percentile(real_fg, pctl)) if real_fg.size else vmax_real
        vmaxp_gen  = float(np.percentile(gen_fg,  pctl)) if gen_fg.size  else vmax_gen
        p95_real   = float(np.percentile(real_fg, 95))   if real_fg.size else vmax_real
        p95_gen    = float(np.percentile(gen_fg,  95))   if gen_fg.size  else vmax_gen

        ax0, ax1, ax2 = axes[row_idx]
        im0 = ax0.imshow(real_m, cmap=cmap, interpolation="bicubic", aspect="auto",
                         origin="lower", vmin=0.0, vmax=vmaxp_real)
        ax0.set_title(f"{label} - Real (p{pctl:g}={vmaxp_real:.2f}, max={vmax_real:.2f})", fontsize=12)
        ax0.set_xlabel("X"); ax0.set_ylabel("Y")
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cb0.set_ticks(np.linspace(*im0.get_clim(), 6).tolist())

        im1 = ax1.imshow(gen_m, cmap=cmap, interpolation="bicubic", aspect="auto",
                         origin="lower", vmin=0.0, vmax=vmaxp_gen)
        ax1.set_title(
            f"{label} - Generated (p{pctl:g}={vmaxp_gen:.2f}, max={vmax_gen:.2f}, p95={p95_gen:.2f})",
            fontsize=12,
        )
        ax1.set_xlabel("X"); ax1.set_ylabel("Y")
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_ticks(np.linspace(*im1.get_clim(), 6).tolist())

        im2 = ax2.imshow(pattern_diff_m, cmap="RdYlGn_r", interpolation="bicubic",
                         aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        pearson_r = float(np.corrcoef(real_vals, gen_vals)[0, 1]) if len(real_vals) > 1 else float("nan")
        ax2.set_title(f"{label} - Pattern Diff (r={pearson_r:.3f})", fontsize=12)
        ax2.set_xlabel("X"); ax2.set_ylabel("Y")
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("|norm(Real) - norm(Gen)|"); cb2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

        mae = float(np.mean(diff_m.compressed())) if diff_m.count() else float("nan")
        print(
            f"{label}: heatmap MAE={mae:.6f}, "
            f"real_max={vmax_real:.3f} p95={p95_real:.3f}, "
            f"gen_max={vmax_gen:.3f} p95={p95_gen:.3f}"
        )

    fig.suptitle(f"Heatmap Comparison: Real vs Generated  [{_FREQ_LABEL}]", fontsize=16, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap comparison: {out_path}")


def _plot_impedance_comparisons(
    *, frequency: np.ndarray, target_impedance: np.ndarray,
    comparisons: list[dict], out_path: Path,
) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, comparisons):
        label = item["label"]
        ax.loglog(frequency, target_impedance,        "--", lw=2.5, label="Target",             color="red")
        ax.loglog(frequency, item["real_ohm"],         "-",  lw=1.2, label="Real",               color="royalblue")
        if (v := item.get("gen_raw_ohm"))    is not None:
            ax.loglog(frequency, v, "--", lw=1, color="#9E9E9E",        alpha=0.7, label="Ch0 raw")
        if (v := item.get("gen_integ_ohm"))  is not None:
            ax.loglog(frequency, v, "--", lw=1, color="mediumseagreen", alpha=0.7, label="Int Ch1")
        if (v := item.get("gen_integ2_ohm")) is not None:
            ax.loglog(frequency, v, "--", lw=1, color="darkorange",     alpha=0.7, label="Int Ch2")
        ax.loglog(frequency, item["gen_blended_ohm"],  "-",  lw=2.5, label="Generated (blended)", color="green")

        if (d1 := item.get("gen_derivative")) is not None and len(np.atleast_1d(d1)) == len(frequency):
            ax2 = ax.twinx()
            ax2.semilogx(frequency, np.asarray(d1).reshape(-1), ":", lw=1.2,
                         color="darkorange", label="Derivative (z-score)")
            ax2.axhline(0, color="darkorange", linewidth=0.6, linestyle=":", alpha=0.4)
            ax2.set_ylabel("First Derivative (z-score)", fontsize=11, color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")
        else:
            ax.legend(fontsize=9, loc="best")

        ax.set_ylim(1e-3, 1e2)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Impedance (Ohm)", fontsize=12)
        ax.set_title(f"{label}: Generated vs Real", fontsize=14)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle(f"Impedance Profile Comparison  [{_FREQ_LABEL}]", fontsize=16, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved impedance comparison: {out_path}")


def _load_occupancy_matrix(k_dir: Path, *, num_samples: int) -> np.ndarray:
    occ_path = k_dir / "occupancy.npy"
    if occ_path.exists():
        occ = np.asarray(np.load(occ_path))
        if occ.ndim == 1:
            return occ.reshape(1, 52)
        if occ.ndim == 2 and occ.shape[1] == 52:
            return occ
        raise ValueError(f"Unexpected occupancy.npy shape {occ.shape} in {occ_path}")
    rows: list[np.ndarray] = []
    for i in range(num_samples):
        p = k_dir / f"data_sample_{i}" / "occupancy_map.npy"
        if not p.exists():
            raise SystemExit(f"No occupancy.npy and missing per-sample occupancy_map.npy: {p}")
        rows.append(np.asarray(np.load(p)).reshape(-1))
    return np.stack(rows, axis=0)


def _plot_checkboxes(
    *, occupancy_matrix: np.ndarray, out_path: Path,
    title_prefix: str, expected_k: int | None = None,
) -> None:
    occ = np.asarray(occupancy_matrix)
    if occ.ndim != 2 or occ.shape[1] != 52:
        raise ValueError(f"Expected occupancy_matrix shape (N,52), got {occ.shape}")

    def _active_mask(v: np.ndarray) -> np.ndarray:
        vf = np.asarray(v, dtype=float).reshape(-1)
        if ACTIVE_POLICY == "threshold" or expected_k is None:
            return vf > float(THRESHOLD)
        k = int(expected_k)
        if k <= 0:  return np.zeros(52, dtype=bool)
        if k >= 52: return np.ones(52, dtype=bool)
        vf = np.where(np.isfinite(vf), vf, -np.inf)
        topk_idx = np.argsort(-vf, kind="stable")[:k]
        mask = np.zeros(52, dtype=bool)
        mask[topk_idx] = True
        return mask

    n = occ.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.2, 1.25 * n)))
    axes_list: list[Axes] = [axes] if isinstance(axes, Axes) else list(axes)

    active_face   = "#C8E6C9"; inactive_face = "#FAFAFA"
    inactive_text = "#9E9E9E"; edge_color    = "#BDBDBD"
    n_rows, n_cols = 4, 13
    cell_w, cell_h = 1.0, 1.0

    for i, ax in enumerate(axes_list):
        v = occ[i].reshape(-1)
        active   = _active_mask(v)
        k_active = int(active.sum())
        ax.set_xlim(0, n_cols * cell_w); ax.set_ylim(0, n_rows * cell_h)
        ax.set_aspect("equal"); ax.axis("off")
        for idx in range(52):
            r = idx // n_cols; c = idx % n_cols
            y_pos = (n_rows - 1 - r) * cell_h; x_pos = c * cell_w
            is_on = bool(active[idx])
            ax.add_patch(FancyBboxPatch(
                (x_pos, y_pos), cell_w, cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=(active_face if is_on else inactive_face),
                edgecolor=edge_color, linewidth=0.9,
            ))
            ax.text(x_pos + cell_w / 2, y_pos + 0.22, f"C{idx + 1}",
                    ha="center", va="center", fontsize=6.5,
                    color=("#212121" if is_on else inactive_text))
        policy = "topK" if ACTIVE_POLICY == "topk" and expected_k is not None else f"> {THRESHOLD:g}"
        k_note = (
            f"  (policy={policy}, active {k_active}/52"
            + (f", expected K={expected_k}" if expected_k is not None else "")
            + ")"
        )
        ax.set_title(
            f"{title_prefix}sample_{i}{k_note}  [{_FREQ_LABEL}]",
            fontsize=9.5, pad=3, loc="left",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved occupancy plot: {out_path}")


def _run_compare_single_k(k: int, *, base_dir: Path, repo_root: Path) -> None:
    k_dir = (base_dir / f"K{k}").resolve()
    if not k_dir.exists():
        raise SystemExit(f"K folder not found: {k_dir}")

    num_samples = _infer_num_samples(k_dir)
    print(f"\n=== K={k}  ({num_samples} samples) ===")

    if RUN_HEATMAP or RUN_IMPEDANCE:
        real_dir = k_dir / "Real"
        if not real_dir.exists():
            raise SystemExit(f"Real outputs folder not found (run move step first): {real_dir}")

        mask = np.load(repo_root / MASK_PATH).astype(bool) if RUN_HEATMAP else None
        comparisons_heatmap: list[dict] = []
        comparisons_imp:     list[dict] = []

        for i in range(num_samples):
            sample_dir = k_dir / f"data_sample_{i}"
            if not sample_dir.exists():
                raise SystemExit(f"Missing sample dir: {sample_dir}")

            label = f"{_FREQ_LABEL + ' ' if _FREQ_LABEL else ''}K{k}/sample_{i}"

            if RUN_HEATMAP:
                heat_candidates = sorted(real_dir.glob(f"Heatmap_real_{i}*"))
                if not heat_candidates:
                    raise SystemExit(f"No real heatmap item for sample {i} under: {real_dir}")
                gen_heat  = _load_generated_heatmap(sample_dir / "heatmap_physical.npy")
                real_heat = _load_map_file(
                    _choose_real_heatmap_mapfile(heat_candidates[0]),
                    resolution=gen_heat.shape[0],
                )
                comparisons_heatmap.append({"generated": gen_heat, "real": real_heat, "label": label})

            if RUN_IMPEDANCE:
                frequency        = np.load(repo_root / FREQUENCY_PATH).squeeze()
                target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH).squeeze()
                gen_blended_log  = _load_generated_impedance_log(sample_dir / "impedance_profile.npy")
                gen_raw_log      = _maybe_load(sample_dir / "impedance_raw.npy")
                gen_integ_log    = _maybe_load(sample_dir / "impedance_integrated.npy")
                gen_integ2_log   = _maybe_load(sample_dir / "impedance_integrated2.npy")
                gen_d1           = _maybe_load(sample_dir / "impedance_derivative.npy")

                imp_candidates = _real_impedance_candidates(real_dir, i)
                if not imp_candidates:
                    raise SystemExit(f"No real impedance item for sample {i} under: {real_dir}")
                real_imp_ohm = _load_real_impedance(
                    _choose_real_impedance_csv(imp_candidates[0]), frequency_hz=frequency
                )
                d: dict = {
                    "label": label,
                    "real_ohm": real_imp_ohm,
                    "gen_blended_ohm": np.exp(gen_blended_log),
                }
                if gen_raw_log    is not None: d["gen_raw_ohm"]    = np.exp(np.asarray(gen_raw_log).reshape(-1))
                if gen_integ_log  is not None: d["gen_integ_ohm"]  = np.exp(np.asarray(gen_integ_log).reshape(-1))
                if gen_integ2_log is not None: d["gen_integ2_ohm"] = np.exp(np.asarray(gen_integ2_log).reshape(-1))
                if gen_d1         is not None: d["gen_derivative"]  = np.asarray(gen_d1).reshape(-1)
                comparisons_imp.append(d)

        if RUN_HEATMAP and comparisons_heatmap:
            assert mask is not None
            _plot_heatmap_comparisons(
                comparisons=comparisons_heatmap, mask=mask, out_path=k_dir / HEATMAP_OUT_NAME,
            )
        if RUN_IMPEDANCE and comparisons_imp:
            frequency        = np.load(repo_root / FREQUENCY_PATH).squeeze()
            target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH).squeeze()
            _plot_impedance_comparisons(
                frequency=frequency, target_impedance=target_impedance,
                comparisons=comparisons_imp, out_path=k_dir / IMPEDANCE_OUT_NAME,
            )

    if RUN_OCCUPANCY:
        occ = _load_occupancy_matrix(k_dir, num_samples=num_samples)
        _plot_checkboxes(
            occupancy_matrix=occ, out_path=k_dir / OCCUPANCY_OUT_NAME,
            title_prefix=f"K{k}/", expected_k=k,
        )


def _run_compare_for_freq(mhz: int | None, *, repo_root: Path) -> None:
    global _FREQ_LABEL
    _FREQ_LABEL = f"{mhz} MHz" if mhz is not None else ""
    base_dir    = repo_root / _base_dir_for_freq(mhz)
    label       = f"freq_{mhz}MHz" if mhz is not None else "(no freq subfolder)"

    print(f"\n{'='*60}")
    print(f"Comparing {label}  ->  {base_dir}")
    print(f"{'='*60}")

    ok: list[int] = []; failed: list[tuple[int, str]] = []; skipped: list[int] = []
    for k in range(K_MIN, K_MAX + 1):
        try:
            _run_compare_single_k(k, base_dir=base_dir, repo_root=repo_root)
            ok.append(k)
        except SystemExit as e:
            msg = str(e)
            failed.append((k, msg))
            if any(s in msg for s in ("folder not found", "No data_sample", "No real")):
                skipped.append(k)
            if FAIL_FAST:
                raise
        except Exception as e:
            failed.append((k, repr(e)))
            traceback.print_exc()
            if FAIL_FAST:
                raise

    print(f"\n=== Summary ({label}) ===")
    print(f"OK: {len(ok)}  Failed: {len(failed)}  Skipped: {len(set(skipped))}")
    if failed:
        print("\nFailures:")
        for k, msg in failed:
            print(f"  K{k}: {msg}")


# ============================================================
# Report helpers
# ============================================================

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: Arial, sans-serif; background: #f5f5f5; color: #222; }
#header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
h1 { padding: 20px 28px 10px; font-size: 3rem; text-align: center; }
.subtitle { text-align: center; font-size: 1.1rem; color: #555; padding-bottom: 12px; }
.tab-bar {
    display: flex; justify-content: center; gap: 24px;
    padding: 0 36px; border-bottom: 3px solid #ccc; background: #fff;
}
.tab-btn {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 18px 48px 14px; border: none; border-bottom: 5px solid transparent;
    background: none; cursor: pointer; font-size: 1.5rem; font-weight: 700;
    color: #555; letter-spacing: 0.03em; transition: color .15s, border-color .15s;
}
.tab-btn svg { width: 3rem; height: 3rem; stroke: currentColor; fill: none; stroke-width: 1.8; }
.tab-btn:hover { color: #000; }
.tab-btn.active { color: #1a73e8; border-bottom-color: #1a73e8; }
#content { position: fixed; left: 0; right: 0; bottom: 0; overflow: hidden; }
.tab-panel {
    display: none; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    overflow-y: auto; padding: 28px;
}
.tab-panel.active { display: block; }
.k-section { margin-bottom: 36px; text-align: center; }
.k-section h3 {
    font-size: 1.75rem; font-weight: 700; margin-bottom: 10px;
    padding: 8px 14px; background: #e8f0fe;
    border-left: 5px solid #1a73e8; border-radius: 3px; text-align: center;
}
.k-section img {
    max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
    background: #fff; display: block; margin: 0 auto;
}
.missing { color: #c00; font-style: italic; }
.k-links { display: flex; justify-content: center; gap: 12px; margin: 8px 0 16px; flex-wrap: wrap; }
.k-link {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 18px;
    font-size: 1rem; font-weight: 600; color: #1a73e8; background: #e8f0fe;
    border: 1.5px solid #1a73e8; border-radius: 20px; cursor: pointer;
    transition: background .15s, color .15s;
}
.k-link:hover { background: #1a73e8; color: #fff; }
"""

_JS = """
function showTab(idx) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
    document.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
}
function goTo(tabIdx, anchorId) {
    showTab(tabIdx);
    requestAnimationFrame(() => {
        const el = document.getElementById(anchorId);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}
function fitContent() {
    const h = document.getElementById('header').offsetHeight;
    document.getElementById('content').style.top = h + 'px';
}
window.addEventListener('load', fitContent);
window.addEventListener('resize', fitContent);
new ResizeObserver(fitContent).observe(document.getElementById('header'));
"""

_ICON_HEATMAP = (
    '<svg viewBox="0 0 24 24">'
    '<path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-8-7-8z"/>'
    '<circle cx="12" cy="10" r="2.5"/>'
    '</svg>'
)
_ICON_IMPEDANCE = (
    '<svg viewBox="0 0 24 24">'
    '<polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/>'
    '</svg>'
)
_ICON_OCCUPANCY = (
    '<svg viewBox="0 0 24 24">'
    '<rect x="3" y="3" width="7" height="7" rx="1"/>'
    '<rect x="14" y="3" width="7" height="7" rx="1"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1"/>'
    '<rect x="14" y="14" width="7" height="7" rx="1"/>'
    '</svg>'
)


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">Image not found: {path}</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def _copy_report_to_dest(out_path: Path, dest_str: str | None) -> None:
    if not dest_str:
        return
    if re.match(r"^[A-Za-z]:\\", dest_str):
        win = PureWindowsPath(dest_str)
        drive = win.drive.rstrip(":").lower()
        dest_path = Path("/mnt") / drive / Path(*win.parts[1:])
    else:
        dest_path = Path(dest_str)
    if dest_path.exists() and dest_path.is_dir():
        target = dest_path / out_path.name
        shutil.copy2(str(out_path), str(target))
        print(f"Copied to   : {target}")
        print(f"Open from Windows: {dest_str}\\{out_path.name}")
    else:
        print(f"! REPORT_COPY_DEST not found, skipping copy: {dest_path}")


def _build_panel(
    panel_id: str,
    tab_idx: int,
    k_range: range,
    img_name: str,
    base_dir: Path,
    freq_scan: list[tuple[str, str | None]],
    all_tabs: list[tuple[str, str, str, str]],
    *,
    active: bool = False,
) -> str:
    active_cls = " active" if active else ""
    parts = [f'<div class="tab-panel{active_cls}" id="{panel_id}">']
    for freq_label, freq_sub in freq_scan:
        freq_dir = base_dir / freq_sub if freq_sub else base_dir
        for k in k_range:
            img_path = freq_dir / f"K{k}" / img_name
            safe_tag = freq_sub.replace("_", "-") if freq_sub else "nf"
            sec_id = f"{panel_id}-{safe_tag}-k{k}"
            heading = f"K = {k}" + (f" @ {freq_label}" if freq_label else "")
            parts.append(f'  <div class="k-section" id="{sec_id}">')
            parts.append(f"    <h3>{heading}</h3>")
            if len(all_tabs) > 1:
                links = []
                for i, (tid, label, _img, icon) in enumerate(all_tabs):
                    if i == tab_idx:
                        continue
                    short = label.replace(" Comparison", "")
                    cross_id = f"{tid}-{safe_tag}-k{k}"
                    links.append(
                        f'<button class="k-link" onclick="goTo({i},{chr(39)}{cross_id}{chr(39)})">' +
                        f"{icon} {short}</button>"
                    )
                parts.append(f'    <div class="k-links">{" ".join(links)}</div>')
            parts.append(f"    {_img_tag(img_path, f'K={k}')}")
            parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def _write_markdown_summary(
    *,
    out_md: Path,
    base_dir: Path,
    freq_scan: list[tuple[str, str | None]],
    k_range: range,
    title: str,
    html_name: str,
) -> None:
    lines = [
        f"# {title}", "",
        f"Interactive gallery: [`{html_name}`](./{html_name})", "",
        "## Heatmap comparisons", "",
        "| PI frequency | K | PNG | Status |",
        "|---|---:|---|---|",
    ]
    for freq_label, freq_sub in freq_scan:
        freq_dir = base_dir / freq_sub if freq_sub else base_dir
        for k in k_range:
            rel = f"{freq_sub}/K{k}/{HEATMAP_OUT_NAME}" if freq_sub else f"K{k}/{HEATMAP_OUT_NAME}"
            img_path = freq_dir / f"K{k}" / HEATMAP_OUT_NAME
            status = "ok" if img_path.is_file() else "missing"
            fl = freq_label or "—"
            lines.append(f"| {fl} | {k} | `{rel}` | {status} |")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {out_md}")


def render_comparison_report(
    *,
    base_dir: Path,
    output_html: Path,
    output_md: Path | None,
    freq_scan: list[tuple[str, str | None]],
    k_min: int,
    k_max: int,
    tabs: list[tuple[str, str, str, str]],
    title: str,
    subtitle: str = "",
    report_copy_dest: str | None = None,
) -> Path:
    base_dir = base_dir.resolve()
    out_path = output_html.resolve()
    k_range = range(k_min, k_max + 1)

    show_tabs = len(tabs) > 1
    buttons = ""
    if show_tabs:
        buttons = "\n".join(
            f'<button class="tab-btn{" active" if i == 0 else ""}" ' +
            f'onclick="showTab({i})">{icon}<span>{label}</span></button>'
            for i, (_, label, _, icon) in enumerate(tabs)
        )

    panels = "\n".join(
        _build_panel(tab_id, i, k_range, img_name, base_dir, freq_scan, tabs, active=(i == 0))
        for i, (tab_id, _, img_name, _icon) in enumerate(tabs)
    )

    sub_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    tab_bar = f'<div class="tab-bar">\n{buttons}\n</div>' if show_tabs else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div id="header">
<h1>{title}</h1>
{sub_html}
{tab_bar}
</div>
<div id="content">
{panels}
</div>
<script>{_JS}</script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")

    if output_md is not None:
        _write_markdown_summary(
            out_md=output_md.resolve(),
            base_dir=base_dir,
            freq_scan=freq_scan,
            k_range=k_range,
            title=title,
            html_name=out_path.name,
        )

    _copy_report_to_dest(out_path, report_copy_dest)
    return out_path


def build_report(*, report_copy_dest: str | None = None) -> Path:
    """Build the HTML comparison report for the current run_all_k configuration."""
    from scrap.generation.run_all_k import PEB_COPY_DEST  # noqa: E402

    if _RUN_PI_FREQ_MHZ is None:
        freq_scan: list[tuple[str, str | None]] = [("", None)]
    elif isinstance(_RUN_PI_FREQ_MHZ, (int, float)):
        mhz = int(_RUN_PI_FREQ_MHZ)
        freq_scan = [(f"{mhz} MHz", f"freq_{mhz}MHz")]
    else:
        freq_scan = [(f"{int(f)} MHz", f"freq_{int(f)}MHz") for f in _RUN_PI_FREQ_MHZ]

    dest = report_copy_dest if report_copy_dest is not None else PEB_COPY_DEST
    tabs = [
        ("tab-heatmap",   "Heatmap Comparison",   HEATMAP_OUT_NAME,   _ICON_HEATMAP),
        ("tab-impedance", "Impedance Comparison",  IMPEDANCE_OUT_NAME, _ICON_IMPEDANCE),
        ("tab-occupancy", "Occupancy Comparison",  OCCUPANCY_OUT_NAME, _ICON_OCCUPANCY),
    ]
    _dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _PROJECT_ROOT / BASE_GENERATED_DIR
    return render_comparison_report(
        base_dir=base,
        output_html=base / f"comparison_report_{_dt}.html",
        output_md=base / f"comparison_summary_{_dt}.md",
        freq_scan=freq_scan,
        k_min=K_MIN,
        k_max=K_MAX,
        tabs=tabs,
        title="Generated vs Real -- Comparison Report",
        report_copy_dest=dest,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not (0 <= K_MIN <= K_MAX <= 52):
        raise SystemExit("Expected 0 <= K_MIN <= K_MAX <= 52")

    repo_root  = _PROJECT_ROOT
    multi_freq = _FREQ_LIST != [None]
    os.chdir(repo_root)

    if REPORT_ONLY:
        print(f"\n{'='*60}")
        print("Report-only mode: skipping move and compare steps")
        print(f"{'='*60}")
    else:
        # -- Step 1: Move PI-* outputs into Real/ folders ---------------------
        print(f"\n{'='*60}")
        print("Step 1: Moving PI-* outputs")
        print(f"{'='*60}")

        if multi_freq:
            freq_list_int = [mhz for mhz in _FREQ_LIST if mhz is not None]
            move_pi_outputs_multi_freq(
                freq_list=freq_list_int,
                k_min=K_MIN, k_max=K_MAX,
                num_samples=_RUN_NUM_SAMPLES,
            )
        else:
            move_pi_outputs_for_k_range(K_MIN, K_MAX, base_generated_dir=_base_dir_for_freq(None))

        # -- Step 2: Compare generated vs real --------------------------------
        print(f"\n{'='*60}")
        print("Step 2: Comparing generated vs real")
        print(f"{'='*60}")

        for mhz in _FREQ_LIST:
            _run_compare_for_freq(mhz, repo_root=repo_root)

    # -- Step 3: Build HTML report --------------------------------------------
    print(f"\n{'='*60}")
    print("Step 3: Building comparison report")
    print(f"{'='*60}")

    build_report()


if __name__ == "__main__":
    main()
