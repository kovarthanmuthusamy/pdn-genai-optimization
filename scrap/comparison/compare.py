"""Compare Generated vs Real (Workflow-Aware).

Purpose: Plot heatmap, impedance, and/or occupancy comparisons for each K (and optional
    freq_*MHz subfolders) using generated samples vs ECADStar Real/ outputs.
Run: python scrap/comparison/compare.py  (set WORKFLOW in CONFIGURATION first)
Inputs / outputs: BASE_GENERATED_DIR / OUTPUT_ROOT from run_all_k or run_multifreq_heatmap_sweep;
    reads configs/{Frequency_data_hz,target_impedance,binary_mask}.npy;
    writes generated_vs_real_*.png per K folder.
Dependencies: matplotlib, numpy, scipy; imports generation config by WORKFLOW.
Agent notes:
    - Type: CLI script with WORKFLOW switch (run_all_k | multifreq_heatmap_sweep)
    - Key symbols: main, _run_single_k, HEATMAP_OUT_NAME, WORKFLOW, _plot_heatmap_comparisons,
    - Config keys: ``WORKFLOW``, ``K_MIN``, ``_freq_src``, ``_FREQ_LIST``, ``_FREQ_LIST``, ``_FREQ_LIST``, ``FREQ_LABEL``, ``RUN_HEATMAP``, ``RUN_IMPEDANCE``, ``RUN_OCCUPANCY``, ``FAIL_FAST``, ``FREQUENCY_PATH``, ``TARGET_IMPEDANCE_PATH``, ``MASK_PATH``, ``HEATMAP_OUT_NAME``, ``IMPEDANCE_OUT_NAME``, ``OCCUPANCY_OUT_NAME``, ``HEATMAP_CMAP``, ``HEATMAP_LEVELS``, ``HEATMAP_DIFF_TOLERANCE``, ``HEATMAP_PATTERN_DIFF_TOLERANCE``, ``HEATMAP_SHARED_COLOR_SCALE``, ``HEATMAP_VMAX_PERCENTILE_FG``, ``MAP_GLOB_PREFERENCE``, ``IMPEDANCE_CSV_GLOB_PREFERENCE``, ``ACTIVE_POLICY``, ``THRESHOLD``
      _plot_impedance_comparisons, _plot_checkboxes
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator

from scrap.comparison.heatmap_sim_metrics import (  # noqa: E402
    compute_sim_heatmap_metrics,
    print_metrics_summary,
    write_sim_metrics_bundle,
)
from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path
setup_path()

# ============================================================
# CONFIGURATION
# ============================================================
WORKFLOW = "multifreq_heatmap_sweep"  # "run_all_k" | "multifreq_heatmap_sweep"

PI_FREQ_OVERRIDE: int | list[int] | None = None  # run_all_k only; None = use run_all_k list

if WORKFLOW == "multifreq_heatmap_sweep":
    from scrap.generation.run_multifreq_heatmap_sweep import (  # noqa: E402
        OUTPUT_ROOT as _OUTPUT_ROOT,
        SWEEP_FREQ_MANIFEST,
        exported_freq_mhz_list,
        exported_k_values,
    )

    _K_VALUES: list[int] = exported_k_values()
    K_MIN = min(_K_VALUES)
    K_MAX = max(_K_VALUES)
    _FREQ_LIST: list[int | None] = []  # filled in main() via exported_freq_mhz_list()
elif WORKFLOW == "run_all_k":
    from scrap.generation.run_all_k import (  # noqa: E402
        K_MIN,
        K_MAX,
        OUTPUT_ROOT as _OUTPUT_ROOT,
        PI_FREQ_MHZ as _RUN_PI_FREQ_MHZ,
    )

    _freq_src = PI_FREQ_OVERRIDE if PI_FREQ_OVERRIDE is not None else _RUN_PI_FREQ_MHZ
    if _freq_src is None:
        _FREQ_LIST = [None]
    elif isinstance(_freq_src, list):
        _FREQ_LIST = list(_freq_src)
    else:
        _FREQ_LIST = [int(_freq_src)]
else:
    raise SystemExit(f"Unknown WORKFLOW={WORKFLOW!r}")


def _base_dir_for_freq(mhz: int | None) -> Path:
    if mhz is None:
        return Path(_OUTPUT_ROOT)
    return Path(f"{_OUTPUT_ROOT}/freq_{mhz}MHz")


# Base folder under repo root containing K{n}/ (overridden per-freq in main(), or by callers)
BASE_GENERATED_DIR: Path | str = Path(_OUTPUT_ROOT)

# Human-readable frequency label used in plot titles (set per-freq in the loop)
FREQ_LABEL = ""  # overridden at runtime

# Which comparisons to run (multifreq_heatmap_sweep forces heatmap-only)
RUN_HEATMAP           = True
RUN_IMPEDANCE         = WORKFLOW != "multifreq_heatmap_sweep"
RUN_OCCUPANCY         = WORKFLOW != "multifreq_heatmap_sweep"

# If True, stop at the first K that fails; otherwise collect and report all failures.
FAIL_FAST = False

# Assets
FREQUENCY_PATH        = Path("configs/Frequency_data_hz.npy")
TARGET_IMPEDANCE_PATH = Path("configs/target_impedance.npy")
MASK_PATH             = Path("configs/binary_mask.npy")

# Output file names (written inside each K{n}/ folder)
HEATMAP_OUT_NAME     = "generated_vs_real_heatmap.png"
IMPEDANCE_OUT_NAME   = "generated_vs_real_impedance_profile.png"
OCCUPANCY_OUT_NAME   = "generated_occupancy.png"

# Heatmap settings
HEATMAP_CMAP           = "jet"
HEATMAP_LEVELS         = 22
HEATMAP_DIFF_TOLERANCE = 0.0   # no suppression — show full |Real − Gen| in diff panel
# Pattern diff: suppress noise below this (normalised [0,1] space)
HEATMAP_PATTERN_DIFF_TOLERANCE = 0.05
# Absolute diff panel colormap resolution and colorbar tick count.
HEATMAP_DIFF_LEVELS = 25
HEATMAP_DIFF_COLORBAR_TICKS = 25
HEATMAP_DIFF_CMAP = "hot"
# vmax mode: "max" (per-panel FG max) | "percentile" | "stats_percentile" | "global_max"
HEATMAP_VMAX_MODE = "percentile"
# Use real heatmap color scale for generated panel (legacy; set True with stats_percentile)
HEATMAP_SHARED_COLOR_SCALE = False
# When True, Real panel colorbar vmax uses Generated panel's FG percentile (real_vmax = gen_vmax).
# Opposite of HEATMAP_SHARED_COLOR_SCALE (gen_vmax = real_vmax). Ignored when stats/global shared vmax is active.
HEATMAP_REAL_VMAX_MATCH_GENERATED = False
# Per-panel foreground percentile when HEATMAP_VMAX_MODE == "percentile".
HEATMAP_VMAX_PERCENTILE_FG = 99.9
# Dataset stats percentile of per-map FG max → shared colorbar vmax (Ω).
HEATMAP_VMAX_STATS_PERCENTILE = 99.9
# Manual override for shared vmax (Ω); None → derive from HEATMAP_STATS_DATA_DIR stats.
HEATMAP_VMAX_GLOBAL_MAX_OHM: float | None = None
HEATMAP_STATS_DATA_DIR: str | None = None
# Number of tick marks on Real/Generated colorbars (cmap resampling uses HEATMAP_LEVELS).
HEATMAP_COLORBAR_TICKS = 6

# Render speed: output DPI + image interpolation. Lower DPI / cheaper interpolation = faster.
HEATMAP_PLOT_DPI = 130
HEATMAP_INTERPOLATION = "bilinear"  # "nearest" (fastest) | "bilinear" | "bicubic" (smoothest)

# Real heatmap .map file chooser (inside Heatmap_real_* directory)
MAP_GLOB_PREFERENCE = ("Z_*.map", "*.map")

# Real impedance CSV chooser (inside Imp_Real* directory)
IMPEDANCE_CSV_GLOB_PREFERENCE = ("*PIPinZ*.csv", "*.csv")

# Occupancy settings
ACTIVE_POLICY = "topk"   # "topk" (activate exactly K slots) or "threshold"
THRESHOLD     = 0.5

# Post-simulation metrics (generated vs ECADStar Real/ .map — not dataset val)
WRITE_SIM_METRICS = True
METRICS_ONLY = False  # True → only write sim_compare_metrics.* (no PNG plots)
# ============================================================


def _project_root() -> Path:
    from repo_paths import REPO_ROOT
    return REPO_ROOT


def _load_heatmap_stats() -> dict:
    data_dir = HEATMAP_STATS_DATA_DIR
    if data_dir is None:
        data_dir = str(_project_root() / "datasets" / "data_multifreq_gmax")
    stats_path = Path(data_dir) / "normalization_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Heatmap stats vmax requires normalization_stats.json: {stats_path}"
        )
    import json

    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    return raw.get("Heatmap") or {}


def _resolve_heatmap_vmax_ohm() -> float:
    """Shared colorbar vmax in Ω from normalization stats."""
    if HEATMAP_VMAX_GLOBAL_MAX_OHM is not None:
        return float(HEATMAP_VMAX_GLOBAL_MAX_OHM)

    hm = _load_heatmap_stats()
    gmax = hm.get("global_max_ohm")
    if gmax is None:
        raise KeyError("global_max_ohm missing in normalization Heatmap stats")

    pct = float(HEATMAP_VMAX_STATS_PERCENTILE)
    for key in (
        f"per_map_max_p{pct:g}",
        f"per_map_max_p{int(pct)}",
        f"per_map_max_p{str(pct).replace('.', '_')}",
    ):
        if key in hm:
            return float(hm[key])

    gmax_f = float(gmax)
    clip_max = float(hm.get("clip_max", 1.0))
    gmax_pct = hm.get("gmax_percentile")
    if gmax_pct is not None and abs(float(gmax_pct) - pct) < 0.05:
        return gmax_f
    # p99.9+ of per-map max ≈ clip ceiling in physical Ω (maps are clipped at clip_max * gmax).
    if pct >= 99.9 - 1e-6:
        return clip_max * gmax_f
    return gmax_f


def _resolve_global_max_ohm() -> float:
    """Backward-compatible alias."""
    return _resolve_heatmap_vmax_ohm()


def _colorbar_ticks(vmin: float, vmax: float, *, n: int | None = None) -> list[float]:
    n_ticks = max(2, int(n if n is not None else HEATMAP_COLORBAR_TICKS))
    return np.linspace(vmin, vmax, n_ticks).tolist()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _infer_num_samples(k_dir: Path) -> int:
    sample_dirs = [p for p in k_dir.glob("data_sample_*") if p.is_dir()]
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* folders found in: {k_dir}")

    def _idx(p: Path) -> int | None:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return None

    indices = sorted(i for i in (_idx(p) for p in sample_dirs) if i is not None)
    expected = list(range(0, max(indices) + 1))
    if indices != expected:
        raise SystemExit(
            "data_sample_* folders are not consecutive starting at 0.\n"
            f"  Found:    {indices[:20]}{' ...' if len(indices) > 20 else ''}\n"
            f"  Expected: {expected[:20]}{' ...' if len(expected) > 20 else ''}"
        )
    return len(expected)


# ── Heatmap + impedance helpers ───────────────────────────────────────────────

def _load_generated_heatmap(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected heatmap to be 2D, got shape {data.shape} from {path}")
    return data


def _load_map_file(file_path: Path, *, resolution: int) -> np.ndarray:
    """Load a real heatmap from a .map file and interpolate to a square grid."""
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
                    x.append(px)
                    y.append(py)
                    z.append(max(pz, 0.01))
                i += 3
            except Exception:
                pass
        i += 1

    if not x:
        raise RuntimeError(f"No data loaded from MAP file: {file_path}")

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)

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


def _parse_pi_number(name: str) -> int | None:
    import re

    m = re.match(r"^PI-(\d+)(?:\..+)?$", name)
    return int(m.group(1)) if m else None


def _real_impedance_candidates(real_dir: Path, sample_i: int) -> list[Path]:
    """Imp_Real{i}* after rename, or fallback to PI-* folders still in Real/."""
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
                x = float(parts[0])
                y = float(parts[1])
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
    x_sorted = x[order]
    z_sorted = z[order]
    good = (x_sorted > 0) & (z_sorted > 0)
    x_sorted = x_sorted[good]
    z_sorted = z_sorted[good]

    if len(x_sorted) < 10:
        return z

    fx = np.log10(frequency_hz.astype(float))
    xi = np.log10(x_sorted.astype(float))
    zi = np.log10(z_sorted.astype(float))
    return 10 ** np.interp(fx, xi, zi)


def _load_generated_impedance_log(path: Path) -> np.ndarray:
    data = np.load(path).squeeze()
    if data.ndim != 1:
        data = data.reshape(-1)
    return data


def _maybe_load(path: Path) -> np.ndarray | None:
    return np.load(path).squeeze() if path.exists() else None


def _panel_title(ax: Axes, headline: str, detail: str = "", *, fontsize: float = 11.0) -> None:
    """Panel title; optional second line only when detail is provided."""
    text = headline if not detail else f"{headline}\n{detail}"
    ax.set_title(text, fontsize=fontsize, pad=8, loc="center")


def _max_diff_caption(max_diff_ohm: float) -> str:
    """Signed peak diff: real_max − gen_max (− → gen higher, + → gen lower)."""
    if not np.isfinite(max_diff_ohm):
        return "Δmax: n/a"
    if abs(max_diff_ohm) < 1e-6:
        return "Δmax ≈ 0 Ω (matched)"
    if max_diff_ohm < 0:
        return f"Δmax = {max_diff_ohm:+.2f} Ω (gen > real)"
    return f"Δmax = {max_diff_ohm:+.2f} Ω (gen < real)"


def _plot_heatmap_comparisons(
    *,
    comparisons: list[dict],
    mask: np.ndarray,
    out_path: Path,
) -> list[dict]:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.8 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    cmap = mpl.colormaps[HEATMAP_CMAP].resampled(HEATMAP_LEVELS).copy()
    cmap.set_bad("white")
    diff_cmap = mpl.colormaps[HEATMAP_DIFF_CMAP].resampled(HEATMAP_DIFF_LEVELS).copy()
    diff_cmap.set_bad("white")
    metric_rows: list[dict] = []

    use_shared = HEATMAP_VMAX_MODE in ("global_max", "stats_percentile") or HEATMAP_SHARED_COLOR_SCALE
    shared_vmax: float | None = None
    shared_label: str | None = None
    if use_shared:
        shared_vmax = _resolve_heatmap_vmax_ohm()
        pct = float(HEATMAP_VMAX_STATS_PERCENTILE)
        if HEATMAP_VMAX_MODE == "stats_percentile":
            shared_label = f"p{pct:g}={shared_vmax:.2f}Ω"
        else:
            shared_label = f"gmax={shared_vmax:.2f}Ω"

    pctl = float(HEATMAP_VMAX_PERCENTILE_FG)

    for row_idx, item in enumerate(comparisons):
        real = item["real"]
        gen = item["generated"]
        label = item["label"]

        real_m = np.ma.masked_where(~mask, real)
        gen_m = np.ma.masked_where(~mask, gen)

        real_vals = real[mask]
        gen_vals = gen[mask]

        amp_diff = np.abs(real - gen)
        amp_diff_m = np.ma.masked_where(~mask, amp_diff)

        vmax_real = float(max(real_m.max(), 1e-12))
        vmax_gen = float(max(gen_m.max(), 1e-12))
        real_fg = real_m.compressed()
        gen_fg = gen_m.compressed()
        if HEATMAP_VMAX_MODE == "max":
            scale_real = vmax_real
            scale_gen = vmax_gen
        else:
            scale_real = float(np.percentile(real_fg, pctl)) if real_fg.size else vmax_real
            scale_gen = float(np.percentile(gen_fg, pctl)) if gen_fg.size else vmax_gen

        if shared_vmax is not None:
            plot_vmax_real = shared_vmax
            plot_vmax_gen = shared_vmax
        else:
            if HEATMAP_SHARED_COLOR_SCALE and HEATMAP_REAL_VMAX_MATCH_GENERATED:
                raise ValueError(
                    "HEATMAP_SHARED_COLOR_SCALE and HEATMAP_REAL_VMAX_MATCH_GENERATED are "
                    "mutually exclusive (gen_from_real vs real_from_gen)."
                )
            if HEATMAP_REAL_VMAX_MATCH_GENERATED:
                plot_vmax_real = scale_gen
                plot_vmax_gen = scale_gen
            elif HEATMAP_SHARED_COLOR_SCALE:
                plot_vmax_real = scale_real
                plot_vmax_gen = scale_real
            else:
                plot_vmax_real = scale_real
                plot_vmax_gen = scale_gen

        ax0, ax1, ax2 = axes[row_idx]

        im0 = ax0.imshow(
            real_m, cmap=cmap, interpolation=HEATMAP_INTERPOLATION, aspect="auto", origin="lower",
            vmin=0.0, vmax=plot_vmax_real,
        )
        _panel_title(ax0, f"{label} · Real (sim)")
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cb0.set_label("Ω")
        cb0.set_ticks(_colorbar_ticks(*im0.get_clim()))

        im1 = ax1.imshow(
            gen_m, cmap=cmap, interpolation=HEATMAP_INTERPOLATION, aspect="auto", origin="lower",
            vmin=0.0, vmax=plot_vmax_gen,
        )
        _panel_title(ax1, f"{label} · Generated")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label("Ω")
        cb1.set_ticks(_colorbar_ticks(*im1.get_clim()))

        if "metrics" in item:
            mrow = item["metrics"]
            pearson_r = float(mrow["pearson_r"])
            max_diff_ohm = float(mrow["max_diff_ohm"])
            metric_rows.append(mrow)
        else:
            pearson_r = float(np.corrcoef(real_vals, gen_vals)[0, 1]) if len(real_vals) > 1 else float("nan")
            max_diff_ohm = vmax_real - vmax_gen

        diff_lim = float(np.nanmax(amp_diff_m.compressed())) if amp_diff_m.count() else 1e-6
        diff_lim = max(diff_lim, 1e-6)
        im2 = ax2.imshow(
            amp_diff_m,
            cmap=diff_cmap,
            interpolation=HEATMAP_INTERPOLATION,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=diff_lim,
        )
        _panel_title(
            ax2,
            f"{label} · |Real − Gen|",
            f"r = {pearson_r:.3f}  ·  {_max_diff_caption(max_diff_ohm)}",
        )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("|Real − Gen| (Ω)")
        cb2.set_ticks(_colorbar_ticks(0.0, diff_lim, n=HEATMAP_DIFF_COLORBAR_TICKS))

    freq_note = "" if WORKFLOW == "multifreq_heatmap_sweep" else f"  [{FREQ_LABEL}]"
    fig.suptitle(
        f"Heatmap comparison — Real (sim) vs Generated{freq_note}",
        fontsize=14,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=HEATMAP_PLOT_DPI)
    plt.close(fig)
    for mrow in metric_rows:
        print_metrics_summary(mrow)
    print(f"  → saved {out_path.name}")
    return metric_rows


def _plot_impedance_comparisons(
    *,
    frequency: np.ndarray,
    target_impedance: np.ndarray,
    comparisons: list[dict],
    out_path: Path,
) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, comparisons):
        label = item["label"]
        ax.loglog(frequency, target_impedance, "--", lw=2.5, label="Target", color="red")
        ax.loglog(frequency, item["real_ohm"],         "-",  lw=1.2, label="Real",             color="royalblue")
        if (v := item.get("gen_raw_ohm"))    is not None: ax.loglog(frequency, v, "--", lw=1, color="#9E9E9E",      alpha=0.7, label="Ch0 raw")
        if (v := item.get("gen_integ_ohm"))  is not None: ax.loglog(frequency, v, "--", lw=1, color="mediumseagreen", alpha=0.7, label="∫Ch1")
        if (v := item.get("gen_integ2_ohm")) is not None: ax.loglog(frequency, v, "--", lw=1, color="darkorange",   alpha=0.7, label="∫∫Ch2")
        ax.loglog(frequency, item["gen_blended_ohm"],  "-",  lw=1.2, label="Generated (blended)", color="green")

        if (d1 := item.get("gen_derivative")) is not None and len(np.atleast_1d(d1)) == len(frequency):
            ax2 = ax.twinx()
            ax2.semilogx(frequency, np.asarray(d1).reshape(-1), ":", lw=1.2, color="darkorange", label="Derivative (z-score)")
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

    fig.suptitle(f"Impedance Profile Comparison  [{FREQ_LABEL}]", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=HEATMAP_PLOT_DPI)
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── Occupancy helpers ─────────────────────────────────────────────────────────

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
    *,
    occupancy_matrix: np.ndarray,
    out_path: Path,
    title_prefix: str,
    expected_k: int | None = None,
) -> None:
    occ = np.asarray(occupancy_matrix)
    if occ.ndim != 2 or occ.shape[1] != 52:
        raise ValueError(f"Expected occupancy_matrix shape (N,52), got {occ.shape}")

    def _active_mask(v: np.ndarray) -> np.ndarray:
        vf = np.asarray(v, dtype=float).reshape(-1)
        if ACTIVE_POLICY == "threshold" or expected_k is None:
            return vf > float(THRESHOLD)
        k = int(expected_k)
        if k <= 0:
            return np.zeros(52, dtype=bool)
        if k >= 52:
            return np.ones(52, dtype=bool)
        vf = np.where(np.isfinite(vf), vf, -np.inf)
        topk_idx = np.argsort(-vf, kind="stable")[:k]
        mask = np.zeros(52, dtype=bool)
        mask[topk_idx] = True
        return mask

    n = occ.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.2, 1.25 * n)))
    if isinstance(axes, Axes):
        axes_list: list[Axes] = [axes]
    else:
        axes_list = list(axes)

    active_face   = "#C8E6C9"
    inactive_face = "#FAFAFA"
    inactive_text = "#9E9E9E"
    edge_color    = "#BDBDBD"
    n_rows, n_cols = 4, 13
    cell_w, cell_h = 1.0, 1.0

    for i, ax in enumerate(axes_list):
        v = occ[i].reshape(-1)
        active   = _active_mask(v)
        k_active = int(active.sum())

        ax.set_xlim(0, n_cols * cell_w)
        ax.set_ylim(0, n_rows * cell_h)
        ax.set_aspect("equal")
        ax.axis("off")

        for idx in range(52):
            r = idx // n_cols
            c = idx % n_cols
            y = (n_rows - 1 - r) * cell_h
            x = c * cell_w
            is_on = bool(active[idx])

            ax.add_patch(FancyBboxPatch(
                (x, y), cell_w, cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=(active_face if is_on else inactive_face),
                edgecolor=edge_color,
                linewidth=0.9,
            ))
            ax.text(
                x + cell_w / 2, y + 0.22, f"C{idx + 1}",
                ha="center", va="center", fontsize=6.5,
                color=("#212121" if is_on else inactive_text),
            )

        policy = "topK" if ACTIVE_POLICY == "topk" and expected_k is not None else f"> {THRESHOLD:g}"
        k_note = (
            f"  (policy={policy}, active {k_active}/52"
            + (f", expected K={expected_k}" if expected_k is not None else "")
            + ")"
        )
        ax.set_title(f"{title_prefix}sample_{i}{k_note}  [{FREQ_LABEL}]", fontsize=9.5, pad=3, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=HEATMAP_PLOT_DPI)
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── Per-K runner ──────────────────────────────────────────────────────────────

def _collect_sim_metrics_single_k(
    k: int,
    *,
    repo_root: Path,
    mhz: int | None = None,
) -> list[dict]:
    """Load gen vs simulated-real heatmaps and return metric rows (no plots)."""
    k_dir = (repo_root / BASE_GENERATED_DIR / f"K{k}").resolve()
    if not k_dir.exists():
        raise SystemExit(f"K folder not found: {k_dir}")

    real_dir = k_dir / "Real"
    if not real_dir.exists():
        raise SystemExit(f"Real outputs folder not found (run move step first): {real_dir}")

    num_samples = _infer_num_samples(k_dir)
    mask = np.load(repo_root / MASK_PATH).astype(bool)
    rows: list[dict] = []

    for i in range(num_samples):
        sample_dir = k_dir / f"data_sample_{i}"
        if not sample_dir.exists():
            raise SystemExit(f"Missing sample dir: {sample_dir}")

        if WORKFLOW == "multifreq_heatmap_sweep":
            label = f"sample_{i}"
        else:
            label = f"K{k}/sample_{i}"
            if FREQ_LABEL:
                label = f"{FREQ_LABEL} {label}"

        heat_candidates = sorted(real_dir.glob(f"Heatmap_real_{i}*"))
        if not heat_candidates:
            raise SystemExit(f"No real heatmap item for sample {i} under: {real_dir}")
        gen_heat = _load_generated_heatmap(sample_dir / "heatmap_physical.npy")
        real_heat = _load_map_file(
            _choose_real_heatmap_mapfile(heat_candidates[0]),
            resolution=gen_heat.shape[0],
        )
        mrow = compute_sim_heatmap_metrics(
            real_heat,
            gen_heat,
            mask,
            mhz=mhz,
            k=k,
            sample=i,
            label=label,
            diff_tolerance=HEATMAP_DIFF_TOLERANCE,
            pattern_diff_tolerance=HEATMAP_PATTERN_DIFF_TOLERANCE,
        )
        rows.append(mrow)

    if rows:
        print(f"  K={k:2d}  metrics ({len(rows)} sample(s))")
        for mrow in rows:
            print_metrics_summary(mrow)
    return rows


def _run_single_k(k: int, *, repo_root: Path, mhz: int | None = None) -> list[dict]:
    k_dir = (repo_root / BASE_GENERATED_DIR / f"K{k}").resolve()
    if not k_dir.exists():
        raise SystemExit(f"K folder not found: {k_dir}")

    num_samples = _infer_num_samples(k_dir)
    all_metric_rows: list[dict] = []

    if RUN_HEATMAP or RUN_IMPEDANCE:
        real_dir = k_dir / "Real"
        if not real_dir.exists():
            raise SystemExit(f"Real outputs folder not found (run move_pi_to_real.py first): {real_dir}")

        mask = np.load(repo_root / MASK_PATH).astype(bool) if RUN_HEATMAP else None
        comparisons_heatmap: list[dict] = []
        comparisons_imp:     list[dict] = []

        for i in range(num_samples):
            sample_dir = k_dir / f"data_sample_{i}"
            if not sample_dir.exists():
                raise SystemExit(f"Missing sample dir: {sample_dir}")

            if WORKFLOW == "multifreq_heatmap_sweep":
                label = f"sample_{i}"
            else:
                label = f"K{k}/sample_{i}"
                if FREQ_LABEL:
                    label = f"{FREQ_LABEL} {label}"

            if RUN_HEATMAP:
                heat_candidates = sorted(real_dir.glob(f"Heatmap_real_{i}*"))
                if not heat_candidates:
                    raise SystemExit(f"No real heatmap item for sample {i} under: {real_dir}")
                gen_heat = _load_generated_heatmap(sample_dir / "heatmap_physical.npy")
                real_heat = _load_map_file(
                    _choose_real_heatmap_mapfile(heat_candidates[0]),
                    resolution=gen_heat.shape[0],
                )
                entry: dict = {"generated": gen_heat, "real": real_heat, "label": label}
                if WRITE_SIM_METRICS and mask is not None:
                    entry["metrics"] = compute_sim_heatmap_metrics(
                        real_heat,
                        gen_heat,
                        mask,
                        mhz=mhz,
                        k=k,
                        sample=i,
                        label=label,
                        diff_tolerance=HEATMAP_DIFF_TOLERANCE,
                        pattern_diff_tolerance=HEATMAP_PATTERN_DIFF_TOLERANCE,
                    )
                comparisons_heatmap.append(entry)

            if RUN_IMPEDANCE:
                frequency        = np.load(repo_root / FREQUENCY_PATH).squeeze()
                target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH).squeeze()
                gen_blended_log = _load_generated_impedance_log(sample_dir / "impedance_profile.npy")
                gen_blended_ohm = np.exp(gen_blended_log)
                gen_raw_log     = _maybe_load(sample_dir / "impedance_raw.npy")
                gen_integ_log   = _maybe_load(sample_dir / "impedance_integrated.npy")
                gen_integ2_log  = _maybe_load(sample_dir / "impedance_integrated2.npy")
                gen_d1          = _maybe_load(sample_dir / "impedance_derivative.npy")

                imp_candidates = _real_impedance_candidates(real_dir, i)
                if not imp_candidates:
                    raise SystemExit(
                        f"No real impedance item for sample {i} under: {real_dir} "
                        f"(expected Imp_Real{i}* or PI-* with CSV)"
                    )
                real_imp_ohm = _load_real_impedance(
                    _choose_real_impedance_csv(imp_candidates[0]),
                    frequency_hz=frequency,
                )

                d: dict = {"label": label, "real_ohm": real_imp_ohm, "gen_blended_ohm": gen_blended_ohm}
                if gen_raw_log    is not None: d["gen_raw_ohm"]    = np.exp(np.asarray(gen_raw_log).reshape(-1))
                if gen_integ_log  is not None: d["gen_integ_ohm"]  = np.exp(np.asarray(gen_integ_log).reshape(-1))
                if gen_integ2_log is not None: d["gen_integ2_ohm"] = np.exp(np.asarray(gen_integ2_log).reshape(-1))
                if gen_d1         is not None: d["gen_derivative"]  = np.asarray(gen_d1).reshape(-1)
                comparisons_imp.append(d)

        if RUN_HEATMAP and comparisons_heatmap:
            assert mask is not None
            print(f"  K={k:2d}  heatmap compare ({num_samples} sample(s))")
            metric_rows = _plot_heatmap_comparisons(
                comparisons=comparisons_heatmap,
                mask=mask,
                out_path=k_dir / HEATMAP_OUT_NAME,
            )
            all_metric_rows.extend(metric_rows)
        if RUN_IMPEDANCE and comparisons_imp:
            frequency        = np.load(repo_root / FREQUENCY_PATH).squeeze()
            target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH).squeeze()
            _plot_impedance_comparisons(
                frequency=frequency,
                target_impedance=target_impedance,
                comparisons=comparisons_imp,
                out_path=k_dir / IMPEDANCE_OUT_NAME,
            )

    if RUN_OCCUPANCY:
        occ = _load_occupancy_matrix(k_dir, num_samples=num_samples)
        _plot_checkboxes(occupancy_matrix=occ, out_path=k_dir / OCCUPANCY_OUT_NAME, title_prefix=f"K{k}/", expected_k=k)

    return all_metric_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def _iter_freqs() -> list[int | None]:
    if WORKFLOW == "multifreq_heatmap_sweep":
        import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

        if sweep.uses_flat_k_layout():
            return [None]
        return list(exported_freq_mhz_list())
    return _FREQ_LIST


def _refresh_multifreq_config() -> None:
    """Re-read K and OUTPUT_ROOT from sweep module (pipeline may have updated them)."""
    if WORKFLOW != "multifreq_heatmap_sweep":
        return
    import scrap.generation.run_multifreq_heatmap_sweep as sweep  # noqa: E402

    global K_MIN, K_MAX, _K_VALUES, _OUTPUT_ROOT
    _K_VALUES = exported_k_values(sweep.OUTPUT_ROOT)
    K_MIN = min(_K_VALUES)
    K_MAX = max(_K_VALUES)
    _OUTPUT_ROOT = sweep.OUTPUT_ROOT


def _iter_k_values() -> list[int]:
    if WORKFLOW == "multifreq_heatmap_sweep":
        return list(_K_VALUES)
    return list(range(K_MIN, K_MAX + 1))


def _metrics_output_root(repo_root: Path) -> Path:
    if WORKFLOW == "multifreq_heatmap_sweep":
        _refresh_multifreq_config()
        return repo_root / _OUTPUT_ROOT
    return repo_root / Path(BASE_GENERATED_DIR)


def run_sim_metrics_only() -> Path:
    """Evaluate generated vs ECADStar-simulated real heatmaps; write CSV/JSON/MD only."""
    _refresh_multifreq_config()
    if WORKFLOW == "multifreq_heatmap_sweep" and not _iter_freqs():
        raise SystemExit(
            "No sweep frequencies found. Run generate first "
            f"(expects {SWEEP_FREQ_MANIFEST} under OUTPUT_ROOT)."
        )

    repo_root = _project_root()
    os.chdir(repo_root)
    all_rows: list[dict] = []

    for mhz in _iter_freqs():
        global BASE_GENERATED_DIR, FREQ_LABEL
        BASE_GENERATED_DIR = _base_dir_for_freq(mhz)
        FREQ_LABEL = f"{mhz} MHz" if mhz is not None else "default freq"
        print(f"\n— {FREQ_LABEL} —")

        ok: list[int] = []
        failed: list[tuple[int, str]] = []
        for k in _iter_k_values():
            try:
                all_rows.extend(_collect_sim_metrics_single_k(k, repo_root=repo_root, mhz=mhz))
                ok.append(k)
            except SystemExit as e:
                failed.append((k, str(e)))
                if FAIL_FAST:
                    raise
            except Exception as e:
                failed.append((k, repr(e)))
                traceback.print_exc()
                if FAIL_FAST:
                    raise

        if failed:
            print(f"  done {len(ok)} OK, {len(failed)} failed")
            for k, msg in failed:
                print(f"    K{k}: {msg}")
        else:
            print(f"  done {len(ok)} K value(s)")

    if not all_rows:
        raise SystemExit("No sim metrics rows collected — check Real/ folders and generated heatmaps.")

    metrics_root = _metrics_output_root(repo_root)
    print(f"\nSim metrics: {len(all_rows)} rows → {metrics_root.name}/")
    _, json_path, _ = write_sim_metrics_bundle(all_rows, metrics_root)
    return json_path


def main() -> None:
    if METRICS_ONLY:
        run_sim_metrics_only()
        return

    _refresh_multifreq_config()
    if not (0 <= K_MIN <= K_MAX <= 52):
        raise SystemExit("Expected 0 <= K_MIN <= K_MAX <= 52")
    if WORKFLOW == "multifreq_heatmap_sweep":
        for k in _K_VALUES:
            if not 0 <= k <= 52:
                raise SystemExit(f"Invalid K in sweep list: {k}")

    repo_root = _project_root()
    os.chdir(repo_root)

    freq_list = _iter_freqs()
    if WORKFLOW == "multifreq_heatmap_sweep" and not freq_list:
        raise SystemExit(
            "No sweep frequencies found. Run run_multifreq_heatmap_sweep.py first "
            f"(expects {SWEEP_FREQ_MANIFEST} or generate_summary.csv under OUTPUT_ROOT)."
        )

    all_sim_metrics: list[dict] = []

    for mhz in freq_list:
        global BASE_GENERATED_DIR, FREQ_LABEL
        BASE_GENERATED_DIR = _base_dir_for_freq(mhz)
        FREQ_LABEL = f"{mhz} MHz" if mhz is not None else "default freq"
        print(f"\n— {FREQ_LABEL} — compare → {BASE_GENERATED_DIR}")

        ok: list[int] = []
        failed: list[tuple[int, str]] = []
        skipped: list[int] = []

        for k in _iter_k_values():
            try:
                rows = _run_single_k(k, repo_root=repo_root, mhz=mhz)
                all_sim_metrics.extend(rows)
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

        if failed:
            print(f"  done {len(ok)} OK, {len(failed)} failed, {len(set(skipped))} skipped")
            for k, msg in failed:
                print(f"    K{k}: {msg}")
        else:
            print(f"  done {len(ok)} K value(s)")

    if WRITE_SIM_METRICS and all_sim_metrics:
        if WORKFLOW == "multifreq_heatmap_sweep":
            _refresh_multifreq_config()
            metrics_root = _project_root() / _OUTPUT_ROOT
        else:
            metrics_root = _project_root() / Path(BASE_GENERATED_DIR)
        print(f"\nSim metrics: {len(all_sim_metrics)} rows → {metrics_root.name}/")
        write_sim_metrics_bundle(all_sim_metrics, metrics_root)


if __name__ == "__main__":
    main()
