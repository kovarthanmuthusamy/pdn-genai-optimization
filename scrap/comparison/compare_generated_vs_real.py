"""Compare generated vs real for one K.

Purpose:
    Plot heatmap and impedance overlays for one K folder (generated ``.npy`` vs ECADStar ``Real/`` exports).

Run:
    python scrap/comparison/compare_generated_vs_real.py

Agent notes:
    - What: Visual QA — generated VAE samples vs ground-truth PI simulation for a single decap budget.
    - Usage: Set ``K_VALUE`` and ``BASE_GENERATED_DIR`` → run. Also see ``comparison/compare.py``.
    - Config keys:
        - ``K_VALUE`` — decap budget folder ``K{n}`` under base dir
        - ``BASE_GENERATED_DIR`` — root containing ``K1/``, ``K2/``, …
        - ``NUM_SAMPLES`` — how many ``data_sample_*`` to plot
        - ``FREQUENCY_PATH``, ``TARGET_IMPEDANCE_PATH``, ``MASK_PATH`` — shared config arrays
        - ``HEATMAP_OUT_NAME``, ``IMPEDANCE_OUT_NAME`` — output PNG filenames
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import pearsonr


# =============================================================================
# CONFIGURATION — edit these before running: python scrap/comparison/compare_generated_vs_real.py
# =============================================================================
K_VALUE = 5
BASE_GENERATED_DIR = Path("scrap/generated_samples_v2")

# If None, infer from data_sample_* folders.
NUM_SAMPLES: int | None = None

# Assets
FREQUENCY_PATH = Path("configs/Frequency_data_hz.npy")
TARGET_IMPEDANCE_PATH = Path("configs/target_impedance.npy")
MASK_PATH = Path("configs/binary_mask.npy")

# Outputs (saved under the K folder)
HEATMAP_OUT_NAME = "generated_vs_real_heatmap.png"
IMPEDANCE_OUT_NAME = "generated_vs_real_impedance_profile.png"

# Heatmap loader settings
HEATMAP_CMAP = "jet"
HEATMAP_LEVELS = 22
# Differences smaller than this (Ohms, physical space) are treated as zero in the diff panel.
# Suppresses colour noise from interpolation / grid-alignment artefacts.
HEATMAP_DIFF_TOLERANCE = 0.25   # Ohms
# Pattern diff: suppress noise below this (normalised [0,1] space)
HEATMAP_PATTERN_DIFF_TOLERANCE = 0.05

# Real heatmap map file chooser (inside the moved Heatmap_real_* directory)
MAP_GLOB_PREFERENCE = ("Z_*.map", "*.map")

# Real impedance CSV chooser (inside the moved Imp_Real* directory)
IMPEDANCE_CSV_GLOB_PREFERENCE = ("*PIPinZ*.csv", "*.csv")

# =============================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _infer_num_samples(k_dir: Path) -> int:
    sample_dirs = [p for p in k_dir.glob("data_sample_*") if p.is_dir()]
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* folders found in: {k_dir}")

    def idx(p: Path) -> int | None:
        name = p.name
        if not name.startswith("data_sample_"):
            return None
        try:
            return int(name.split("_")[-1])
        except Exception:
            return None

    indices = sorted(i for i in (idx(p) for p in sample_dirs) if i is not None)
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


def _load_generated_heatmap(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected heatmap to be 2D after channel squeeze, got shape {data.shape} from {path}")
    return data


def _load_map_file(file_path: Path, *, resolution: int) -> np.ndarray:
    """Load real heatmap from a .map file and interpolate to a square grid."""
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
                    z.append(max(pz, 0.01))  # avoid zeros
                i += 3
            except Exception:
                pass
        i += 1

    if not x:
        raise RuntimeError(f"No data loaded from MAP file: {file_path}")

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)

    # Scale to mm (matches previous script convention)
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


def _choose_real_heatmap_mapfile(heatmap_real_item: Path) -> Path:
    """Given Heatmap_real_{i}* item, return the concrete .map file to read."""
    if heatmap_real_item.is_file() and heatmap_real_item.suffix.lower() == ".map":
        return heatmap_real_item

    if heatmap_real_item.is_dir():
        for pattern in MAP_GLOB_PREFERENCE:
            candidates = sorted(heatmap_real_item.rglob(pattern))
            if candidates:
                return candidates[0]

    raise FileNotFoundError(f"Could not locate a .map file under: {heatmap_real_item}")


def _choose_real_impedance_csv(imp_real_item: Path) -> Path:
    """Given Imp_Real{i}* item, return the concrete .csv file to read."""
    if imp_real_item.is_file() and imp_real_item.suffix.lower() == ".csv":
        return imp_real_item

    if imp_real_item.is_dir():
        for pattern in IMPEDANCE_CSV_GLOB_PREFERENCE:
            candidates = sorted(imp_real_item.rglob(pattern))
            if candidates:
                return candidates[0]

    raise FileNotFoundError(f"Could not locate a .csv file under: {imp_real_item}")


def _iter_numeric_rows(path: Path) -> Iterable[tuple[float, float]]:
    """Yield (x, y) pairs from a CSV/TSV/space-delimited file.

    Many ECADStar exports are not strict CSV (often tab/space separated, with headers).
    This parser is delimiter-agnostic: it extracts the first two numeric columns per line.
    """
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue

            # Normalize delimiters: treat commas/semicolons as whitespace
            s = s.replace(",", " ").replace(";", " ")
            parts = s.split()
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except Exception:
                continue

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            yield x, y


def _load_real_impedance(csv_path: Path, *, frequency_hz: np.ndarray) -> np.ndarray:
    """Load real impedance in Ohms; interpolate to `frequency_hz` if frequency column exists."""
    rows = list(_iter_numeric_rows(csv_path))
    if not rows:
        raise RuntimeError(f"No numeric data found in: {csv_path}")

    x = np.asarray([r[0] for r in rows], dtype=float)
    z = np.asarray([r[1] for r in rows], dtype=float)

    # If x looks like frequency (monotonic-ish and in same scale), interpolate in log-log space.
    # Otherwise assume z is already aligned with frequency and just return z.
    if len(z) == len(frequency_hz):
        return z

    x_pos = x[(x > 0) & np.isfinite(x)]
    z_pos = z[(z > 0) & np.isfinite(z)]
    if len(x_pos) < 10 or len(z_pos) < 10:
        return z

    # Try monotonic sorting for interpolation.
    order = np.argsort(x)
    x_sorted = x[order]
    z_sorted = z[order]

    # Guard: require positive for log interpolation
    good = (x_sorted > 0) & (z_sorted > 0)
    x_sorted = x_sorted[good]
    z_sorted = z_sorted[good]

    if len(x_sorted) < 10:
        return z

    # Interpolate in log-log space onto requested frequency grid
    fx = np.log10(frequency_hz.astype(float))
    xi = np.log10(x_sorted.astype(float))
    zi = np.log10(z_sorted.astype(float))
    z_out = 10 ** np.interp(fx, xi, zi)
    return z_out


def _load_generated_impedance_log(path: Path) -> np.ndarray:
    data = np.load(path).squeeze()
    if data.ndim != 1:
        data = data.reshape(-1)
    return data


def _maybe_load(path: Path) -> np.ndarray | None:
    return np.load(path).squeeze() if path.exists() else None


def _plot_heatmap_comparisons(
    *,
    comparisons: list[dict],
    mask: np.ndarray,
    out_path: Path,
) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = np.array([axes])

    cmap = mpl.colormaps[HEATMAP_CMAP].resampled(HEATMAP_LEVELS).copy()
    cmap.set_bad("white")

    for row_idx, item in enumerate(comparisons):
        real = item["real"]
        gen = item["generated"]
        label = item["label"]

        real_m = np.ma.masked_where(~mask, real)
        gen_m = np.ma.masked_where(~mask, gen)
        diff = np.abs(real - gen)
        diff = np.where(diff < HEATMAP_DIFF_TOLERANCE, 0.0, diff)   # suppress sub-tolerance noise
        diff_m = np.ma.masked_where(~mask, diff)

        # Pattern diff: min-max normalise each map within the mask independently,
        # then difference — captures structural/spatial similarity free of scale.
        real_vals = real[mask]
        gen_vals = gen[mask]
        real_norm = np.zeros_like(real, dtype=float)
        gen_norm = np.zeros_like(gen, dtype=float)
        real_norm[mask] = (real_vals - real_vals.min()) / max(real_vals.max() - real_vals.min(), 1e-12)
        gen_norm[mask] = (gen_vals - gen_vals.min()) / max(gen_vals.max() - gen_vals.min(), 1e-12)
        pattern_diff = np.abs(real_norm - gen_norm)
        pattern_diff[mask] = np.where(
            pattern_diff[mask] < HEATMAP_PATTERN_DIFF_TOLERANCE, 0.0, pattern_diff[mask]
        )
        pattern_diff_m = np.ma.masked_where(~mask, pattern_diff)

        vmax_real = float(max(real_m.max(), 1e-12))
        vmax_gen = float(max(gen_m.max(), 1e-12))

        ax0, ax1, ax2, ax3 = axes[row_idx]

        im0 = ax0.imshow(
            real_m,
            cmap=cmap,
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax_real,
        )
        ax0.set_title(f"{label} - Real", fontsize=12)
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        vmin0, vmax0 = im0.get_clim()
        cb0.set_ticks(np.linspace(float(vmin0), float(vmax0), 6).tolist())

        im1 = ax1.imshow(
            gen_m,
            cmap=cmap,
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax_gen,
        )
        ax1.set_title(f"{label} - Generated", fontsize=12)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        vmin1, vmax1 = im1.get_clim()
        cb1.set_ticks(np.linspace(float(vmin1), float(vmax1), 6).tolist())

        im2 = ax2.imshow(
            diff_m,
            cmap="hot",
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax2.set_title(f"{label} - |Real - Gen|", fontsize=12)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("Abs Diff")
        cb2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

        mae = float(np.mean(diff_m.compressed())) if diff_m.count() else float("nan")
        pattern_mae = float(np.mean(pattern_diff_m.compressed())) if pattern_diff_m.count() else float("nan")
        if len(real_vals) > 1:
            pearson_r = float(np.corrcoef(real_vals, gen_vals)[0, 1])
        else:
            pearson_r = float("nan")
        print(
            f"{label}: heatmap MAE={mae:.6f}, pattern MAE={pattern_mae:.4f}, "
            f"Pearson r={pearson_r:.4f}, real_vmax={vmax_real:.6f}, gen_vmax={vmax_gen:.6f}"
        )

        im3 = ax3.imshow(
            pattern_diff_m,
            cmap="RdYlGn_r",
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax3.set_title(f"{label} - Pattern Diff", fontsize=12)
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label("|norm(Real) − norm(Gen)|")
        cb3.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle("Heatmap Comparison: Real vs Generated", fontsize=16, y=1.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap comparison plot: {out_path}")


def _plot_impedance_comparisons(
    *,
    frequency: np.ndarray,
    target_impedance: np.ndarray,
    comparisons: list[dict],
    out_path: Path,
) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, comparisons):
        label = item["label"]
        real = item["real_ohm"]
        gen_blended = item["gen_blended_ohm"]

        ax.loglog(frequency, target_impedance, "--", lw=2.5, label="Target", color="red")
        ax.loglog(frequency, real, "-", lw=1.2, label="Real", color="royalblue")

        # exp027-style optional curves
        if (raw := item.get("gen_raw_ohm")) is not None:
            ax.loglog(frequency, raw, "--", lw=1, color="#9E9E9E", alpha=0.7, label="Ch0 raw")
        if (integ := item.get("gen_integ_ohm")) is not None:
            ax.loglog(frequency, integ, "--", lw=1, color="mediumseagreen", alpha=0.7, label="∫Ch1")
        if (integ2 := item.get("gen_integ2_ohm")) is not None:
            ax.loglog(frequency, integ2, "--", lw=1, color="darkorange", alpha=0.7, label="∫∫Ch2")

        ax.loglog(frequency, gen_blended, "-", lw=2.5, color="green", label="Generated (blended)")

        # Optional derivative overlay (from older compare_impedance script)
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

    fig.suptitle("Impedance Profile Comparison", fontsize=16, y=1.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved impedance comparison plot: {out_path}")


def main() -> None:
    repo_root = _project_root()
    os.chdir(repo_root)

    k_dir = (repo_root / BASE_GENERATED_DIR / f"K{K_VALUE}").resolve()
    if not k_dir.exists():
        raise SystemExit(f"K folder not found: {k_dir}")

    real_dir = k_dir / "Real"
    if not real_dir.exists():
        raise SystemExit(f"Real outputs folder not found (run move_pi_to_real.py first): {real_dir}")

    num_samples = NUM_SAMPLES if NUM_SAMPLES is not None else _infer_num_samples(k_dir)

    frequency = np.load(repo_root / FREQUENCY_PATH).squeeze()
    target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH).squeeze()
    mask = np.load(repo_root / MASK_PATH).astype(bool)

    print(f"K folder: {k_dir}")
    print(f"Samples:  {num_samples}")

    comparisons_heatmap: list[dict] = []
    comparisons_imp: list[dict] = []

    for i in range(num_samples):
        sample_dir = k_dir / f"data_sample_{i}"
        if not sample_dir.exists():
            raise SystemExit(f"Missing sample dir: {sample_dir}")

        # Generated
        gen_heat = _load_generated_heatmap(sample_dir / "heatmap_physical.npy")
        gen_blended_log = _load_generated_impedance_log(sample_dir / "impedance_profile.npy")
        gen_blended_ohm = np.exp(gen_blended_log)

        gen_raw_log = _maybe_load(sample_dir / "impedance_raw.npy")
        gen_integ_log = _maybe_load(sample_dir / "impedance_integrated.npy")
        gen_integ2_log = _maybe_load(sample_dir / "impedance_integrated2.npy")
        gen_d1 = _maybe_load(sample_dir / "impedance_derivative.npy")

        # Real: impedance
        imp_candidates = sorted(real_dir.glob(f"Imp_Real{i}*"))
        if not imp_candidates:
            raise SystemExit(f"No real impedance item found for sample {i} under: {real_dir}")
        imp_item = imp_candidates[0]
        real_imp_csv = _choose_real_impedance_csv(imp_item)
        real_imp_ohm = _load_real_impedance(real_imp_csv, frequency_hz=frequency)

        # Real: heatmap
        heat_candidates = sorted(real_dir.glob(f"Heatmap_real_{i}*"))
        if not heat_candidates:
            raise SystemExit(f"No real heatmap item found for sample {i} under: {real_dir}")
        heat_item = heat_candidates[0]
        map_path = _choose_real_heatmap_mapfile(heat_item)
        real_heat = _load_map_file(map_path, resolution=gen_heat.shape[0])

        label = f"K{K_VALUE}/sample_{i}"

        comparisons_heatmap.append({"generated": gen_heat, "real": real_heat, "label": label})

        d = {
            "label": label,
            "real_ohm": real_imp_ohm,
            "gen_blended_ohm": gen_blended_ohm,
        }
        if gen_raw_log is not None:
            d["gen_raw_ohm"] = np.exp(np.asarray(gen_raw_log).reshape(-1))
        if gen_integ_log is not None:
            d["gen_integ_ohm"] = np.exp(np.asarray(gen_integ_log).reshape(-1))
        if gen_integ2_log is not None:
            d["gen_integ2_ohm"] = np.exp(np.asarray(gen_integ2_log).reshape(-1))
        if gen_d1 is not None:
            d["gen_derivative"] = np.asarray(gen_d1).reshape(-1)

        comparisons_imp.append(d)

    # Save plots
    heatmap_out = k_dir / HEATMAP_OUT_NAME
    impedance_out = k_dir / IMPEDANCE_OUT_NAME

    _plot_heatmap_comparisons(comparisons=comparisons_heatmap, mask=mask, out_path=heatmap_out)
    _plot_impedance_comparisons(
        frequency=frequency,
        target_impedance=target_impedance,
        comparisons=comparisons_imp,
        out_path=impedance_out,
    )


if __name__ == "__main__":
    main()
