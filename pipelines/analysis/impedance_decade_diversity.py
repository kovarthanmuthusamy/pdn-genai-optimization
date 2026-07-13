#!/usr/bin/env python3
"""Per-frequency-bin impedance diversity across layouts (231 PI-spectrum bins).

X-axis: ``configs/Frequency_data_hz.npy`` (Hz) — same as PI-spectrum / impedance plots.
Y-axis: magnitude |Z| stored in ``layouts/<design_id>/imp.npy``.

For each of the 231 bins, measures layout-to-layout spread (std, IQR, CV).

Run:
    python pipelines/analysis/impedance_decade_diversity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import repo_path, setup_path

setup_path()

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = repo_path("datasets", "data_multifreq_train")
FREQ_PATH = repo_path("configs", "Frequency_data_hz.npy")
REPORT_JSON = repo_path("experiments", "impedance_freq_diversity.json")
REPORT_NPZ = repo_path("experiments", "impedance_freq_diversity.npz")

MAX_LAYOUTS: int | None = None  # None = all layouts
SEED = 0
TOP_N = 25  # print top-N highest-diversity bins
USE_LOG10_MAGNITUDE = True  # primary rank on log10|Z|; also saves linear-ohm stats

# =============================================================================


def _load_imp(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r").astype(np.float64).reshape(-1)
    if x.size != 231:
        raise ValueError(f"{path}: expected 231 impedance points, got {x.size}")
    return x


def _iter_layout_imps(layouts_dir: Path, *, max_layouts: int | None, seed: int):
    paths = sorted(p / "imp.npy" for p in layouts_dir.iterdir() if (p / "imp.npy").is_file())
    if not paths:
        raise FileNotFoundError(f"No layouts/*/imp.npy under {layouts_dir}")
    if max_layouts is not None and len(paths) > max_layouts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=max_layouts, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    imps = np.stack([_load_imp(p) for p in paths], axis=0)  # (n_layouts, 231)
    return imps, paths


def analyze_per_bin_diversity(imps: np.ndarray, freq_hz: np.ndarray) -> dict:
    """Diversity at each of the 231 frequency bins across layouts."""
    z = np.clip(imps, 1e-12, None)
    logz = np.log10(z)

    std_ohm = z.std(axis=0)
    std_log10 = logz.std(axis=0)
    mean_ohm = z.mean(axis=0)
    median_ohm = np.median(z, axis=0)
    cv_ohm = std_ohm / np.clip(mean_ohm, 1e-12, None)

    q75 = np.percentile(z, 75, axis=0)
    q25 = np.percentile(z, 25, axis=0)
    iqr_ohm = q75 - q25

    q75_log = np.percentile(logz, 75, axis=0)
    q25_log = np.percentile(logz, 25, axis=0)
    iqr_log10 = q75_log - q25_log

    p10_ohm = np.percentile(z, 10, axis=0)
    p90_ohm = np.percentile(z, 90, axis=0)
    range_p90_p10_ohm = p90_ohm - p10_ohm

    rank_key = std_log10 if USE_LOG10_MAGNITUDE else std_ohm
    order = np.argsort(-rank_key)

    freq_mhz = freq_hz / 1e6
    per_bin = []
    for i in range(freq_hz.size):
        per_bin.append(
            {
                "bin": int(i),
                "freq_hz": float(freq_hz[i]),
                "freq_mhz": float(freq_mhz[i]),
                "std_log10": float(std_log10[i]),
                "std_ohm": float(std_ohm[i]),
                "iqr_log10": float(iqr_log10[i]),
                "iqr_ohm": float(iqr_ohm[i]),
                "cv_ohm": float(cv_ohm[i]),
                "mean_ohm": float(mean_ohm[i]),
                "median_ohm": float(median_ohm[i]),
                "p10_ohm": float(p10_ohm[i]),
                "p90_ohm": float(p90_ohm[i]),
                "range_p90_p10_ohm": float(range_p90_p10_ohm[i]),
            }
        )

    ranked = [per_bin[int(i)] for i in order]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    return {
        "freq_hz": freq_hz,
        "freq_mhz": freq_mhz,
        "std_log10": std_log10,
        "std_ohm": std_ohm,
        "iqr_log10": iqr_log10,
        "iqr_ohm": iqr_ohm,
        "cv_ohm": cv_ohm,
        "mean_ohm": mean_ohm,
        "median_ohm": median_ohm,
        "range_p90_p10_ohm": range_p90_p10_ohm,
        "per_bin": per_bin,
        "ranked_bins": ranked,
        "rank_metric": "std_log10" if USE_LOG10_MAGNITUDE else "std_ohm",
    }


def print_report(stats: dict, *, n_layouts: int, top_n: int) -> None:
    metric = stats["rank_metric"]
    print(f"\nPer-bin impedance diversity ({n_layouts:,} layouts, 231 frequency bins)")
    print(f"Rank metric: {metric}  (layout-to-layout spread at each exact frequency)")
    print(f"{'rank':<5} {'bin':>4} {'freq_MHz':>10} {'std_log10':>10} {'std_ohm':>10} {'iqr_log10':>10}")
    print("-" * 55)
    for row in stats["ranked_bins"][:top_n]:
        print(
            f"{row['rank']:<5} {row['bin']:>4} {row['freq_mhz']:>10.3f} "
            f"{row['std_log10']:>10.4f} {row['std_ohm']:>10.4f} {row['iqr_log10']:>10.4f}"
        )

    # summary bands
    mhz = stats["freq_mhz"]
    s = stats["std_log10"]
    for lo, hi, label in [(1, 10, "1–10 MHz"), (10, 100, "10–100 MHz"), (100, 600, "100–600 MHz")]:
        m = (mhz >= lo) & (mhz <= hi)
        if m.any():
            print(f"  {label}: mean std_log10 = {s[m].mean():.4f}, max = {s[m].max():.4f} @ {mhz[m][np.argmax(s[m])]:.3f} MHz")


def main() -> None:
    layouts_dir = Path(DATA_ROOT) / "layouts"
    if not layouts_dir.is_dir():
        raise SystemExit(f"Missing layouts dir: {layouts_dir}")

    freq_hz = np.load(FREQ_PATH).astype(np.float64).reshape(-1)
    if freq_hz.size != 231:
        raise SystemExit(f"Expected 231 frequencies in {FREQ_PATH}, got {freq_hz.size}")

    print(f"Data root : {DATA_ROOT}")
    print(f"Layouts   : {layouts_dir}")
    print(f"Frequency : {FREQ_PATH}  ({freq_hz.min()/1e6:g}–{freq_hz.max()/1e6:g} MHz, {freq_hz.size} bins)")
    print(f"Sample cap: {MAX_LAYOUTS or 'all'}")

    imps, paths = _iter_layout_imps(layouts_dir, max_layouts=MAX_LAYOUTS, seed=SEED)
    stats = analyze_per_bin_diversity(imps, freq_hz)
    print_report(stats, n_layouts=len(paths), top_n=TOP_N)

    report = {
        "data_root": str(DATA_ROOT),
        "freq_path": str(FREQ_PATH),
        "n_layouts": len(paths),
        "max_layouts": MAX_LAYOUTS,
        "rank_metric": stats["rank_metric"],
        "per_bin": stats["per_bin"],
        "top_bins": stats["ranked_bins"][:TOP_N],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    np.savez_compressed(
        REPORT_NPZ,
        freq_hz=stats["freq_hz"],
        freq_mhz=stats["freq_mhz"],
        std_log10=stats["std_log10"],
        std_ohm=stats["std_ohm"],
        iqr_log10=stats["iqr_log10"],
        iqr_ohm=stats["iqr_ohm"],
        cv_ohm=stats["cv_ohm"],
        mean_ohm=stats["mean_ohm"],
        median_ohm=stats["median_ohm"],
        range_p90_p10_ohm=stats["range_p90_p10_ohm"],
    )

    print(f"\nWrote {REPORT_JSON}")
    print(f"Wrote {REPORT_NPZ}  (arrays length 231, plot std_log10 vs freq_mhz)")


if __name__ == "__main__":
    main()
