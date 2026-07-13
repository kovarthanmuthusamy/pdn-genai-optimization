#!/usr/bin/env python3
"""Temp check: low-value skew in data_multifreq_gmax normalized heatmaps.

Run:
    .venv/bin/python scratch/check_gmax_low_value_skew.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from repo_paths import REPO_ROOT, setup_path

setup_path()

DATA_DIR = REPO_ROOT / "datasets/data_multifreq_gmax"
OUT_JSON = REPO_ROOT / "scratch/gmax_low_value_skew_report.json"
SAMPLE_PER_MHZ = 120
SEED = 42
LOW_BINS = [0.0, 0.0005, 0.001, 0.0016, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.02]


def _skew(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - m) ** 3).mean() / (s**3))


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def main() -> None:
    stats = json.loads((DATA_DIR / "normalization_stats.json").read_text(encoding="utf-8"))
    hm_stats = stats["Heatmap"]
    gmax = float(hm_stats["global_max_ohm"])
    fg_thr_stats = float(hm_stats["fg_norm_threshold"])
    bg_ohm = float(hm_stats.get("bg_ohm_threshold", 0.05))
    cfg_thr = 0.0016  # from exp043 config.yaml heatmap_fg_threshold

    # Group manifest rows by rounded MHz
    by_mhz: dict[float, list[str]] = defaultdict(list)
    with (DATA_DIR / "manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mhz = round(float(row["freq_mhz"]), 1)
            stem = Path(row["sample_name"]).stem
            by_mhz[mhz].append(stem)

    rng = random.Random(SEED)
    sampled: list[tuple[float, str]] = []
    for mhz in sorted(by_mhz):
        pool = by_mhz[mhz]
        k = min(SAMPLE_PER_MHZ, len(pool))
        for stem in rng.sample(pool, k):
            sampled.append((mhz, stem))

    print(f"Dataset: {DATA_DIR}")
    print(f"gmax={gmax:.4f} Ω  bg_ohm={bg_ohm}  fg_thr_stats={fg_thr_stats:.6f}  cfg_thr={cfg_thr}")
    print(f"Sampling {len(sampled)} heatmaps across {len(by_mhz)} MHz bins …\n")

    all_pix: list[float] = []
    fg_stats: list[float] = []
    fg_stats_thr: list[float] = []
    fg_cfg: list[float] = []
    zero_frac_per_map: list[float] = []
    clipped_by_cfg_per_map: list[float] = []
    hist_all = np.zeros(len(LOW_BINS) - 1, dtype=np.int64)
    hist_fg = np.zeros(len(LOW_BINS) - 1, dtype=np.int64)

    per_mhz: dict[str, dict] = {}

    for mhz, stem in sampled:
        arr = np.load(DATA_DIR / "heatmap" / f"{stem}.npy", mmap_mode="r").astype(np.float64)
        flat = arr.reshape(-1)
        all_pix.append(flat)

        fg_mask_stats = flat > fg_thr_stats
        fg_mask_cfg = flat > cfg_thr
        n = flat.size
        zero_frac_per_map.append(float((flat == 0.0).mean()))

        if fg_mask_stats.any():
            fg_vals = flat[fg_mask_stats]
            fg_stats.append(fg_vals)
            clipped = fg_vals[(fg_vals > fg_thr_stats) & (fg_vals <= cfg_thr)]
            clipped_by_cfg_per_map.append(float(clipped.size / max(fg_mask_stats.sum(), 1)))

        if fg_mask_cfg.any():
            fg_cfg.append(flat[fg_mask_cfg])

        # per-bin counts (FG by stats threshold)
        if fg_mask_stats.any():
            fg_only = flat[fg_mask_stats]
            h, _ = np.histogram(fg_only, bins=LOW_BINS)
            hist_fg += h
        h_all, _ = np.histogram(flat, bins=LOW_BINS)
        hist_all += h_all

    all_cat = np.concatenate(all_pix)
    fg_cat_stats = np.concatenate(fg_stats) if fg_stats else np.array([])
    fg_cat_cfg = np.concatenate(fg_cfg) if fg_cfg else np.array([])

    def summarize(name: str, x: np.ndarray) -> dict:
        if x.size == 0:
            return {"n": 0}
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        return {
            "n": int(x.size),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "skew": _skew(x),
            "pct": {f"p{q}": _pct(x, q) for q in qs},
        }

    # Per-MHz low-tail (p10, p50, p90) on FG (stats thr)
    for mhz in sorted(by_mhz):
        stems = [s for m, s in sampled if m == mhz]
        if not stems:
            continue
        vals = []
        for stem in stems:
            flat = np.load(DATA_DIR / "heatmap" / f"{stem}.npy", mmap_mode="r").reshape(-1)
            fg = flat > fg_thr_stats
            if fg.any():
                vals.append(flat[fg])
        if not vals:
            continue
        v = np.concatenate(vals)
        per_mhz[str(mhz)] = {
            "n_maps": len(stems),
            "fg_pixels": int(v.size),
            "p10": _pct(v, 10),
            "p50": _pct(v, 50),
            "p90": _pct(v, 90),
            "p99": _pct(v, 99),
            "mean": float(v.mean()),
            "frac_below_cfg_thr": float((v <= cfg_thr).mean()),
        }

    n_fg_stats = int(fg_cat_stats.size)
    n_between = int(((fg_cat_stats > fg_thr_stats) & (fg_cat_stats <= cfg_thr)).sum()) if n_fg_stats else 0

    report = {
        "dataset": str(DATA_DIR),
        "global_max_ohm": gmax,
        "bg_ohm_threshold": bg_ohm,
        "fg_norm_threshold_stats": fg_thr_stats,
        "fg_norm_threshold_config": cfg_thr,
        "n_sampled_maps": len(sampled),
        "all_pixels": summarize("all", all_cat),
        "fg_pixels_stats_thr": summarize("fg_stats", fg_cat_stats),
        "fg_pixels_config_thr": summarize("fg_cfg", fg_cat_cfg),
        "zero_pixel_frac_mean_per_map": float(np.mean(zero_frac_per_map)),
        "fg_pixels_between_stats_and_cfg": {
            "count": n_between,
            "frac_of_fg_stats": float(n_between / max(n_fg_stats, 1)),
            "physical_ohm_range": [
                float(fg_thr_stats * gmax),
                float(cfg_thr * gmax),
            ],
        },
        "histogram_bins_edges": LOW_BINS,
        "histogram_all_pixels": hist_all.tolist(),
        "histogram_fg_stats_thr": hist_fg.tolist(),
        "per_mhz_fg": per_mhz,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary
    print("=" * 72)
    print("ALL PIXELS (incl. zeros / mask)")
    s = report["all_pixels"]
    print(f"  mean={s['mean']:.6f}  skew={s['skew']:.2f}  "
          f"p50={s['pct']['p50']:.6f}  p90={s['pct']['p90']:.6f}  max={s['max']:.4f}")
    print(f"  exact-zero fraction (mean per map): {report['zero_pixel_frac_mean_per_map']:.1%}")

    print("\nFG PIXELS (stats thr = phys > {:.4f} Ω)".format(bg_ohm))
    s = report["fg_pixels_stats_thr"]
    print(f"  n={s['n']:,}  mean={s['mean']:.6f}  skew={s['skew']:.2f}")
    for q in (1, 5, 10, 25, 50, 90, 99):
        print(f"    p{q:>4} = {s['pct'][f'p{q}']:.6f}  ({s['pct'][f'p{q}'] * gmax:.4f} Ω)")

    print("\nFG PIXELS (config thr = {:.4f} norm ≈ {:.4f} Ω)".format(cfg_thr, cfg_thr * gmax))
    s2 = report["fg_pixels_config_thr"]
    print(f"  n={s2['n']:,}  mean={s2['mean']:.6f}  skew={s2['skew']:.2f}")
    for q in (1, 5, 10, 25, 50, 90, 99):
        print(f"    p{q:>4} = {s2['pct'][f'p{q}']:.6f}")

    gap = report["fg_pixels_between_stats_and_cfg"]
    print(f"\n⚠ FG pixels ZEROED by training prep (stats_thr < v <= cfg_thr):")
    print(f"  {gap['count']:,} pixels ({gap['frac_of_fg_stats']:.1%} of FG)")
    print(f"  physical range: {gap['physical_ohm_range'][0]:.4f} – {gap['physical_ohm_range'][1]:.4f} Ω")

    print("\nFG histogram (stats threshold) — bin edges:", LOW_BINS)
    total_fg = hist_fg.sum()
    for i in range(len(LOW_BINS) - 1):
        lo, hi = LOW_BINS[i], LOW_BINS[i + 1]
        frac = hist_fg[i] / max(total_fg, 1)
        bar = "#" * int(frac * 50)
        print(f"  [{lo:.4f}, {hi:.4f}): {frac:6.2%}  {bar}")

    print("\nPer-MHz FG p10 / p50 / p90 (norm):")
    for mhz, d in sorted(per_mhz.items(), key=lambda x: float(x[0])):
        print(
            f"  {mhz:>6} MHz  p10={d['p10']:.5f}  p50={d['p50']:.5f}  "
            f"p90={d['p90']:.5f}  below_cfg={d['frac_below_cfg_thr']:.1%}"
        )

    print(f"\nFull report → {OUT_JSON}")


if __name__ == "__main__":
    main()
