"""Normalize a raw store per frequency bin: maps by median / IQR of log(1 + ohm), spectra by log z-score.

What: raw store (processing / merge output) -> training-ready store.
        map       log(1 + ohm) on board pixels -> (x - median) / IQR of the sample's frequency bin, not
                  clipped; off-board pixels set to the bin's background value (its 0.5th-percentile z
                  minus 1.5). Mode name: robust_log1p_per_mhz_unbounded.
        spectrum  log(ohm) -> (x - mean) / std, one pair over every point of every placement.
        placement type codes copied (type.npy int8, occ.npy float32).
    Bins are BIN_WIDTH_MHZ wide; below LOW_BAND_MAX_MHZ their centres sit every LOW_BAND_STEP_MHZ
    (1.0, 1.5, ... 10.0 MHz), because the board's map level changes about 15-fold between 1 and 10 MHz.
    A bin with fewer than MIN_SAMPLES_PER_BIN samples gets no statistics of its own. Every map is
    normalized with the statistics interpolated in log10(MHz) between the two bracketing bin
    centres -- the same reader the model uses (stats.HeatmapNormStats.bin_stats(interp=True)).
Usage: python -m thesis_code.s03_normalization.normalize
    Writes OUTPUT_DIR/s03_normalization/normalized/ (heatmap/, PI_freq/, layouts/, manifest.csv,
    normalization_stats.json, dataset_meta.json). With STATS_JSON set, the given statistics are
    applied instead of fitted (as for the dataset's test pool).
Config keys: INPUT_DIR, STATS_JSON, BIN_WIDTH_MHZ, LOW_BAND_MAX_MHZ, LOW_BAND_STEP_MHZ, MIN_SAMPLES_PER_BIN.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.paths import out_dir
from thesis_code.s03_normalization.stats import NORM_MODE, load_norm_stats

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
INPUT_DIR = settings.DATA_DIR / "raw"     # raw store (merge output); the dataset's raw/ by default
STATS_JSON: Path | None = None            # None = fit the statistics on INPUT_DIR
BIN_WIDTH_MHZ = 10.0
LOW_BAND_MAX_MHZ = 10.0
LOW_BAND_STEP_MHZ = 0.5
MIN_SAMPLES_PER_BIN = 20
MAX_PIXELS_PER_BIN = 40_000_000           # stop adding samples to a bin past this many board pixels
# =============================================================================

IMPEDANCE_NORM_MODE = "log_mean_std"
IQR_EPS = 1e-4                      # floor on the IQR of a frequency bin
CLIP_PERCENTILES = (0.5, 99.5)      # recorded per bin as clip_min / clip_max (not applied)
BACKGROUND_OFFSET = 1.5             # background = clip_min - 1.5
IMPEDANCE_FLOOR_OHM = 1e-10
N_DECAP_TYPES = 3


def mhz_key(mhz: float) -> str:
    """Key of a frequency bin in ``by_mhz`` (one decimal, e.g. "155.0")."""
    return f"{float(mhz):.1f}"


def board_log_pixels(raw: np.ndarray) -> np.ndarray:
    """log(1 + ohm) of the on-board pixels of one raw (2, H, W) map."""
    return np.log1p(raw[0].astype(np.float32)[raw[1] == 1])


def robust_bin_stats(log_fg: np.ndarray) -> dict:
    """Median / IQR of one frequency bin's board pixels, and the derived clip and background values."""
    med = float(np.median(log_fg))
    q25, q75 = np.percentile(log_fg, [25, 75])
    iqr = max(float(q75 - q25), IQR_EPS)
    z = (log_fg - med) / iqr
    clip_min = float(np.percentile(z, CLIP_PERCENTILES[0]))
    clip_max = float(np.percentile(z, CLIP_PERCENTILES[1]))
    return {"median": med, "iqr": iqr, "clip_min": clip_min, "clip_max": clip_max,
            "background_value": round(clip_min - BACKGROUND_OFFSET, 4), "fg_pixel_count": int(log_fg.size),
            "z_min": float(z.min()), "z_max": float(z.max())}


def normalize_heatmap(raw: np.ndarray, median: float, iqr: float, background_value: float) -> np.ndarray:
    """Raw (2, H, W) map -> stored (1, H, W) float32 z-map (not clipped)."""
    ohm, mask = raw[0].astype(np.float32), raw[1]
    z = (np.log1p(ohm) - median) / iqr
    z[mask == 0] = background_value
    return z[np.newaxis].astype(np.float32)


def impedance_stats(imp_paths: list[Path]) -> dict:
    """Mean / std of log(ohm) over all points of the given placements."""
    log_imp = np.log(np.maximum(np.concatenate([np.load(p).flatten() for p in imp_paths]), IMPEDANCE_FLOOR_OHM))
    mean = float(log_imp.mean())
    std = float(log_imp.std()) or 1.0
    z = (log_imp - mean) / std
    return {"layout_count": len(imp_paths), "log_mean": mean, "log_std": std,
            "z_min": float(z.min()), "z_max": float(z.max())}


def normalize_impedance(raw_imp: np.ndarray, stats: dict) -> np.ndarray:
    """Raw spectrum in ohm (231 values) -> stored (1, 231) float32 z-scores."""
    log_data = np.log(np.maximum(raw_imp.astype(np.float32), IMPEDANCE_FLOOR_OHM))
    return ((log_data - stats["log_mean"]) / stats["log_std"]).flatten()[np.newaxis].astype(np.float32)


def bin_centers(freqs: np.ndarray, width: float) -> np.ndarray:
    lo = np.floor(freqs.min() / width) * width
    hi = np.ceil(freqs.max() / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo + width / 2.0, hi, width)


def bin_layout(freqs: np.ndarray, width: float, low_max: float = 0.0,
               low_step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres and the edges between neighbouring bins. Regular bins are ``width`` wide; up to
    ``low_max`` the centres sit every ``low_step``. A frequency on an edge belongs to the lower bin."""
    centers = bin_centers(freqs, width)
    if low_max <= 0 or low_step <= 0 or freqs.min() >= low_max:
        return centers, (centers[:-1] + centers[1:]) / 2.0
    first = np.ceil(round(float(freqs.min()) / low_step, 6)) * low_step
    low = np.round(np.arange(first, low_max + low_step / 2.0, low_step), 6)
    reg = centers[centers > low_max]
    edges = np.concatenate([(low[:-1] + low[1:]) / 2.0, [low_max], (reg[:-1] + reg[1:]) / 2.0])
    return np.concatenate([low, reg]), edges


def compute_bin_stats(rows: list[dict], hm_dir: Path, centers: np.ndarray, bin_idx: np.ndarray) -> dict[str, dict]:
    """Per-bin median / IQR plus the bin's sample count."""
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    n_pixels: dict[int, int] = defaultdict(int)
    for i, r in enumerate(rows):
        if n_pixels[bin_idx[i]] >= MAX_PIXELS_PER_BIN:
            continue
        log_fg = board_log_pixels(np.load(hm_dir / r["sample_name"]).astype(np.float32))
        if log_fg.size:
            acc[bin_idx[i]].append(log_fg)
            n_pixels[bin_idx[i]] += log_fg.size
    by_mhz: dict[str, dict] = {}
    for b, chunks in sorted(acc.items()):
        n_samples = int((bin_idx == b).sum())
        if n_samples < MIN_SAMPLES_PER_BIN:
            continue
        by_mhz[mhz_key(float(centers[b]))] = {**robust_bin_stats(np.concatenate(chunks)), "n_samples": n_samples}
    if not by_mhz:
        raise SystemExit("no usable frequency bins")
    return by_mhz


def _fit_stats(rows: list[dict], raw_root: Path, layouts: list[Path]) -> tuple[dict, list[float], bool]:
    freqs = np.array([float(r["freq_mhz"]) for r in rows])
    centers, edges = bin_layout(freqs, BIN_WIDTH_MHZ, LOW_BAND_MAX_MHZ, LOW_BAND_STEP_MHZ)
    fine = LOW_BAND_MAX_MHZ > 0 and len(centers) > len(bin_centers(freqs, BIN_WIDTH_MHZ))
    by_mhz = compute_bin_stats(rows, raw_root / "heatmap", centers, np.searchsorted(edges, freqs, side="left"))
    kept = sorted(float(k) for k in by_mhz)
    imp = impedance_stats([p / "imp.npy" for p in layouts if (p / "imp.npy").is_file()])
    hm = {"norm_mode": NORM_MODE, "unbounded": True, "anchors_mhz": kept, "by_mhz": by_mhz,
          "clip_min": min(b["clip_min"] for b in by_mhz.values()),
          "clip_max": max(b["clip_max"] for b in by_mhz.values()),
          "z_max": max(b["z_max"] for b in by_mhz.values()),
          "background_value": min(b["background_value"] for b in by_mhz.values()),
          "bin_width_mhz": BIN_WIDTH_MHZ,
          "low_band_max_mhz": LOW_BAND_MAX_MHZ if fine else None,
          "low_band_step_mhz": LOW_BAND_STEP_MHZ if fine else None}
    stats = {"background_value": hm["background_value"], "Heatmap": hm, "Impedance": imp,
             "PI_freq": {"min_hz": float(freqs.min() * 1e6), "max_hz": float(freqs.max() * 1e6),
                         "log10_min": float(np.log10(freqs.min() * 1e6)),
                         "log10_range": float(np.log10(freqs.max() * 1e6) - np.log10(freqs.min() * 1e6)),
                         "count": len(rows), "unique_mhz_rounded": int(len(np.unique(np.round(freqs, 1))))},
             "storage": "layout_store"}
    return stats, kept, fine


def normalize_store(raw_root: Path, out_root: Path, stats_json: Path | None,
                    design_ids: set[str] | None = None) -> Path:
    """Normalize ``raw_root`` into ``out_root``. ``stats_json`` given: apply it; None: fit statistics.
    ``design_ids`` restricts the written samples and placements (statistics are never fitted on a subset)."""
    with (raw_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty manifest in {raw_root}")
    layouts = [p for p in sorted((raw_root / "layouts").iterdir()) if p.is_dir()]
    if stats_json is None:
        if design_ids is not None:
            raise ValueError("statistics are fitted on the whole store; pass stats_json with design_ids")
        stats, kept, fine = _fit_stats(rows, raw_root, layouts)
    else:
        stats = json.loads(Path(stats_json).read_text(encoding="utf-8"))
        kept, fine = [float(x) for x in stats["Heatmap"]["anchors_mhz"]], False
    if design_ids is not None:
        rows = [r for r in rows if r["design_id"] in design_ids]
        layouts = [p for p in layouts if p.name in design_ids]

    if out_root.is_dir():
        shutil.rmtree(out_root)
    for d in ("heatmap", "PI_freq", "layouts"):
        (out_root / d).mkdir(parents=True)
    (out_root / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    freqs = np.array([float(r["freq_mhz"]) for r in rows])
    meta = {"schema_version": 1, "stage": "normalized", "source_script": "thesis_code/s03_normalization/normalize.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(out_root),
            "storage": "layout_store", "pi_frequencies_mhz": kept, "n_decap_types": N_DECAP_TYPES,
            "occupancy_schema": "type_codes_0_to_T", "heatmap_norm": NORM_MODE,
            "impedance_norm": IMPEDANCE_NORM_MODE, "frequency_scheme": "per_layout_random",
            "bin_width_mhz": BIN_WIDTH_MHZ if stats_json is None else None,
            "low_band_max_mhz": LOW_BAND_MAX_MHZ if stats_json is None and fine else None,
            "low_band_step_mhz": LOW_BAND_STEP_MHZ if stats_json is None and fine else None,
            "stats_from": str(stats_json) if stats_json is not None else None,
            "counts": {"manifest_rows": len(rows), "unique_layouts": len({r["design_id"] for r in rows}),
                       "distinct_frequencies": int(len(np.unique(freqs))), "frequency_bins": len(kept)}}
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for p in layouts:
        dst = out_root / "layouts" / p.name
        dst.mkdir(parents=True, exist_ok=True)
        src = p / "type.npy" if (p / "type.npy").is_file() else p / "occ.npy"
        codes = np.asarray(np.load(src)).reshape(-1).astype(np.int8)
        np.save(dst / "type.npy", codes)
        np.save(dst / "occ.npy", codes.astype(np.float32))
        if (p / "imp.npy").is_file():
            np.save(dst / "imp.npy", normalize_impedance(np.load(p / "imp.npy"), stats["Impedance"]))

    hstats = load_norm_stats(out_root).heatmap
    for r in rows:
        b = hstats.bin_stats(float(r["freq_mhz"]), interp=True)
        raw = np.load(raw_root / "heatmap" / r["sample_name"]).astype(np.float32)
        np.save(out_root / "heatmap" / r["sample_name"], normalize_heatmap(raw, b.median, b.iqr, b.background_value))
        shutil.copyfile(raw_root / "PI_freq" / r["sample_name"], out_root / "PI_freq" / r["sample_name"])
    if design_ids is None:
        shutil.copyfile(raw_root / "manifest.csv", out_root / "manifest.csv")
    else:
        with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"normalized {len(rows):,} samples, {len(layouts):,} placements -> {out_root}")
    return out_root


def main() -> None:
    normalize_store(INPUT_DIR, out_dir("s03_normalization") / "normalized", STATS_JSON)


if __name__ == "__main__":
    main()
