#!/usr/bin/env python3
"""Normalize a held-out dataset using training normalization stats.

Run: python pipelines/normalize/apply_stats.py"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/normalize/apply_stats.py
# =============================================================================

IN_ROOT = Path("datasets/data_eval")  # raw eval dataset (heatmap/, Imp/, Occ_map/)
OUT_ROOT = Path("datasets/data_eval_norm")  # normalized output
STATS_JSON = Path("datasets/data_norm/normalization_stats.json")  # training stats (no leakage)
OVERWRITE = False  # delete OUT_ROOT before writing
MAX_SAMPLES: int | None = None  # debug cap (None = all)
PRINT_EVERY = 2000

# =============================================================================

_SAMPLE_RE = re.compile(r"sample_(\d+)\.npy$")


def _parse_sample_id(name: str) -> int | None:
    m = _SAMPLE_RE.fullmatch(name)
    if not m:
        return None
    return int(m.group(1))


def _load_stats(stats_json: Path) -> tuple[dict, dict]:
    stats = json.loads(stats_json.read_text(encoding="utf-8"))

    if "Heatmap" not in stats or "Impedance" not in stats:
        raise SystemExit(f"Stats JSON missing expected keys: {stats_json}")

    hm = stats["Heatmap"]
    imp = stats["Impedance"]

    required_hm = ("log_mean", "log_std", "clip_min", "clip_max")
    required_imp = ("log_mean", "log_std")

    for k in required_hm:
        if k not in hm:
            raise SystemExit(f"Stats JSON missing Heatmap.{k}: {stats_json}")
    for k in required_imp:
        if k not in imp:
            raise SystemExit(f"Stats JSON missing Impedance.{k}: {stats_json}")

    bg_value = stats.get("background_value", hm.get("background_value"))
    if bg_value is None:
        raise SystemExit(f"Stats JSON missing background_value: {stats_json}")

    hm = dict(hm)
    hm["background_value"] = float(bg_value)

    return hm, imp


def _normalize_heatmap(raw: np.ndarray, *, hm_stats: dict) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)

    # Accept either (2,H,W) raw or already-normalized (1,H,W).
    if raw.ndim == 3 and raw.shape[0] == 1:
        return raw.astype(np.float32, copy=False)

    if raw.ndim != 3 or raw.shape[0] < 2:
        raise ValueError(f"Unexpected heatmap shape: {raw.shape}")

    ch0 = raw[0]
    mask = raw[1]

    log_mean = float(hm_stats["log_mean"])
    log_std = float(hm_stats["log_std"])
    clip_min = float(hm_stats["clip_min"])
    clip_max = float(hm_stats["clip_max"])
    bg_value = float(hm_stats["background_value"])

    log_ch0 = np.log1p(np.maximum(ch0, 0.0))
    z = (log_ch0 - log_mean) / (log_std if log_std != 0.0 else 1.0)
    z = np.clip(z, clip_min, clip_max)

    bg_mask = mask <= 0.5
    z[bg_mask] = bg_value

    return z[np.newaxis].astype(np.float32, copy=False)


def _normalize_impedance(raw: np.ndarray, *, imp_stats: dict) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.size != 231:
        raise ValueError(f"Unexpected impedance length: {raw.size} (expected 231)")

    log_mean = float(imp_stats["log_mean"])
    log_std = float(imp_stats["log_std"])

    log_data = np.log(np.maximum(raw, 1e-10))
    z = (log_data - log_mean) / (log_std if log_std != 0.0 else 1.0)

    diff1 = np.diff(z)
    deriv1 = np.append(diff1, diff1[-1] if diff1.size else 0.0)

    diff2 = np.diff(diff1)
    if diff2.size:
        deriv2 = np.append(diff2, [diff2[-1], diff2[-1]])
    else:
        deriv2 = np.asarray([0.0, 0.0] + [0.0] * (z.size - 2), dtype=np.float32)

    tri = np.stack([z, deriv1, deriv2], axis=0)
    return tri.astype(np.float32, copy=False)


def main() -> None:
    in_root: Path = IN_ROOT
    out_root: Path = OUT_ROOT

    if not in_root.exists():
        raise SystemExit(f"Input root not found: {in_root}")
    if not (in_root / "heatmap").is_dir() or not (in_root / "Imp").is_dir() or not (in_root / "Occ_map").is_dir():
        raise SystemExit(f"Input root missing expected subdirs heatmap/ Imp/ Occ_map/: {in_root}")

    if OVERWRITE and out_root.exists():
        shutil.rmtree(out_root)

    hm_out = out_root / "heatmap"
    imp_out = out_root / "Imp"
    occ_out = out_root / "Occ_map"
    hm_out.mkdir(parents=True, exist_ok=True)
    imp_out.mkdir(parents=True, exist_ok=True)
    occ_out.mkdir(parents=True, exist_ok=True)

    hm_stats, imp_stats = _load_stats(STATS_JSON)

    occ_dir = in_root / "Occ_map"
    ids: list[int] = []
    for p in occ_dir.glob("sample_*.npy"):
        sid = _parse_sample_id(p.name)
        if sid is not None:
            ids.append(sid)
    ids.sort()
    if not ids:
        raise SystemExit(f"No sample_*.npy files found in {occ_dir}")

    if MAX_SAMPLES is not None:
        ids = ids[: max(0, int(MAX_SAMPLES))]

    print("=" * 60)
    print("NORMALIZE DATASET (USING TRAINING STATS)")
    print("=" * 60)
    print(f"Input:   {in_root}")
    print(f"Output:  {out_root}")
    print(f"Stats:   {STATS_JSON}")
    print(f"Samples: {len(ids)}")

    done = 0
    skipped = 0
    errors = 0

    for i, sid in enumerate(ids):
        name = f"sample_{sid}.npy"

        out_hm = hm_out / name
        out_imp = imp_out / name
        out_occ = occ_out / name

        if out_hm.exists() and out_imp.exists() and out_occ.exists():
            skipped += 1
            continue

        try:
            hm_raw = np.load(in_root / "heatmap" / name)
            imp_raw = np.load(in_root / "Imp" / name)
            occ_raw = np.load(in_root / "Occ_map" / name)

            hm_norm = _normalize_heatmap(hm_raw, hm_stats=hm_stats)
            imp_norm = _normalize_impedance(imp_raw, imp_stats=imp_stats)
            occ_norm = np.asarray(occ_raw, dtype=np.float32).reshape(-1)
            if occ_norm.size != 52:
                raise ValueError(f"Unexpected occupancy length: {occ_norm.size} (expected 52)")

            np.save(out_hm, hm_norm)
            np.save(out_imp, imp_norm)
            np.save(out_occ, occ_norm.astype(np.float32, copy=False))

            done += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ✗ {name}: {e}")

        if PRINT_EVERY and (i + 1) % int(PRINT_EVERY) == 0:
            print(f"  Progress: {i + 1}/{len(ids)} | wrote {done} | skipped {skipped} | errors {errors}")

    # Copy manifest if present (best effort)
    manifest_in = in_root / "manifest.csv"
    if manifest_in.exists():
        shutil.copy2(manifest_in, out_root / "manifest.csv")

    # Record provenance of stats used
    prov = {
        "input_root": str(in_root),
        "stats_json": str(STATS_JSON),
        "background_value": float(hm_stats["background_value"]),
    }
    (out_root / "normalization_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    from libs.dataset_meta import write_dataset_meta

    write_dataset_meta(
        out_root,
        stage="normalized",
        source_script="pipelines/normalize/apply_stats.py",
        extra={"stats_source": str(STATS_JSON), "samples_written": done},
    )

    print("\nDone")
    print(f"  Wrote:   {done}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")


if __name__ == "__main__":
    main()
