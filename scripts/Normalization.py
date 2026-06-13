"""
Normalize data_multifreq → data_multifreq_norm (layout-centric).

Input (from Data_processing_multifreq.py):
  layouts/{design_id}/imp.npy, occ.npy
  heatmap/sample_*.npy, PI_freq/sample_*.npy, manifest.csv

Output:
  Same structure; heatmaps log(1+x) z-scored, impedance log z-scored in layouts/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_vae.others.multifreq_anchors import load_anchors_mhz  # noqa: E402
from src_vae.others.multifreq_layout_store import (  # noqa: E402
    invalidate_training_caches,
    iter_layout_imp_paths,
    iter_layout_occ_paths,
    layout_imp_path,
    layout_occ_path,
    layout_dir,
    layouts_root,
    load_manifest_index,
    manifest_path,
    validate_multifreq_dataset,
)

OVERWRITE_OUTPUT = os.getenv("DATA_OVERWRITE", "1").strip().lower() not in ("0", "false", "no")


def prepare_output_dir(output_root: Path, *, overwrite: bool = OVERWRITE_OUTPUT) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root} (set DATA_OVERWRITE=1 to replace)")
        print(f"Removing existing folder: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def calculate_heatmap_stats(heatmap_dir: Path, percentile_lower=0.1, percentile_upper=99.9) -> dict:
    heatmap_files = sorted(Path(heatmap_dir).glob("*.npy"))
    print(f"\n[1/4] Heatmap log(1+x) stats from {len(heatmap_files)} files...")

    all_log_fg = []
    for path in tqdm(heatmap_files, desc="Heatmap stats"):
        data = np.load(path).astype(np.float32)
        if data.ndim < 3 or data.shape[0] < 2:
            raise ValueError(f"Unexpected heatmap shape in {path}: {data.shape}")
        ch0, mask = data[0], data[1]
        fg_vals = ch0[mask == 1]
        if fg_vals.size > 0:
            all_log_fg.append(np.log1p(fg_vals))

    all_log_fg = np.concatenate(all_log_fg)
    log_mean = float(all_log_fg.mean())
    log_std = float(all_log_fg.std()) or 1.0
    z_scores = (all_log_fg - log_mean) / log_std
    clip_min = float(np.percentile(z_scores, percentile_lower))
    clip_max = float(np.percentile(z_scores, percentile_upper))
    bg_value = round(clip_min - 1.5, 4)
    print(f"  log(1+x): mean={log_mean:.4f} std={log_std:.4f}  bg={bg_value}")

    return {
        "log_mean": log_mean,
        "log_std": log_std,
        "fg_pixel_count": int(all_log_fg.size),
        "z_min": float(z_scores.min()),
        "z_max": float(z_scores.max()),
        "clip_min": clip_min,
        "clip_max": clip_max,
        "background_value": bg_value,
    }


def normalize_heatmaps(heatmap_dir: Path, output_heatmap_dir: Path, heatmap_stats: dict) -> None:
    output_heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    print(f"\n[2/4] Normalizing {len(heatmap_files)} heatmaps...")

    log_mean = heatmap_stats["log_mean"]
    log_std = heatmap_stats["log_std"]
    bg_value = heatmap_stats["background_value"]
    clip_min = heatmap_stats["clip_min"]
    clip_max = heatmap_stats["clip_max"]

    for path in tqdm(heatmap_files, desc="Heatmaps"):
        data = np.load(path).astype(np.float32)
        ch0, mask = data[0], data[1]
        z_ch0 = (np.log1p(ch0) - log_mean) / log_std
        z_ch0 = np.clip(z_ch0, clip_min, clip_max)
        z_ch0[mask == 0] = bg_value
        np.save(output_heatmap_dir / path.name, z_ch0[np.newaxis].astype(np.float32))


def calculate_impedance_stats(data_root: Path) -> dict:
    imp_files = iter_layout_imp_paths(data_root)
    if not imp_files:
        raise FileNotFoundError(f"No layouts/*/imp.npy under {data_root}")
    print(f"\n[3/4] Impedance stats from {len(imp_files)} layouts...")

    imp_values = np.concatenate([np.load(f).flatten() for f in tqdm(imp_files, desc="Impedance stats")])
    log_imp = np.log(np.maximum(imp_values, 1e-10))
    i_mean = float(log_imp.mean())
    i_std = float(log_imp.std()) or 1.0
    z_scores = (log_imp - i_mean) / i_std
    print(f"  log z-score: mean={i_mean:.4f} std={i_std:.4f}")
    return {
        "layout_count": len(imp_files),
        "log_mean": i_mean,
        "log_std": i_std,
        "z_min": float(z_scores.min()),
        "z_max": float(z_scores.max()),
    }


def normalize_layout_impedance(data_root: Path, output_root: Path, imp_stats: dict) -> None:
    imp_log_mean = imp_stats["log_mean"]
    imp_log_std = imp_stats["log_std"]
    out_layouts = layouts_root(output_root)
    out_layouts.mkdir(parents=True, exist_ok=True)

    for path in tqdm(iter_layout_imp_paths(data_root), desc="Impedance → layouts"):
        design_id = path.parent.name
        log_data = np.log(np.maximum(np.load(path).astype(np.float32), 1e-10))
        z = ((log_data - imp_log_mean) / imp_log_std).flatten()
        sub = layout_dir(output_root, design_id)
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "imp.npy", z[np.newaxis].astype(np.float32))


def copy_layout_occupancy(data_root: Path, output_root: Path) -> None:
    out_layouts = layouts_root(output_root)
    out_layouts.mkdir(parents=True, exist_ok=True)
    occ_files = iter_layout_occ_paths(data_root)
    print(f"\n[4/4] Copying {len(occ_files)} occupancy layouts...")
    for occ_path in tqdm(occ_files, desc="Occupancy"):
        design_id = occ_path.parent.name
        sub = layout_dir(output_root, design_id)
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "occ.npy", np.load(occ_path).astype(np.float32))


def validate_pifreq(pifreq_dir: Path, expected_mhz=None) -> dict:
    if not pifreq_dir.is_dir():
        return {}
    if expected_mhz is None:
        expected_mhz = [float(x) for x in load_anchors_mhz()]
    files = sorted(pifreq_dir.glob("*.npy"))
    mhz_vals = [float(np.load(p)) / 1e6 for p in files]
    uniq = sorted(set(round(m, 1) for m in mhz_vals))
    print(f"  PI_freq: {len(files)} files, MHz (rounded): {uniq}")
    return {"count": len(files), "unique_mhz_rounded": uniq, "expected_anchors_mhz": expected_mhz}


def copy_pifreq(pifreq_dir: Path, output_pifreq_dir: Path) -> None:
    if not pifreq_dir.is_dir():
        return
    output_pifreq_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pifreq_dir.glob("*.npy"))
    print(f"\n[5/5] Copying {len(files)} PI_freq files (raw Hz)...")
    for path in tqdm(files, desc="PI_freq"):
        np.save(output_pifreq_dir / path.name, np.load(path))


def _missing_heatmap_names(data_root: Path, output_root: Path) -> list[str]:
    raw_hm = data_root / "heatmap"
    out_hm = output_root / "heatmap"
    if not raw_hm.is_dir():
        return []
    raw_names = {p.name for p in raw_hm.glob("sample_*.npy")}
    if not out_hm.is_dir():
        return sorted(raw_names)
    out_names = {p.name for p in out_hm.glob("sample_*.npy")}
    return sorted(raw_names - out_names)


def normalize_append(data_root: Path, output_root: Path) -> int:
    """Normalize new heatmap rows; copy/normalize new layouts from raw layouts/."""
    validate_multifreq_dataset(data_root)
    stats_file = output_root / "normalization_stats.json"
    if not stats_file.is_file():
        raise FileNotFoundError(f"{stats_file} missing — run full Normalization.py first")

    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    heatmap_stats = stats["Heatmap"]
    imp_stats = stats.get("Impedance", {})
    missing = _missing_heatmap_names(data_root, output_root)
    if not missing:
        print("Append: no new sample_*.npy to normalize.")
        return 0

    print(f"Append: {len(missing)} new heatmap sample(s)...")
    out_heatmap = output_root / "heatmap"
    out_layouts = layouts_root(output_root)
    out_pifreq = output_root / "PI_freq"
    for d in (out_heatmap, out_layouts, out_pifreq):
        d.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_index(data_root)
    imp_log_mean = imp_stats.get("log_mean", 0.0)
    imp_log_std = imp_stats.get("log_std", 1.0) or 1.0
    hm_in = data_root / "heatmap"
    pif_in = data_root / "PI_freq"

    layouts_touched: set[str] = set()
    for name in tqdm(missing, desc="Append normalize"):
        raw_hm = hm_in / name
        if not raw_hm.is_file():
            continue
        data = np.load(raw_hm).astype(np.float32)
        ch0, mask = data[0], data[1]
        z_ch0 = (np.log1p(ch0) - heatmap_stats["log_mean"]) / heatmap_stats["log_std"]
        z_ch0 = np.clip(z_ch0, heatmap_stats["clip_min"], heatmap_stats["clip_max"])
        z_ch0[mask == 0] = heatmap_stats["background_value"]
        np.save(out_heatmap / name, z_ch0[np.newaxis].astype(np.float32))

        stem = Path(name).stem
        did = manifest.get(stem)
        if did and did not in layouts_touched:
            out_sub = layout_dir(output_root, did)
            imp_src = layout_imp_path(data_root, did)
            occ_src = layout_occ_path(data_root, did)
            if imp_src.is_file() and not (out_sub / "imp.npy").is_file():
                out_sub.mkdir(parents=True, exist_ok=True)
                log_data = np.log(np.maximum(np.load(imp_src).astype(np.float32), 1e-10))
                z = ((log_data - imp_log_mean) / imp_log_std).flatten()
                np.save(out_sub / "imp.npy", z[np.newaxis].astype(np.float32))
            if occ_src.is_file() and not (out_sub / "occ.npy").is_file():
                out_sub.mkdir(parents=True, exist_ok=True)
                np.save(out_sub / "occ.npy", np.load(occ_src).astype(np.float32))
            layouts_touched.add(did)

        pf_path = pif_in / name
        if pf_path.is_file():
            np.save(out_pifreq / name, np.load(pf_path))

    if manifest_path(data_root).is_file():
        shutil.copy2(manifest_path(data_root), manifest_path(output_root))
    stats["PI_freq"] = stats.get("PI_freq", {})
    stats["PI_freq"]["anchor_mhz"] = [float(x) for x in load_anchors_mhz()]
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    invalidate_training_caches(output_root)
    return len(missing)


def run_full_pipeline(data_root: Path, output_root: Path) -> None:
    validate_multifreq_dataset(data_root)
    prepare_output_dir(output_root, overwrite=OVERWRITE_OUTPUT)

    heatmap_stats = calculate_heatmap_stats(data_root / "heatmap")
    normalize_heatmaps(data_root / "heatmap", output_root / "heatmap", heatmap_stats)
    imp_stats = calculate_impedance_stats(data_root)
    normalize_layout_impedance(data_root, output_root, imp_stats)
    copy_layout_occupancy(data_root, output_root)
    pifreq_stats = validate_pifreq(data_root / "PI_freq")
    copy_pifreq(data_root / "PI_freq", output_root / "PI_freq")

    manifest_info = {}
    src_manifest = manifest_path(data_root)
    if src_manifest.is_file():
        with src_manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        design_ids = {r["design_id"] for r in rows if r.get("design_id")}
        manifest_info = {
            "rows": len(rows),
            "unique_design_id": len(design_ids),
            "freq_mhz_counts": dict(Counter(r.get("freq_mhz", "") for r in rows if r.get("freq_mhz"))),
        }
        shutil.copy2(src_manifest, manifest_path(output_root))
        print(f"\n  manifest.csv: {manifest_info['rows']} rows, {manifest_info['unique_design_id']} layouts")

    stats = {
        "background_value": heatmap_stats["background_value"],
        "Heatmap": heatmap_stats,
        "Impedance": imp_stats,
        "PI_freq": {
            "min_hz": 1e6,
            "max_hz": 600e6,
            "log10_min": 6.0,
            "log10_range": float(np.log10(600e6) - np.log10(1e6)),
            "anchor_mhz": [float(x) for x in load_anchors_mhz()],
            **pifreq_stats,
        },
        "multifreq_manifest": manifest_info,
        "storage": "layout_store",
    }
    stats_path = output_root / "normalization_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    invalidate_training_caches(output_root)
    print(f"\n{'=' * 60}\n✓ NORMALIZATION COMPLETE\n{'=' * 60}")
    print(f"Dataset: {output_root}\nStats: {stats_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize data_multifreq → data_multifreq_norm (layouts/)")
    ap.add_argument("--append", action="store_true", help="Only new sample_*.npy + layouts")
    args = ap.parse_args()

    root = _REPO_ROOT
    data_root = root / "datasets" / "data_multifreq"
    output_root = root / "datasets" / "data_multifreq_norm"

    if args.append:
        n = normalize_append(data_root, output_root)
        print(f"\n✓ Append complete ({n} samples)")
        return

    if not data_root.exists():
        raise SystemExit(f"Input missing: {data_root}")
    print(f"\n{'=' * 60}\nNORMALIZATION (layout store)\n{'=' * 60}")
    print(f"Input:  {data_root}\nOutput: {output_root}")
    run_full_pipeline(data_root, output_root)


if __name__ == "__main__":
    main()
