"""Multifreq dataset normalization pipeline.

Purpose:
    Read raw multifreq dataset; apply log/global-max normalization; write ``data_multifreq_norm``
    with ``normalization_stats.json`` and updated ``dataset_meta.json``.

Run:
    python pipelines/normalize/multifreq.py

Agent notes:
    - What: Produces VAE-ready normalized tensors from ``pipelines/data/processing_multifreq.py`` output.
    - Usage: Set ``DATA_DIR`` (input) and ``OUTPUT_DIR`` → run. ``APPEND=True`` adds new samples only.
    - Config keys:
        - ``DATA_DIR`` — raw multifreq root
        - ``OUTPUT_DIR`` — normalized output; ``None`` defaults beside input
        - ``APPEND`` — incremental normalize vs full rebuild
    - Key symbols: ``normalize_multifreq``, ``prepare_output_dir``
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

import numpy as np
from tqdm import tqdm

from repo_paths import REPO_ROOT as _REPO_ROOT, setup_path
setup_path()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_vae.others.multifreq_anchors import load_anchors_mhz, nearest_anchor_mhz  # noqa: E402
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
    manifest_design_ids,
    iter_manifest_layout_imp_paths,
    prune_orphan_layouts,
    repair_multifreq_dataset,
    validate_multifreq_dataset,
)
from libs.dataset_meta import write_dataset_meta  # noqa: E402
from src_vae.others.heatmap_gmax_norm import physical_to_gmax_norm  # noqa: E402

OVERWRITE_OUTPUT = os.getenv("DATA_OVERWRITE", "1").strip().lower() not in ("0", "false", "no")

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/normalize/multifreq.py
# =============================================================================

# Input multifreq dataset (raw, from processing_multifreq.py)
DATA_DIR = _REPO_ROOT / "datasets" / "data_multifreq_train"

# True → heatmaps Ω/global_max (direct gmax dataset); False → log(1+x) z-score or robust
USE_GLOBAL_MAX_HEATMAP = False

# True → median/IQR per training-anchor MHz (recommended for exp048+)
USE_ROBUST_PER_MHZ = os.getenv("NORM_ROBUST_PER_MHZ", "1").strip().lower() in ("1", "true", "yes")
# True → no foreground z-clip at normalize/load (exp049 unbounded log z-score)
USE_UNBOUNDED_Z = os.getenv("NORM_UNBOUNDED_Z", "0").strip().lower() in ("1", "true", "yes")
ROBUST_IQR_EPS = 1e-4
ROBUST_CLIP_PERCENTILE_LOWER = 0.5
ROBUST_CLIP_PERCENTILE_UPPER = 99.5

# Output dataset (None = auto: data_multifreq_gmax if gmax else data_multifreq_norm)
if USE_ROBUST_PER_MHZ and USE_UNBOUNDED_Z:
    OUTPUT_DIR: Path | None = "data_multi_norm_unbounded"
elif USE_ROBUST_PER_MHZ:
    OUTPUT_DIR: Path | None = _REPO_ROOT / "datasets" / "data_multifreq_train_norm_robust"
else:
    OUTPUT_DIR: Path | None = "data_multi_norm"

# Global-max heatmap options (only when USE_GLOBAL_MAX_HEATMAP=True)
GLOBAL_MAX_OHM: float | None = None  # None = scan all maps at GMAX_PERCENTILE
GMAX_PERCENTILE = 99.5
BG_OHM = 0.05  # foreground threshold in physical Ω
CLIP_MAX = 1.02  # clip normalized heatmaps to [0, CLIP_MAX]

# True = only add new sample_*.npy rows; False = full rebuild (wipes OUTPUT_DIR)
APPEND = False

# True = prune orphan heatmaps / manifest rows before normalize (fixes count mismatch)
REPAIR_DATASET = True

# True = only copy layouts listed in manifest.csv (not every folder under source layouts/)
MANIFEST_LAYOUTS_ONLY = True

# True = remove layouts/ folders in OUTPUT not referenced by manifest
PRUNE_ORPHAN_LAYOUTS = True

# Keep only layouts with decap count K <= MAX_K (None = no filter).
MAX_K: int | None = None

# =============================================================================


def _k_from_occ(data_root: Path, design_id: str) -> int:
    occ = np.load(layout_occ_path(data_root, design_id), mmap_mode="r").reshape(-1)
    return int((occ > 0.5).sum())


def load_design_k(data_root: Path) -> dict[str, int]:
    """design_id → decap count K (from shared occupancy)."""
    seen: dict[str, int] = {}
    mp = manifest_path(data_root)
    if not mp.is_file():
        return seen
    with mp.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = row.get("design_id")
            if not did or did in seen:
                continue
            seen[did] = _k_from_occ(data_root, did)
    return seen


def _allowed_design_ids(data_root: Path, max_k: int) -> tuple[dict[str, int], set[str]]:
    design_k = load_design_k(data_root)
    allowed = {did for did, k in design_k.items() if k <= max_k}
    return design_k, allowed


def _filter_heatmap_paths(
    heatmap_dir: Path,
    manifest_index: dict[str, str],
    allowed_design_ids: set[str],
) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(heatmap_dir.glob("*.npy")):
        did = manifest_index.get(path.stem)
        if did and did in allowed_design_ids:
            paths.append(path)
    return paths


def _filter_manifest_rows(rows: list[dict], allowed_design_ids: set[str]) -> list[dict]:
    return [r for r in rows if r.get("design_id") in allowed_design_ids]


def _print_k_filter_summary(
    design_k: dict[str, int],
    allowed_design_ids: set[str],
    *,
    max_k: int,
    manifest_rows_before: int,
    manifest_rows_after: int,
) -> None:
    dropped_ids = set(design_k) - allowed_design_ids
    dropped_k = Counter(design_k[did] for did in dropped_ids)
    print(
        f"\nK filter (MAX_K={max_k}): "
        f"{len(allowed_design_ids):,} layouts kept, "
        f"{len(dropped_ids):,} layouts dropped"
    )
    print(
        f"  manifest rows: {manifest_rows_before:,} → {manifest_rows_after:,} "
        f"({manifest_rows_before - manifest_rows_after:,} removed)"
    )
    if dropped_k:
        hi = sorted(dropped_k)
        print(f"  dropped K range: {min(hi)}..{max(hi)}  (counts: {dict(sorted(dropped_k.items()))})")


def _maybe_repair_dataset(data_root: Path) -> None:
    if not REPAIR_DATASET:
        return
    stats = repair_multifreq_dataset(data_root, dry_run=False)
    if (
        stats["orphan_heatmaps"]
        or stats["orphan_manifest_stems"]
        or stats["manifest_dupes_dropped"]
    ):
        print(
            f"\nDataset repair: removed {stats['removed_heatmap_files']} orphan heatmap(s), "
            f"dropped {stats['orphan_manifest_stems']} manifest row(s) without heatmap, "
            f"deduped {stats['manifest_dupes_dropped']} duplicate manifest row(s) → "
            f"{stats['heatmap_files_after']} heatmaps / {stats['manifest_rows_after']} manifest rows"
        )


def _is_gmax_heatmap_stats(heatmap_stats: dict) -> bool:
    return (
        heatmap_stats.get("norm_mode") == "global_max"
        or heatmap_stats.get("global_max_ohm") is not None
    )


def _output_dir() -> Path:
    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR).resolve()
    if USE_GLOBAL_MAX_HEATMAP:
        name = "data_multifreq_gmax"
    else:
        name = "data_multifreq_norm"
    return (_REPO_ROOT / "datasets" / name).resolve()


def _raw_heatmap_plane_mask(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim < 3 or data.shape[0] < 2:
        raise ValueError(f"Unexpected heatmap shape: {data.shape}")
    return data[0].astype(np.float32), data[1]


def prepare_output_dir(output_root: Path, *, overwrite: bool = OVERWRITE_OUTPUT) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root} (set DATA_OVERWRITE=1 to replace)")
        print(f"Removing existing folder: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def calculate_heatmap_stats_gmax(
    heatmap_dir: Path,
    *,
    global_max_ohm: float | None = None,
    gmax_percentile: float = GMAX_PERCENTILE,
    bg_ohm: float = BG_OHM,
    clip_max: float = CLIP_MAX,
    heatmap_files: list[Path] | None = None,
) -> dict:
    """Scan raw physical Ω heatmaps; return global-max stats."""
    if heatmap_files is None:
        heatmap_files = sorted(Path(heatmap_dir).glob("*.npy"))
    print(f"\n[1/4] Heatmap global-max stats from {len(heatmap_files)} files...")

    per_map_max: list[float] = []
    fg_pixel_count = 0
    for path in tqdm(heatmap_files, desc="Gmax scan"):
        ch0, mask = _raw_heatmap_plane_mask(np.load(path).astype(np.float32))
        fg = (mask == 1) & (ch0 > bg_ohm)
        if fg.any():
            per_map_max.append(float(ch0[fg].max()))
            fg_pixel_count += int(fg.sum())

    if not per_map_max:
        raise ValueError(f"No foreground pixels > {bg_ohm} Ω under {heatmap_dir}")

    scanned = False
    if global_max_ohm is None:
        global_max_ohm = float(np.percentile(per_map_max, gmax_percentile))
        scanned = True
        print(
            f"  global_max_ohm = {global_max_ohm:.4f} Ω "
            f"(p{gmax_percentile} of {len(per_map_max)} map FG maxima)"
        )
    else:
        global_max_ohm = float(global_max_ohm)
        print(f"  global_max_ohm = {global_max_ohm:.4f} Ω (fixed)")

    fg_norm_threshold = float(bg_ohm / global_max_ohm)
    meta: dict = {
        "norm_mode": "global_max",
        "global_max_ohm": global_max_ohm,
        "background_value": 0.0,
        "bg_ohm_threshold": bg_ohm,
        "fg_norm_threshold": fg_norm_threshold,
        "clip_min": 0.0,
        "clip_max": float(clip_max),
        "log_mean": 0.0,
        "log_std": 1.0,
        "fg_pixel_count": fg_pixel_count,
        "per_map_max_count": len(per_map_max),
        "per_map_max_min": float(min(per_map_max)),
        "per_map_max_max": float(max(per_map_max)),
    }
    if scanned:
        meta["gmax_percentile"] = float(gmax_percentile)
    else:
        meta["global_max_ohm_fixed"] = True
    return meta


def _load_stem_to_mhz(data_root: Path) -> dict[str, float]:
    """Map heatmap stem → freq_mhz from manifest.csv."""
    mp: dict[str, float] = {}
    mp_path = manifest_path(data_root)
    if not mp_path.is_file():
        return mp
    with mp_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row.get("sample_name", "")).stem
            if not stem:
                continue
            fm = row.get("freq_mhz")
            if fm:
                mp[stem] = float(fm)
    return mp


def _is_robust_per_mhz_stats(heatmap_stats: dict) -> bool:
    mode = heatmap_stats.get("norm_mode", "")
    return mode in ("robust_log1p_per_mhz", "robust_log1p_per_mhz_unbounded") or bool(
        heatmap_stats.get("by_mhz")
    )


def _is_unbounded_heatmap_stats(heatmap_stats: dict) -> bool:
    mode = str(heatmap_stats.get("norm_mode", ""))
    return heatmap_stats.get("unbounded", False) or mode.endswith("_unbounded")


def _mhz_key(mhz: float) -> str:
    return f"{float(mhz):.1f}"


def calculate_heatmap_stats_robust_per_mhz(
    heatmap_dir: Path,
    data_root: Path,
    *,
    percentile_lower: float = ROBUST_CLIP_PERCENTILE_LOWER,
    percentile_upper: float = ROBUST_CLIP_PERCENTILE_UPPER,
    iqr_eps: float = ROBUST_IQR_EPS,
    heatmap_files: list[Path] | None = None,
) -> dict:
    """Robust median/IQR stats per training-anchor MHz."""
    stem_mhz = _load_stem_to_mhz(data_root)
    if heatmap_files is None:
        heatmap_files = sorted(Path(heatmap_dir).glob("*.npy"))
    anchors = [float(x) for x in load_anchors_mhz(DATA_DIR)]
    print(f"\n[1/4] Heatmap robust per-MHz stats from {len(heatmap_files)} files...")

    bins: dict[str, list[np.ndarray]] = { _mhz_key(a): [] for a in anchors }

    for path in tqdm(heatmap_files, desc="Heatmap robust bins"):
        stem = path.stem
        mhz = stem_mhz.get(stem)
        if mhz is None:
            pif = data_root / "PI_freq" / f"{stem}.npy"
            if pif.is_file():
                mhz = float(np.load(pif).item()) / 1e6
        if mhz is None:
            continue
        anchor = nearest_anchor_mhz(float(mhz), anchors)
        key = _mhz_key(anchor)
        try:
            data = np.load(path).astype(np.float32)
        except (FileNotFoundError, OSError):
            continue
        ch0, mask = _raw_heatmap_plane_mask(data)
        fg = ch0[mask == 1]
        if fg.size > 0:
            bins.setdefault(key, []).append(np.log1p(fg))

    by_mhz: dict[str, dict] = {}
    global_mins: list[float] = []
    global_maxs: list[float] = []

    for anchor in anchors:
        key = _mhz_key(anchor)
        chunks = bins.get(key, [])
        if not chunks:
            print(f"  WARNING: no samples for {key} MHz")
            continue
        log_fg = np.concatenate(chunks)
        med = float(np.median(log_fg))
        q25, q75 = np.percentile(log_fg, [25, 75])
        iqr = max(float(q75 - q25), iqr_eps)
        z = (log_fg - med) / iqr
        clip_min = float(np.percentile(z, percentile_lower))
        clip_max = float(np.percentile(z, percentile_upper))
        bg = round(clip_min - 1.5, 4)
        by_mhz[key] = {
            "median": med,
            "iqr": iqr,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "background_value": bg,
            "fg_pixel_count": int(log_fg.size),
            "z_min": float(z.min()),
            "z_max": float(z.max()),
        }
        global_mins.append(clip_min)
        global_maxs.append(clip_max)
        print(f"  {key} MHz: median={med:.4f} iqr={iqr:.4f} clip=[{clip_min:.3f},{clip_max:.3f}] z_max={z.max():.2f}")

    if not by_mhz:
        raise ValueError("No per-MHz robust stats computed — check manifest freq_mhz")

    g_clip_min = min(global_mins)
    g_clip_max = max(global_maxs)
    g_bg = min(b["background_value"] for b in by_mhz.values())
    g_z_max = max(b["z_max"] for b in by_mhz.values())

    norm_mode = (
        "robust_log1p_per_mhz_unbounded"
        if USE_UNBOUNDED_Z
        else "robust_log1p_per_mhz"
    )

    return {
        "norm_mode": norm_mode,
        "unbounded": bool(USE_UNBOUNDED_Z),
        "anchors_mhz": anchors,
        "by_mhz": by_mhz,
        "clip_min": g_clip_min,
        "clip_max": g_clip_max,
        "z_max": g_z_max,
        "background_value": g_bg,
        "log_mean": 0.0,
        "log_std": 1.0,
        "fg_pixel_count": sum(b["fg_pixel_count"] for b in by_mhz.values()),
    }


def _normalize_raw_heatmap_to_array(
    data: np.ndarray,
    heatmap_stats: dict,
    *,
    mhz: float | None = None,
) -> np.ndarray:
    """Normalize one raw (2,H,W) heatmap to saved (1,H,W) tensor."""
    ch0, mask = _raw_heatmap_plane_mask(data)
    if _is_gmax_heatmap_stats(heatmap_stats):
        gmax = float(heatmap_stats["global_max_ohm"])
        bg_ohm = float(heatmap_stats.get("bg_ohm_threshold", BG_OHM))
        clip_hi = float(heatmap_stats.get("clip_max", CLIP_MAX))
        norm = physical_to_gmax_norm(ch0, global_max_ohm=gmax, bg_ohm=bg_ohm)
        norm = np.clip(norm, 0.0, clip_hi).astype(np.float32)
        return norm[np.newaxis]

    if _is_robust_per_mhz_stats(heatmap_stats):
        if mhz is None:
            raise ValueError("mhz required for robust per-MHz normalize")
        anchor = nearest_anchor_mhz(float(mhz), heatmap_stats.get("anchors_mhz", load_anchors_mhz()))
        bin_st = heatmap_stats["by_mhz"][_mhz_key(anchor)]
        med = float(bin_st["median"])
        iqr = float(bin_st["iqr"])
        clip_min = float(bin_st["clip_min"])
        clip_max = float(bin_st["clip_max"])
        bg_value = float(bin_st["background_value"])
        z_ch0 = (np.log1p(ch0) - med) / iqr
        if not _is_unbounded_heatmap_stats(heatmap_stats):
            z_ch0 = np.clip(z_ch0, clip_min, clip_max)
        z_ch0[mask == 0] = bg_value
        return z_ch0[np.newaxis].astype(np.float32)

    log_mean = heatmap_stats["log_mean"]
    log_std = heatmap_stats["log_std"]
    bg_value = heatmap_stats["background_value"]
    clip_min = heatmap_stats["clip_min"]
    clip_max = heatmap_stats["clip_max"]
    z_ch0 = (np.log1p(ch0) - log_mean) / log_std
    if not _is_unbounded_heatmap_stats(heatmap_stats):
        z_ch0 = np.clip(z_ch0, clip_min, clip_max)
    z_ch0[mask == 0] = bg_value
    return z_ch0[np.newaxis].astype(np.float32)


def calculate_heatmap_stats(
    heatmap_dir: Path,
    percentile_lower=0.1,
    percentile_upper=99.9,
    heatmap_files: list[Path] | None = None,
) -> dict:
    if heatmap_files is None:
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
    norm_mode = "log_zscore_unbounded" if USE_UNBOUNDED_Z else "log_zscore"
    print(f"  log(1+x): mean={log_mean:.4f} std={log_std:.4f}  bg={bg_value}  mode={norm_mode}")

    return {
        "norm_mode": norm_mode,
        "unbounded": bool(USE_UNBOUNDED_Z),
        "log_mean": log_mean,
        "log_std": log_std,
        "fg_pixel_count": int(all_log_fg.size),
        "z_min": float(z_scores.min()),
        "z_max": float(z_scores.max()),
        "clip_min": clip_min,
        "clip_max": clip_max,
        "background_value": bg_value,
    }


def normalize_heatmaps(
    heatmap_dir: Path,
    output_heatmap_dir: Path,
    heatmap_stats: dict,
    *,
    stem_mhz: dict[str, float] | None = None,
    heatmap_files: list[Path] | None = None,
) -> None:
    output_heatmap_dir.mkdir(parents=True, exist_ok=True)
    if heatmap_files is None:
        heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    if _is_gmax_heatmap_stats(heatmap_stats):
        mode = "global_max"
    elif _is_robust_per_mhz_stats(heatmap_stats):
        mode = heatmap_stats.get("norm_mode", "robust_log1p_per_mhz")
    else:
        mode = heatmap_stats.get("norm_mode", "log_zscore")
    print(f"\n[2/4] Normalizing {len(heatmap_files)} heatmaps ({mode})...")

    for path in tqdm(heatmap_files, desc="Heatmaps"):
        data = np.load(path).astype(np.float32)
        mhz = stem_mhz.get(path.stem) if stem_mhz else None
        np.save(
            output_heatmap_dir / path.name,
            _normalize_raw_heatmap_to_array(data, heatmap_stats, mhz=mhz),
        )


def calculate_impedance_stats(
    data_root: Path,
    *,
    allowed_design_ids: set[str] | None = None,
) -> dict:
    if MANIFEST_LAYOUTS_ONLY and manifest_path(data_root).is_file():
        imp_files = iter_manifest_layout_imp_paths(data_root)
        if allowed_design_ids is not None:
            imp_files = [p for p in imp_files if p.parent.name in allowed_design_ids]
        n_disk = len(list(iter_layout_imp_paths(data_root)))
        n_ids = len(manifest_design_ids(data_root))
        print(
            f"\n[3/4] Impedance stats from {len(imp_files)} manifest layouts "
            f"({n_ids} unique design_id; {n_disk} layout folders on disk)..."
        )
    else:
        imp_files = iter_layout_imp_paths(data_root)
        if allowed_design_ids is not None:
            imp_files = [p for p in imp_files if p.parent.name in allowed_design_ids]
        print(f"\n[3/4] Impedance stats from {len(imp_files)} layouts...")
    if not imp_files:
        raise FileNotFoundError(f"No layouts/*/imp.npy under {data_root} for manifest layouts")

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


def normalize_layout_impedance(
    data_root: Path,
    output_root: Path,
    imp_stats: dict,
    *,
    allowed_design_ids: set[str] | None = None,
) -> None:
    imp_log_mean = imp_stats["log_mean"]
    imp_log_std = imp_stats["log_std"]
    out_layouts = layouts_root(output_root)
    out_layouts.mkdir(parents=True, exist_ok=True)

    if MANIFEST_LAYOUTS_ONLY and manifest_path(data_root).is_file():
        imp_paths = iter_manifest_layout_imp_paths(data_root)
        if allowed_design_ids is not None:
            imp_paths = [p for p in imp_paths if p.parent.name in allowed_design_ids]
    else:
        imp_paths = iter_layout_imp_paths(data_root)
        if allowed_design_ids is not None:
            imp_paths = [p for p in imp_paths if p.parent.name in allowed_design_ids]
    for path in tqdm(imp_paths, desc="Impedance → layouts"):
        design_id = path.parent.name
        log_data = np.log(np.maximum(np.load(path).astype(np.float32), 1e-10))
        z = ((log_data - imp_log_mean) / imp_log_std).flatten()
        sub = layout_dir(output_root, design_id)
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "imp.npy", z[np.newaxis].astype(np.float32))


def copy_layout_occupancy(
    data_root: Path,
    output_root: Path,
    *,
    allowed_design_ids: set[str] | None = None,
) -> None:
    out_layouts = layouts_root(output_root)
    out_layouts.mkdir(parents=True, exist_ok=True)
    if MANIFEST_LAYOUTS_ONLY and manifest_path(data_root).is_file():
        design_ids = sorted(manifest_design_ids(data_root))
        if allowed_design_ids is not None:
            design_ids = [did for did in design_ids if did in allowed_design_ids]
        print(f"\n[4/4] Copying occupancy for {len(design_ids)} manifest layouts...")
        iterable = design_ids
        desc = "Occupancy"
    else:
        occ_files = iter_layout_occ_paths(data_root)
        print(f"\n[4/4] Copying {len(occ_files)} occupancy layouts...")
        iterable = occ_files
        desc = "Occupancy"

    for item in tqdm(iterable, desc=desc):
        if isinstance(item, str):
            design_id = item
            occ_path = layout_occ_path(data_root, design_id)
            if not occ_path.is_file():
                continue
        else:
            occ_path = item
            design_id = occ_path.parent.name
        sub = layout_dir(output_root, design_id)
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "occ.npy", np.load(occ_path).astype(np.float32))


def validate_pifreq(pifreq_dir: Path, expected_mhz=None) -> dict:
    if not pifreq_dir.is_dir():
        return {}
    if expected_mhz is None:
        expected_mhz = [float(x) for x in load_anchors_mhz(DATA_DIR)]
    files = sorted(pifreq_dir.glob("*.npy"))
    mhz_vals = [float(np.load(p)) / 1e6 for p in files]
    uniq = sorted(set(round(m, 1) for m in mhz_vals))
    print(f"  PI_freq: {len(files)} files, MHz (rounded): {uniq}")
    return {"count": len(files), "unique_mhz_rounded": uniq, "expected_anchors_mhz": expected_mhz}


def copy_pifreq(
    pifreq_dir: Path,
    output_pifreq_dir: Path,
    *,
    allowed_stems: set[str] | None = None,
) -> None:
    if not pifreq_dir.is_dir():
        return
    output_pifreq_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pifreq_dir.glob("*.npy"))
    if allowed_stems is not None:
        files = [p for p in files if p.stem in allowed_stems]
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
    _maybe_repair_dataset(data_root)
    validate_multifreq_dataset(data_root)
    stats_file = output_root / "normalization_stats.json"
    if not stats_file.is_file():
        raise FileNotFoundError(f"{stats_file} missing — run full pipeline first (APPEND=False)")

    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    heatmap_stats = stats["Heatmap"]
    if USE_GLOBAL_MAX_HEATMAP != _is_gmax_heatmap_stats(heatmap_stats):
        raise ValueError(
            "Append mode mismatch: USE_GLOBAL_MAX_HEATMAP="
            f"{USE_GLOBAL_MAX_HEATMAP} but existing stats are "
            f"{'global_max' if _is_gmax_heatmap_stats(heatmap_stats) else 'log_zscore'}"
        )
    imp_stats = stats.get("Impedance", {})
    missing = _missing_heatmap_names(data_root, output_root)
    if MAX_K is not None:
        design_k, allowed_design_ids = _allowed_design_ids(data_root, MAX_K)
        manifest_index = load_manifest_index(data_root)
        missing = [
            name
            for name in missing
            if manifest_index.get(Path(name).stem) in allowed_design_ids
        ]
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
        np.save(out_heatmap / name, _normalize_raw_heatmap_to_array(data, heatmap_stats))

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
        with manifest_path(data_root).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if MAX_K is not None:
            _, allowed_design_ids = _allowed_design_ids(data_root, MAX_K)
            rows = _filter_manifest_rows(rows, allowed_design_ids)
        out_manifest = manifest_path(output_root)
        with out_manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    stats["PI_freq"] = stats.get("PI_freq", {})
    stats["PI_freq"]["anchor_mhz"] = [float(x) for x in load_anchors_mhz(DATA_DIR)]
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    invalidate_training_caches(output_root)
    if PRUNE_ORPHAN_LAYOUTS and manifest_path(output_root).is_file():
        n_pruned = prune_orphan_layouts(output_root)
        if n_pruned:
            print(f"  Pruned {n_pruned} orphan layout folder(s) not in manifest.csv")
    write_dataset_meta(
        output_root,
        stage="global_max" if _is_gmax_heatmap_stats(heatmap_stats) else "normalized",
        source_script="pipelines/normalize/multifreq.py",
        extra={
            "mode": "append",
            "new_heatmap_samples": len(missing),
            "heatmap_norm": "global_max" if _is_gmax_heatmap_stats(heatmap_stats) else "log_zscore",
        },
    )
    return len(missing)


def run_full_pipeline(data_root: Path, output_root: Path) -> None:
    _maybe_repair_dataset(data_root)
    validate_multifreq_dataset(data_root)
    prepare_output_dir(output_root, overwrite=OVERWRITE_OUTPUT)

    allowed_design_ids: set[str] | None = None
    heatmap_files: list[Path] | None = None
    allowed_stems: set[str] | None = None
    manifest_index = load_manifest_index(data_root)

    if MAX_K is not None:
        design_k, allowed_design_ids = _allowed_design_ids(data_root, MAX_K)
        heatmap_files = _filter_heatmap_paths(data_root / "heatmap", manifest_index, allowed_design_ids)
        allowed_stems = {p.stem for p in heatmap_files}
        with manifest_path(data_root).open(newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        rows_after = len(_filter_manifest_rows(all_rows, allowed_design_ids))
        _print_k_filter_summary(
            design_k,
            allowed_design_ids,
            max_k=MAX_K,
            manifest_rows_before=len(all_rows),
            manifest_rows_after=rows_after,
        )
        if not heatmap_files:
            raise ValueError(f"No heatmaps left after MAX_K={MAX_K} filter")

    if USE_GLOBAL_MAX_HEATMAP:
        heatmap_stats = calculate_heatmap_stats_gmax(
            data_root / "heatmap",
            global_max_ohm=GLOBAL_MAX_OHM,
            gmax_percentile=GMAX_PERCENTILE,
            bg_ohm=BG_OHM,
            clip_max=CLIP_MAX,
            heatmap_files=heatmap_files,
        )
    else:
        if USE_ROBUST_PER_MHZ:
            heatmap_stats = calculate_heatmap_stats_robust_per_mhz(
                data_root / "heatmap", data_root,
                heatmap_files=heatmap_files,
            )
        else:
            heatmap_stats = calculate_heatmap_stats(
                data_root / "heatmap",
                heatmap_files=heatmap_files,
            )
    stem_mhz = _load_stem_to_mhz(data_root) if _is_robust_per_mhz_stats(heatmap_stats) else None
    normalize_heatmaps(
        data_root / "heatmap",
        output_root / "heatmap",
        heatmap_stats,
        stem_mhz=stem_mhz,
        heatmap_files=heatmap_files,
    )
    imp_stats = calculate_impedance_stats(data_root, allowed_design_ids=allowed_design_ids)
    normalize_layout_impedance(
        data_root, output_root, imp_stats, allowed_design_ids=allowed_design_ids,
    )
    copy_layout_occupancy(data_root, output_root, allowed_design_ids=allowed_design_ids)
    pifreq_stats = validate_pifreq(data_root / "PI_freq")
    copy_pifreq(data_root / "PI_freq", output_root / "PI_freq", allowed_stems=allowed_stems)

    manifest_info = {}
    src_manifest = manifest_path(data_root)
    if src_manifest.is_file():
        with src_manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if allowed_design_ids is not None:
            rows = _filter_manifest_rows(rows, allowed_design_ids)
        design_ids = {r["design_id"] for r in rows if r.get("design_id")}
        manifest_info = {
            "rows": len(rows),
            "unique_design_id": len(design_ids),
            "freq_mhz_counts": dict(Counter(r.get("freq_mhz", "") for r in rows if r.get("freq_mhz"))),
        }
        if MAX_K is not None:
            manifest_info["max_k_filter"] = MAX_K
        out_manifest = manifest_path(output_root)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  manifest.csv: {manifest_info['rows']} rows, {manifest_info['unique_design_id']} layouts")

    if _is_gmax_heatmap_stats(heatmap_stats):
        bg_top = 0.0
    else:
        bg_top = float(heatmap_stats["background_value"])
    stats = {
        "background_value": bg_top,
        "Heatmap": heatmap_stats,
        "Impedance": imp_stats,
        "PI_freq": {
            "min_hz": 1e6,
            "max_hz": 600e6,
            "log10_min": 6.0,
            "log10_range": float(np.log10(600e6) - np.log10(1e6)),
            "anchor_mhz": [float(x) for x in load_anchors_mhz(DATA_DIR)],
            **pifreq_stats,
        },
        "multifreq_manifest": manifest_info,
        "storage": "layout_store",
    }
    if MAX_K is not None:
        stats["max_k_filter"] = MAX_K
    stats_path = output_root / "normalization_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    invalidate_training_caches(output_root)
    if PRUNE_ORPHAN_LAYOUTS and manifest_path(output_root).is_file():
        n_pruned = prune_orphan_layouts(output_root)
        if n_pruned:
            print(f"  Pruned {n_pruned} orphan layout folder(s) not in manifest.csv")
    if _is_gmax_heatmap_stats(heatmap_stats):
        hm_mode = "global_max"
    elif _is_robust_per_mhz_stats(heatmap_stats):
        hm_mode = heatmap_stats.get("norm_mode", "robust_log1p_per_mhz")
    else:
        hm_mode = heatmap_stats.get("norm_mode", "log_zscore")
    write_dataset_meta(
        output_root,
        stage="global_max" if USE_GLOBAL_MAX_HEATMAP else "normalized",
        source_script="pipelines/normalize/multifreq.py",
        extra={
            "mode": "full",
            "source_dataset": str(data_root),
            "heatmap_norm": hm_mode,
            **({"max_k_filter": MAX_K} if MAX_K is not None else {}),
        },
    )
    print(f"\n{'=' * 60}\n✓ NORMALIZATION COMPLETE ({hm_mode})\n{'=' * 60}")
    print(f"Dataset: {output_root}\nStats: {stats_path}")
    if USE_GLOBAL_MAX_HEATMAP:
        print(f"  Heatmap global_max_ohm = {heatmap_stats['global_max_ohm']:.4f} Ω")


def main() -> None:
    data_root = DATA_DIR.resolve()
    output_root = _output_dir()

    if APPEND:
        n = normalize_append(data_root, output_root)
        print(f"\n✓ Append complete ({n} samples)")
        return

    if not data_root.exists():
        raise SystemExit(f"Input missing: {data_root}")
    if USE_GLOBAL_MAX_HEATMAP:
        hm_mode = "global_max (Ω/gmax)"
    elif USE_ROBUST_PER_MHZ:
        hm_mode = (
            "robust log(1+x) median/IQR per-MHz (unbounded z)"
            if USE_UNBOUNDED_Z
            else "robust log(1+x) median/IQR per-MHz"
        )
    else:
        hm_mode = (
            "log(1+x) z-score (unbounded z)"
            if USE_UNBOUNDED_Z
            else "log(1+x) z-score"
        )
    print(f"\n{'=' * 60}\nNORMALIZATION (layout store)\n{'=' * 60}")
    print(f"Input:       {data_root}")
    print(f"Output:      {output_root}")
    print(f"Heatmaps:    {hm_mode}")
    if MAX_K is not None:
        print(f"K filter:    keep layouts with K <= {MAX_K}")
    if USE_GLOBAL_MAX_HEATMAP:
        gmax_note = (
            f"scan p{GMAX_PERCENTILE}"
            if GLOBAL_MAX_OHM is None
            else f"fixed {GLOBAL_MAX_OHM} Ω"
        )
        print(f"Gmax:        {gmax_note}  bg={BG_OHM} Ω  clip={CLIP_MAX}")
    run_full_pipeline(data_root, output_root)


if __name__ == "__main__":
    main()
