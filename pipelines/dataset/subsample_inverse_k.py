#!/usr/bin/env python3
"""Inverse-K exponential subsampling of multifreq dataset layouts.

Run:
    python pipelines/dataset/subsample_inverse_k.py"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from repo_paths import REPO_ROOT as _ROOT, setup_path
setup_path()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src_vae.others.multifreq_layout_store import (  # noqa: E402
    invalidate_training_caches,
    layout_occ_path,
    manifest_path,
)

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/dataset/subsample_inverse_k.py
# =============================================================================

DATA_DIR = _ROOT / "datasets" / "data_multifreq_norm"  # multifreq dataset to subsample
EXECUTE = False  # True = delete layouts; False = dry-run only
SEED = 42
N_REF = 1000.0  # reference layout count at K_ANCHOR
TAU = 12.0  # decay for K > K_LOW_MAX (sharp drop at high K)
TAU_LOW = 35.0  # decay for K <= K_LOW_MAX (keep more low/mid K)
K_LOW_MAX = 20
K_ANCHOR = 2
N_MIN = 150  # min layouts kept per K
N_MAX = 1000  # max layouts kept per K
KEEP_EDGE_K = False  # True = include edge K {0,1,51,52} in inverse-exp
REPORT_PATH: Path | None = None  # None = <data-dir>/subsample_inverse_k_report.json

# =============================================================================

MANIFEST_FIELDS = [
    "sample_name",
    "design_id",
    "freq_label",
    "freq_mhz",
    "freq_hz",
    "source_folder",
    "pi_number",
    "decap_index",
]

# K buckets the user treats as out-of-scope for uniform 1000/combo generation.
DEFAULT_EDGE_K = frozenset({0, 1, 51, 52})


def _k_from_occ(data_dir: Path, design_id: str) -> int:
    occ = np.load(layout_occ_path(data_dir, design_id), mmap_mode="r").reshape(-1)
    return int((occ > 0.5).sum())


def load_design_k(data_dir: Path) -> dict[str, int]:
    """design_id → decap count K (from shared occupancy)."""
    seen: dict[str, int] = {}
    with manifest_path(data_dir).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = row["design_id"]
            if did not in seen:
                seen[did] = _k_from_occ(data_dir, did)
    return seen


def target_layouts_per_k(
    k_values: list[int],
    *,
    n_ref: float,
    tau: float,
    tau_low: float,
    k_low_max: int,
    k_anchor: int,
    n_min: int,
    n_max: int,
) -> dict[int, int]:
    """Inverse-exponential keep budget per K (capped by available count).

    For K <= k_low_max uses ``tau_low`` (gentler decay → keep more low/mid K).
    For K > k_low_max uses ``tau`` (sharper drop toward n_min at high K).
    """
    avail = Counter(k_values)
    targets: dict[int, int] = {}
    for k in sorted(avail):
        tau_eff = tau_low if k <= k_low_max else tau
        raw = round(n_ref * math.exp(-(k - k_anchor) / tau_eff))
        n = max(n_min, min(n_max, raw, avail[k]))
        targets[k] = n
    return targets


def select_designs_to_keep(
    design_k: dict[str, int],
    targets: dict[int, int],
    *,
    seed: int,
    drop_edge_k: bool,
    edge_k: frozenset[int],
) -> tuple[set[str], set[str], dict[int, dict[str, int]]]:
    """Return (keep_set, delete_set, per_k_stats)."""
    by_k: dict[int, list[str]] = defaultdict(list)
    for did, k in design_k.items():
        by_k[k].append(did)

    rng = random.Random(seed)
    keep: set[str] = set()
    stats: dict[int, dict[str, int]] = {}

    for k in sorted(by_k):
        designs = sorted(by_k[k])
        avail = len(designs)
        if drop_edge_k and k in edge_k:
            n_keep = 0
        else:
            n_keep = targets.get(k, 0)
            n_keep = min(n_keep, avail)

        if n_keep >= avail:
            chosen = designs
        else:
            chosen = rng.sample(designs, n_keep)

        keep.update(chosen)
        stats[k] = {
            "available": avail,
            "target": targets.get(k, 0),
            "kept": len(chosen),
            "deleted": avail - len(chosen),
        }

    delete = set(design_k) - keep
    return keep, delete, stats


def load_manifest_rows(data_dir: Path) -> list[dict[str, str]]:
    with manifest_path(data_dir).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stems_for_designs(rows: list[dict[str, str]], design_ids: set[str]) -> set[str]:
    stems: set[str] = set()
    for row in rows:
        if row["design_id"] in design_ids:
            stems.add(Path(row["sample_name"]).stem)
    return stems


def delete_layout_files(
    data_dir: Path,
    delete_designs: set[str],
    delete_stems: set[str],
    *,
    execute: bool,
) -> dict[str, int]:
    """Remove layout dirs and per-MHz npy files."""
    counts = {
        "layout_dirs": 0,
        "heatmap": 0,
        "pi_freq": 0,
        "errors": 0,
    }
    layouts_root = data_dir / "layouts"
    hm_dir = data_dir / "heatmap"
    pf_dir = data_dir / "PI_freq"

    for did in sorted(delete_designs):
        layout_sub = layouts_root / did
        if layout_sub.is_dir():
            counts["layout_dirs"] += 1
            if execute:
                try:
                    shutil.rmtree(layout_sub)
                except OSError:
                    counts["errors"] += 1

    for stem in sorted(delete_stems):
        for sub, key in ((hm_dir, "heatmap"), (pf_dir, "pi_freq")):
            p = sub / f"{stem}.npy"
            if p.is_file():
                counts[key] += 1
                if execute:
                    try:
                        p.unlink()
                    except OSError:
                        counts["errors"] += 1

    return counts


def write_manifest(
    data_dir: Path,
    rows: list[dict[str, str]],
    keep_designs: set[str],
    *,
    execute: bool,
) -> int:
    kept_rows = [r for r in rows if r["design_id"] in keep_designs]
    if not execute:
        return len(kept_rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    src = manifest_path(data_dir)
    backup = data_dir / f"manifest.csv.bak_{ts}"
    shutil.copy2(src, backup)
    print(f"  manifest backup → {backup.name}")

    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept_rows)
    return len(kept_rows)


def print_plan(
    design_k: dict[str, int],
    stats: dict[int, dict[str, int]],
    n_rows_before: int,
    n_rows_after: int,
    file_counts: dict[str, int],
) -> None:
    print("\n=== Per-K layout allocation ===")
    print(f"{'K':>4}  {'avail':>6}  {'target':>6}  {'keep':>6}  {'del':>6}")
    total_a = total_t = total_k = total_d = 0
    for k in sorted(stats):
        s = stats[k]
        total_a += s["available"]
        total_t += s["target"]
        total_k += s["kept"]
        total_d += s["deleted"]
        print(
            f"{k:4d}  {s['available']:6d}  {s['target']:6d}  "
            f"{s['kept']:6d}  {s['deleted']:6d}",
        )
    print(
        f"{'TOT':>4}  {total_a:6d}  {total_t:6d}  {total_k:6d}  {total_d:6d}",
    )
    print(f"\nManifest rows: {n_rows_before:,} → {n_rows_after:,}")
    print(
        f"Files to remove: {file_counts['layout_dirs']:,} layout dirs, "
        f"{file_counts['heatmap']:,} heatmaps, {file_counts['pi_freq']:,} PI_freq",
    )


def main() -> None:
    data_dir = DATA_DIR.resolve()
    if not manifest_path(data_dir).is_file():
        raise SystemExit(f"Missing manifest: {manifest_path(data_dir)}")

    print(f"Dataset: {data_dir}")
    print(
        f"Policy: n_keep(K) = clamp({N_MIN}, {N_MAX}, "
        f"round({N_REF} * exp(-(K - {K_ANCHOR}) / τ_eff)))",
    )
    print(
        f"  τ_eff = {TAU_LOW} for K <= {K_LOW_MAX}, "
        f"τ_eff = {TAU} for K > {K_LOW_MAX}",
    )
    if KEEP_EDGE_K:
        print(f"Edge K {sorted(DEFAULT_EDGE_K)}: included in inverse-exp")
    else:
        print(f"Edge K {sorted(DEFAULT_EDGE_K)}: drop all layouts (not in 1000/combo set)")

    design_k = load_design_k(data_dir)
    k_list = list(design_k.values())
    targets = target_layouts_per_k(
        k_list,
        n_ref=N_REF,
        tau=TAU,
        tau_low=TAU_LOW,
        k_low_max=K_LOW_MAX,
        k_anchor=K_ANCHOR,
        n_min=N_MIN,
        n_max=N_MAX,
    )

    keep, delete, stats = select_designs_to_keep(
        design_k,
        targets,
        seed=SEED,
        drop_edge_k=not KEEP_EDGE_K,
        edge_k=DEFAULT_EDGE_K,
    )

    rows = load_manifest_rows(data_dir)
    n_rows_before = len(rows)
    delete_stems = stems_for_designs(rows, delete)
    n_rows_after = sum(1 for r in rows if r["design_id"] in keep)

    file_counts = delete_layout_files(
        data_dir, delete, delete_stems, execute=False,
    )
    print_plan(design_k, stats, n_rows_before, n_rows_after, file_counts)

    mode = "EXECUTE" if EXECUTE else "DRY-RUN"
    print(f"\n[{mode}] layouts kept={len(keep):,}  deleted={len(delete):,}")

    report_path = REPORT_PATH or (data_dir / "subsample_inverse_k_report.json")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": not EXECUTE,
        "data_dir": str(data_dir),
        "params": {
            "n_ref": N_REF,
            "tau": TAU,
            "tau_low": TAU_LOW,
            "k_low_max": K_LOW_MAX,
            "k_anchor": K_ANCHOR,
            "n_min": N_MIN,
            "n_max": N_MAX,
            "seed": SEED,
            "keep_edge_k": KEEP_EDGE_K,
            "edge_k_dropped": sorted(DEFAULT_EDGE_K) if not KEEP_EDGE_K else [],
        },
        "summary": {
            "layouts_before": len(design_k),
            "layouts_kept": len(keep),
            "layouts_deleted": len(delete),
            "manifest_rows_before": n_rows_before,
            "manifest_rows_after": n_rows_after,
        },
        "per_k": stats,
        "targets": {str(k): v for k, v in sorted(targets.items())},
    }

    if not EXECUTE:
        print("\nDry-run only — no files changed. Set EXECUTE = True to apply.")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report → {report_path}")
        return

    print("\nApplying deletions …")
    file_counts = delete_layout_files(
        data_dir, delete, delete_stems, execute=True,
    )
    n_written = write_manifest(data_dir, rows, keep, execute=True)
    invalidate_training_caches(data_dir)
    print(
        f"  removed {file_counts['layout_dirs']} layout dirs, "
        f"{file_counts['heatmap']} heatmaps, {file_counts['pi_freq']} PI_freq",
    )
    if file_counts["errors"]:
        print(f"  WARNING: {file_counts['errors']} delete errors")
    print(f"  manifest rows written: {n_written:,}")
    print("  invalidated multifreq_meta.json + k_values_cache.npy")

    report["summary"]["files_removed"] = file_counts
    report["dry_run"] = False
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report → {report_path}")
    print("Done. Rebuild training caches on next dataloader load.")


if __name__ == "__main__":
    main()
