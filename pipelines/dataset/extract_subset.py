#!/usr/bin/env python3
"""Extract a layout subset from full multifreq into a smaller folder.

Purpose:
    Match layouts present in a reference dataset, then copy or symlink all multifreq rows
    (every MHz anchor) for those layouts from source to destination.

Run:
    python pipelines/dataset/extract_subset.py

Agent notes:
    - What: Build a train/eval split folder by layout ID (not by manifest row filter).
    - Usage: Defaults to dry-run. Set ``EXECUTE=True`` to write ``DST``. Use ``OVERWRITE`` or ``RESUME``, not both.
    - Config keys:
        - ``SRC`` — full multifreq source (e.g. ``datasets/data_multifreq``)
        - ``REF`` — reference with ``layouts/`` to match (e.g. ``data_multifreq_norm``)
        - ``DST`` — output folder to create
        - ``EXECUTE`` — ``False`` = preview only; ``True`` = copy/symlink
        - ``HARD_COPY`` — copy files instead of symlinks
        - ``OVERWRITE`` — wipe ``DST`` before extract
        - ``RESUME`` — fill missing files only
    - Key symbol: ``extract_subset``
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from repo_paths import REPO_ROOT as _ROOT, setup_path
setup_path()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from libs.dataset_meta import write_dataset_meta  # noqa: E402
from src_vae.others.multifreq_layout_store import (  # noqa: E402
    invalidate_training_caches,
    layout_dir,
    layouts_root,
    manifest_path,
)

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/dataset/extract_subset.py
# =============================================================================

SRC = _ROOT / "datasets" / "data_multifreq"
REF = _ROOT / "datasets" / "data_multifreq_norm_z_score"

# How to pick layout IDs from REF:
#   "manifest" — unique design_id in ref/manifest.csv (matches training rows; recommended)
#   "layouts"  — every folder under ref/layouts/ (can include orphans not in manifest)
#   "intersect" — must appear in both manifest and layouts/
REF_MATCH = "manifest"

DST = _ROOT / "datasets" / "data_multifreq_train"

EXECUTE = False
HARD_COPY = False

# If DST exists: remove and rebuild (use when subset definition changed)
OVERWRITE = True 

# If DST exists: add missing heatmap/PI_freq/layout files and refresh manifest
RESUME = False

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


def _layout_ids_from_ref(ref: Path, match: str = REF_MATCH) -> set[str]:
    """Reference layout IDs for subset extraction."""
    manifest_ids: set[str] = set()
    mp = manifest_path(ref)
    if mp.is_file():
        with mp.open(newline="", encoding="utf-8") as f:
            manifest_ids = {row["design_id"] for row in csv.DictReader(f) if row.get("design_id")}

    layout_ids: set[str] = set()
    layouts = layouts_root(ref)
    if layouts.is_dir():
        layout_ids = {p.name for p in layouts.iterdir() if p.is_dir()}

    match = (match or REF_MATCH).strip().lower()
    if match == "manifest":
        if manifest_ids:
            extra = layout_ids - manifest_ids
            if extra:
                print(
                    f"  ref/layouts has {len(extra):,} folder(s) not in manifest "
                    f"(using {len(manifest_ids):,} manifest design_id only)"
                )
            return manifest_ids
        if layout_ids:
            return layout_ids
    elif match == "layouts":
        if layout_ids:
            missing = manifest_ids - layout_ids
            if manifest_ids and missing:
                print(f"  Warning: {len(missing):,} manifest design_id(s) missing ref/layouts/")
            return layout_ids
        if manifest_ids:
            return manifest_ids
    elif match == "intersect":
        if manifest_ids and layout_ids:
            both = manifest_ids & layout_ids
            print(
                f"  intersect: {len(both):,} layouts "
                f"(manifest={len(manifest_ids):,}, folders={len(layout_ids):,})"
            )
            return both
        return manifest_ids or layout_ids

    raise FileNotFoundError(
        f"Reference has no layouts/ and no manifest: {ref}\n"
        "Expected ref/layouts/<design_id>/ or ref/manifest.csv",
    )


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _link_or_copy(src: Path, dst: Path, *, hard_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if hard_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _link_or_copy_layout_dir(
    src_sub: Path,
    dst_sub: Path,
    *,
    hard_copy: bool,
    resume: bool,
) -> None:
    dst_sub.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_sub.iterdir()):
        if not item.is_file():
            continue
        dst_item = dst_sub / item.name
        if resume and dst_item.exists():
            continue
        _link_or_copy(item, dst_item, hard_copy=hard_copy)


def _prepare_dst(dst: Path, *, overwrite: bool, resume: bool) -> str:
    """Return action taken when destination already exists."""
    if not dst.exists():
        dst.mkdir(parents=True)
        for sub in ("heatmap", "PI_freq", "layouts"):
            (dst / sub).mkdir(exist_ok=True)
        return "create"

    existing_manifest = manifest_path(dst)
    existing_rows = 0
    if existing_manifest.is_file():
        existing_rows = sum(1 for _ in _load_manifest_rows(existing_manifest))

    if overwrite:
        print(f"\nRemoving existing destination ({existing_rows:,} manifest rows): {dst}")
        shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for sub in ("heatmap", "PI_freq", "layouts"):
            (dst / sub).mkdir(exist_ok=True)
        return "overwrite"

    if resume:
        for sub in ("heatmap", "PI_freq", "layouts"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
        print(f"\nResuming into existing destination ({existing_rows:,} manifest rows)")
        return "resume"

    raise FileExistsError(
        f"Destination already exists: {dst} ({existing_rows:,} manifest rows). "
        "Set OVERWRITE=True to replace, or RESUME=True to add missing files."
    )


def extract_subset(
    *,
    src: Path,
    ref: Path,
    dst: Path,
    execute: bool,
    hard_copy: bool,
    overwrite: bool = False,
    resume: bool = False,
    ref_match: str | None = None,
) -> dict:
    src_manifest = manifest_path(src)
    if not src_manifest.is_file():
        raise FileNotFoundError(f"Missing source manifest: {src_manifest}")

    match_mode = (ref_match or REF_MATCH).strip().lower()
    ref_layout_ids = _layout_ids_from_ref(ref, match_mode)
    src_rows = _load_manifest_rows(src_manifest)

    # All raw manifest rows for matched layouts (every MHz present in source).
    kept = [r for r in src_rows if r.get("design_id") in ref_layout_ids]
    kept_designs = {r["design_id"] for r in kept}
    missing_in_src = sorted(ref_layout_ids - kept_designs)

    summary: dict = {
        "src": str(src.resolve()),
        "ref": str(ref.resolve()),
        "dst": str(dst.resolve()),
        "match_mode": match_mode,
        "ref_layouts": len(ref_layout_ids),
        "src_rows_total": len(src_rows),
        "rows_kept": len(kept),
        "layouts_kept": len(kept_designs),
        "missing_layouts_in_src": len(missing_in_src),
        "hard_copy": hard_copy,
        "mode": "link" if not hard_copy else "copy",
    }

    print("=" * 60)
    print("EXTRACT multifreq subset (layout match → all raw data)")
    print("=" * 60)
    print(f"Source : {src}")
    print(f"Ref    : {ref}  ({len(ref_layout_ids):,} layouts, match={match_mode})")
    print(f"Dest   : {dst}")
    print(f"Mode   : {'EXECUTE' if execute else 'DRY-RUN'}  ({summary['mode']})")
    print(f"Rows   : {len(src_rows):,} → {len(kept):,}  (all MHz for matched layouts)")
    print(f"Layouts: {len(kept_designs):,} / {len(ref_layout_ids):,}")

    if missing_in_src:
        print(f"\nERROR: {len(missing_in_src)} ref layout(s) have no rows in source manifest:")
        for did in missing_in_src[:10]:
            print(f"  {did}")
        raise SystemExit(1)

    kept_freq = Counter(float(r["freq_mhz"]) for r in kept if r.get("freq_mhz"))
    rows_per_layout = Counter(r["design_id"] for r in kept)
    freq_counts = sorted(rows_per_layout.values())
    min_r = freq_counts[0] if freq_counts else 0
    max_r = freq_counts[-1] if freq_counts else 0

    print(f"\nMHz anchors in extracted subset ({len(kept_freq)} unique):")
    for mhz in sorted(kept_freq):
        print(f"  {mhz:g} MHz: {kept_freq[mhz]:,} rows")
    print(f"Rows per layout: min={min_r:,}  max={max_r:,}  mean={len(kept) / max(len(kept_designs), 1):.1f}")

    if not execute:
        if dst.exists():
            existing_rows = 0
            if manifest_path(dst).is_file():
                existing_rows = len(_load_manifest_rows(manifest_path(dst)))
            print(
                f"\nDestination exists ({existing_rows:,} rows). "
                "Set EXECUTE=True with OVERWRITE=True to replace, or EXECUTE=True with RESUME=True to fill gaps."
            )
        else:
            print("\nDry-run only. Set EXECUTE=True in the CONFIG block to create the folder.")
        return summary

    if overwrite and resume:
        raise ValueError("Set only one of OVERWRITE or RESUME, not both.")

    action = _prepare_dst(dst, overwrite=overwrite, resume=resume)
    summary["dst_action"] = action

    missing_hm = missing_pf = missing_layout = 0
    skipped_hm = skipped_pf = 0
    for row in tqdm(kept, desc="heatmap / PI_freq"):
        name = row["sample_name"]
        hm_src = src / "heatmap" / name
        pf_src = src / "PI_freq" / name
        hm_dst = dst / "heatmap" / name
        pf_dst = dst / "PI_freq" / name

        if not hm_src.is_file():
            missing_hm += 1
            continue
        if resume and hm_dst.is_file():
            skipped_hm += 1
        else:
            _link_or_copy(hm_src, hm_dst, hard_copy=hard_copy)

        if pf_src.is_file():
            if resume and pf_dst.is_file():
                skipped_pf += 1
            else:
                _link_or_copy(pf_src, pf_dst, hard_copy=hard_copy)
        else:
            missing_pf += 1

    for did in tqdm(sorted(kept_designs), desc="layouts"):
        src_sub = layout_dir(src, did)
        dst_sub = layout_dir(dst, did)
        if not src_sub.is_dir():
            missing_layout += 1
            continue
        _link_or_copy_layout_dir(src_sub, dst_sub, hard_copy=hard_copy, resume=resume)

    with manifest_path(dst).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    invalidate_training_caches(dst)
    write_dataset_meta(
        dst,
        stage="raw",
        source_script="pipelines/dataset/extract_subset.py",
        extra={
            "subset_of": str(src),
            "ref_layouts_from": str(ref),
            "ref_layout_count": len(ref_layout_ids),
        },
    )

    meta = {
        **summary,
        "mhz_in_subset": {f"{mhz:g}": kept_freq[mhz] for mhz in sorted(kept_freq)},
        "created": datetime.now(timezone.utc).isoformat(),
        "missing_heatmap": missing_hm,
        "missing_pi_freq": missing_pf,
        "missing_layouts": missing_layout,
        "skipped_existing_heatmap": skipped_hm,
        "skipped_existing_pi_freq": skipped_pf,
        "next_steps": [
            "pipelines/normalize/multifreq.py: DATA_DIR=data_multifreq_train, APPEND=False (first build)",
            "pipelines/normalize/multifreq.py: APPEND=True only when adding new raw rows later",
        ],
    }
    note_path = dst / "subset_meta.json"
    note_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone → {dst}")
    print(f"  manifest rows : {len(kept):,}")
    print(f"  layouts       : {len(kept_designs):,}")
    print(f"  MHz anchors   : {len(kept_freq)}")
    print(f"  missing hm/pf : {missing_hm} / {missing_pf}")
    if action == "resume":
        print(f"  skipped hm/pf : {skipped_hm} / {skipped_pf}")
    print(f"  meta          : {note_path}")
    return meta


def main() -> None:
    extract_subset(
        src=SRC.resolve(),
        ref=REF.resolve(),
        dst=DST.resolve(),
        execute=EXECUTE,
        hard_copy=HARD_COPY,
        overwrite=OVERWRITE,
        resume=RESUME,
        ref_match=REF_MATCH,
    )


if __name__ == "__main__":
    main()
