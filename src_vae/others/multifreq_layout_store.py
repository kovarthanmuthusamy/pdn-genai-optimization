"""Layout-centric on-disk layout for multifreq PI datasets.

Purpose: Path helpers, manifest indexing, validation, cache invalidation, and migration
    from legacy per-sample Imp/Occ_map to shared layout folders.
Run: Imported by ``dataloader.py`` and ``datasets/*.py`` maintenance scripts.
Inputs / outputs: Dataset root with ``manifest.csv``, ``layouts/{design_id}/``,
    ``heatmap/``, ``PI_freq/``; returns paths and design_id ↔ sample stem maps.
Dependencies: ``numpy``, ``csv``, ``shutil`` (stdlib).
Agent notes:
    - Type: library module (import-only).
    - Key symbols: ``has_layout_store``, ``load_manifest_index``, ``validate_multifreq_dataset``,
    - Config keys: none (library — import only)
      ``invalidate_training_caches``, ``migrate_to_layout_store``, ``layout_imp_path``, ``layout_occ_path``.
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.csv"
LAYOUTS_DIR_NAME = "layouts"


def manifest_path(data_dir: Path) -> Path:
    return Path(data_dir) / MANIFEST_NAME


def layouts_root(data_dir: Path) -> Path:
    return Path(data_dir) / LAYOUTS_DIR_NAME


def layout_dir(data_dir: Path, design_id: str) -> Path:
    return layouts_root(data_dir) / design_id


def layout_imp_path(data_dir: Path, design_id: str) -> Path:
    return layout_dir(data_dir, design_id) / "imp.npy"


def layout_occ_path(data_dir: Path, design_id: str) -> Path:
    return layout_dir(data_dir, design_id) / "occ.npy"


def has_layout_store(data_dir: Path) -> bool:
    root = Path(data_dir)
    return layouts_root(root).is_dir() and manifest_path(root).is_file()


def load_manifest_index(data_dir: Path) -> dict[str, str]:
    """Map heatmap stem (``sample_1``) → ``design_id``."""
    path = manifest_path(data_dir)
    if not path.is_file():
        return {}
    index: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("sample_name", "")
            did = row.get("design_id", "")
            if not name or not did:
                continue
            stem = Path(name).stem
            index[stem] = did
    return index


def load_manifest_rows(data_dir: Path) -> list[dict[str, str]]:
    """All manifest rows (may include duplicate ``sample_name``)."""
    path = manifest_path(data_dir)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_design_ids(data_dir: Path) -> set[str]:
    """Unique ``design_id`` values referenced in ``manifest.csv``."""
    return {
        str(row["design_id"])
        for row in load_manifest_rows(data_dir)
        if row.get("design_id")
    }


def manifest_layout_keys(data_dir: Path) -> dict[tuple[int, int], str]:
    """Map ``(pi_number, decap_index)`` → ``design_id`` from manifest."""
    out: dict[tuple[int, int], str] = {}
    for row in load_manifest_rows(data_dir):
        did = row.get("design_id")
        if not did:
            continue
        try:
            pi = int(row["pi_number"])
            di = int(row["decap_index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[(pi, di)] = str(did)
    return out


def manifest_pi_to_design_id(data_dir: Path) -> dict[int, str]:
    """Map ``pi_number`` → ``design_id`` when unique in manifest (fallback for append)."""
    by_pi: dict[int, list[str]] = {}
    for row in load_manifest_rows(data_dir):
        did = row.get("design_id")
        if not did:
            continue
        try:
            pi = int(row["pi_number"])
        except (KeyError, TypeError, ValueError):
            continue
        by_pi.setdefault(pi, []).append(str(did))
    return {pi: ids[0] for pi, ids in by_pi.items() if len(ids) == 1}


def resolve_manifest_design_id(
    sample: dict,
    by_pi_decap: dict[tuple[int, int], str],
    by_pi: dict[int, str],
) -> str | None:
    """Match a raw sample to an existing manifest ``design_id``."""
    try:
        pi = int(sample["pi_number"])
        di = int(sample["decap_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if (pi, di) in by_pi_decap:
        return by_pi_decap[(pi, di)]
    return by_pi.get(pi)


def iter_layout_imp_paths(data_dir: Path) -> list[Path]:
    """All ``layouts/*/imp.npy`` under ``data_dir``."""
    layouts = layouts_root(Path(data_dir))
    if not layouts.is_dir():
        return []
    return sorted(layouts.glob("*/imp.npy"))


def iter_layout_occ_paths(data_dir: Path) -> list[Path]:
    layouts = layouts_root(Path(data_dir))
    if not layouts.is_dir():
        return []
    return sorted(layouts.glob("*/occ.npy"))


def iter_manifest_layout_imp_paths(data_dir: Path) -> list[Path]:
    """``layouts/{design_id}/imp.npy`` for manifest ``design_id`` values only."""
    ids = manifest_design_ids(data_dir)
    if not ids:
        return iter_layout_imp_paths(data_dir)
    root = Path(data_dir)
    paths: list[Path] = []
    missing: list[str] = []
    for did in sorted(ids):
        p = layout_imp_path(root, did)
        if p.is_file():
            paths.append(p)
        else:
            missing.append(did)
    if missing:
        print(
            f"  Warning: {len(missing)} manifest design_id(s) missing layouts/*/imp.npy "
            f"(first: {missing[0]})"
        )
    return paths


def prune_orphan_layouts(
    data_dir: Path,
    *,
    allowed_design_ids: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Remove ``layouts/{design_id}/`` folders not referenced in manifest."""
    root = Path(data_dir)
    layouts = layouts_root(root)
    if not layouts.is_dir():
        return 0
    allowed = allowed_design_ids if allowed_design_ids is not None else manifest_design_ids(root)
    if not allowed:
        return 0
    removed = 0
    for sub in layouts.iterdir():
        if not sub.is_dir():
            continue
        if sub.name not in allowed:
            if not dry_run:
                shutil.rmtree(sub)
            removed += 1
    if removed and not dry_run:
        invalidate_training_caches(root)
    return removed


def filter_dataset_to_design_ids(
    data_dir: Path,
    allowed_design_ids: set[str],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Keep only rows/files for ``allowed_design_ids`` (manifest, heatmap, PI_freq, layouts)."""
    root = Path(data_dir)
    rows = load_manifest_rows(root)
    kept = [r for r in rows if r.get("design_id") in allowed_design_ids]
    kept_stems = {Path(r["sample_name"]).stem for r in kept if r.get("sample_name")}
    all_stems = _heatmap_stems(root)

    removed_hm = removed_pf = 0
    if not dry_run:
        hm_dir = root / "heatmap"
        pf_dir = root / "PI_freq"
        for stem in sorted(all_stems - kept_stems):
            hp = hm_dir / f"{stem}.npy"
            if hp.is_file():
                hp.unlink()
                removed_hm += 1
            if pf_dir.is_dir():
                pp = pf_dir / f"{stem}.npy"
                if pp.is_file():
                    pp.unlink()
                    removed_pf += 1
        if rows:
            fieldnames = list(rows[0].keys())
            with manifest_path(root).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)
        prune_orphan_layouts(root, allowed_design_ids=allowed_design_ids, dry_run=False)
        invalidate_training_caches(root)

    return {
        "manifest_rows_before": len(rows),
        "manifest_rows_after": len(kept),
        "allowed_design_ids": len(allowed_design_ids),
        "removed_heatmap_files": removed_hm,
        "removed_pifreq_files": removed_pf,
        "dry_run": int(dry_run),
    }


def _heatmap_stems(data_dir: Path) -> set[str]:
    hm = Path(data_dir) / "heatmap"
    if not hm.is_dir():
        return set()
    return {p.stem for p in hm.glob("*.npy")}


def repair_multifreq_dataset(
    data_dir: Path,
    *,
    prune_orphan_heatmaps: bool = True,
    prune_orphan_manifest: bool = True,
    prune_orphan_pifreq: bool = True,
    dedupe_manifest: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """Align ``heatmap/``, ``PI_freq/``, and ``manifest.csv`` (in-place).

    Orphan heatmaps (no manifest row) cannot be trained and are removed when
    ``prune_orphan_heatmaps`` is True. Manifest rows without a heatmap file are dropped.
    """
    root = Path(data_dir)
    hm_stems = _heatmap_stems(root)
    rows = load_manifest_rows(root)
    if not rows and not hm_stems:
        return {"manifest_rows_before": 0, "heatmap_files_before": 0}

    manifest_stems: set[str] = set()
    kept_rows: list[dict[str, str]] = []
    dupes_dropped = 0
    for row in rows:
        name = row.get("sample_name", "")
        if not name:
            continue
        stem = Path(name).stem
        if dedupe_manifest and stem in manifest_stems:
            dupes_dropped += 1
            continue
        manifest_stems.add(stem)
        kept_rows.append(row)

    orphan_hm = sorted(hm_stems - manifest_stems)
    orphan_manifest = sorted(manifest_stems - hm_stems)
    if prune_orphan_manifest:
        kept_rows = [r for r in kept_rows if Path(r.get("sample_name", "")).stem in hm_stems]

    removed_hm = 0
    removed_pf = 0
    if prune_orphan_heatmaps and orphan_hm and not dry_run:
        hm_dir = root / "heatmap"
        pf_dir = root / "PI_freq"
        for stem in orphan_hm:
            hp = hm_dir / f"{stem}.npy"
            if hp.is_file():
                hp.unlink()
                removed_hm += 1
            if prune_orphan_pifreq and pf_dir.is_dir():
                pp = pf_dir / f"{stem}.npy"
                if pp.is_file():
                    pp.unlink()
                    removed_pf += 1

    manifest_rows_before = len(rows)
    manifest_written = 0
    if not dry_run and (
        orphan_hm
        or orphan_manifest
        or dupes_dropped
        or len(kept_rows) != manifest_rows_before
    ):
        mp = manifest_path(root)
        fieldnames = list(rows[0].keys()) if rows else [
            "sample_name", "design_id", "freq_label", "freq_mhz", "freq_hz",
            "source_folder", "pi_number", "decap_index",
        ]
        with mp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept_rows)
        manifest_written = len(kept_rows)
        invalidate_training_caches(root)

    hm_after = len(_heatmap_stems(root)) if not dry_run else len(hm_stems) - len(orphan_hm)
    stats = {
        "manifest_rows_before": manifest_rows_before,
        "manifest_rows_after": manifest_written or len(kept_rows),
        "heatmap_files_before": len(hm_stems),
        "heatmap_files_after": hm_after,
        "orphan_heatmaps": len(orphan_hm),
        "orphan_manifest_stems": len(orphan_manifest),
        "manifest_dupes_dropped": dupes_dropped,
        "removed_heatmap_files": removed_hm,
        "removed_pifreq_files": removed_pf,
        "dry_run": int(dry_run),
    }
    return stats


def validate_multifreq_dataset(data_dir: Path) -> None:
    """Raise if heatmap/manifest/layouts/PI_freq are not present for multifreq training."""
    root = Path(data_dir)
    missing = [
        name
        for name, ok in (
            ("heatmap/", (root / "heatmap").is_dir()),
            ("manifest.csv", manifest_path(root).is_file()),
            ("layouts/", layouts_root(root).is_dir()),
            ("PI_freq/", (root / "PI_freq").is_dir()),
        )
        if not ok
    ]
    if missing:
        raise FileNotFoundError(
            f"Multifreq dataset incomplete at {root} — missing: {', '.join(missing)}. "
            "Build with pipelines/data/Data_processing_multifreq.py"
        )
    n_hm = len(list((root / "heatmap").glob("*.npy")))
    manifest = load_manifest_index(root)
    n_manifest = len(manifest)
    if n_manifest != n_hm:
        raise ValueError(
            f"manifest unique samples ({n_manifest}) != heatmap files ({n_hm}) under {root}. "
            "Run repair_multifreq_dataset() or set REPAIR_DATASET=True in "
            "pipelines/normalize/multifreq.py"
        )


def invalidate_training_caches(data_dir: Path) -> None:
    root = Path(data_dir)
    for name in ("multifreq_meta.json", "k_values_cache.npy"):
        p = root / name
        if p.is_file():
            p.unlink()


def migrate_to_layout_store(
    data_dir: Path,
    *,
    prune_legacy: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Build ``layouts/{design_id}/`` from manifest + per-sample Imp/Occ_map."""
    root = Path(data_dir)
    manifest = manifest_path(root)
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}")

    imp_dir = root / "Imp"
    occ_dir = root / "Occ_map"
    layouts = layouts_root(root)
    layouts.mkdir(parents=True, exist_ok=True)

    by_design: dict[str, str] = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = row.get("design_id", "")
            name = row.get("sample_name", "")
            if did and name and did not in by_design:
                by_design[did] = Path(name).stem

    written = 0
    skipped = 0
    missing = 0
    for did, stem in sorted(by_design.items()):
        sub = layouts / did
        imp_out = sub / "imp.npy"
        occ_out = sub / "occ.npy"
        if imp_out.is_file() and occ_out.is_file():
            skipped += 1
            continue
        imp_src = imp_dir / f"{stem}.npy"
        occ_src = occ_dir / f"{stem}.npy"
        if not imp_src.is_file() or not occ_src.is_file():
            missing += 1
            continue
        if dry_run:
            written += 1
            continue
        sub.mkdir(parents=True, exist_ok=True)
        if not imp_out.is_file():
            shutil.copy2(imp_src, imp_out)
        if not occ_out.is_file():
            shutil.copy2(occ_src, occ_out)
        written += 1

    pruned = 0
    if prune_legacy and not dry_run:
        for d in (imp_dir, occ_dir):
            if not d.is_dir():
                continue
            for p in d.glob("*.npy"):
                p.unlink()
                pruned += 1

    if not dry_run and (written or pruned):
        invalidate_training_caches(root)

    return {
        "layouts": len(by_design),
        "written": written,
        "skipped_existing": skipped,
        "missing_src": missing,
        "pruned_legacy_files": pruned,
    }
