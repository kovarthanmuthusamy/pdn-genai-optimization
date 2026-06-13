"""Layout-centric storage for multifreq PI datasets.

Per layout (design_id): ``layouts/{design_id}/imp.npy``, ``occ.npy`` (once).
Per (layout, MHz) row: ``heatmap/sample_N.npy``, ``PI_freq/sample_N.npy``, ``manifest.csv``.
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
            "Build with Data_Creation/Data_processing_multifreq.py"
        )
    n_hm = len(list((root / "heatmap").glob("*.npy")))
    manifest = load_manifest_index(root)
    if len(manifest) != n_hm:
        raise ValueError(
            f"manifest.csv rows ({len(manifest)}) != heatmap files ({n_hm}) under {root}"
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
