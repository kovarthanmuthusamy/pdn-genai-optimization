"""Write ``dataset_meta.json`` summarizing an on-disk training dataset."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

META_FILENAME = "dataset_meta.json"
SCHEMA_VERSION = 1


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _dataset_size_bytes(root: Path, subdirs: list[str]) -> int:
    """Sum known data subdirs plus loose files at dataset root."""
    total = sum(_dir_size_bytes(root / name) for name in subdirs)
    if root.is_dir():
        for p in root.iterdir():
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def _mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 2)


def _read_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    if not manifest.is_file():
        return []
    with manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _subdir_sizes_mb(root: Path, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        p = root / name
        if p.exists():
            out[name] = _mb(_dir_size_bytes(p))
    return out


def _freq_summary_from_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"pi_frequencies_mhz": [], "samples_per_mhz": {}, "samples_per_freq_label": {}}

    mhz_vals: list[float] = []
    per_mhz: Counter[str] = Counter()
    per_label: Counter[str] = Counter()

    for r in rows:
        if r.get("freq_mhz"):
            try:
                mhz = float(r["freq_mhz"])
                mhz_vals.append(mhz)
                per_mhz[str(int(mhz) if mhz == int(mhz) else mhz)] += 1
            except ValueError:
                pass
        if r.get("freq_label"):
            per_label[r["freq_label"]] += 1

    unique_mhz = sorted(set(mhz_vals))
    return {
        "pi_frequencies_mhz": unique_mhz,
        "samples_per_mhz": dict(sorted(per_mhz.items(), key=lambda x: float(x[0]))),
        "samples_per_freq_label": dict(sorted(per_label.items())),
    }


def _count_npy(dir_path: Path) -> int:
    return sum(1 for _ in dir_path.glob("*.npy")) if dir_path.is_dir() else 0


def _count_layout_dirs(layouts: Path) -> int:
    if not layouts.is_dir():
        return 0
    return sum(1 for p in layouts.iterdir() if p.is_dir())


def build_dataset_meta(
    data_root: Path | str,
    *,
    stage: str,
    source_script: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict for ``data_root`` (raw multifreq, legacy single-freq, or normalized)."""
    root = Path(data_root).resolve()
    manifest = root / "manifest.csv"
    rows = _read_manifest_rows(manifest)

    layouts_dir = root / "layouts"
    heatmap_dir = root / "heatmap"
    imp_dir = root / "Imp"
    occ_dir = root / "Occ_map"
    pifreq_dir = root / "PI_freq"

    is_layout_store = layouts_dir.is_dir() and manifest.is_file()
    is_legacy = heatmap_dir.is_dir() and not is_layout_store

    if is_layout_store:
        storage = "layout_store"
        subdirs = ["heatmap", "layouts", "PI_freq"]
        counts = {
            "manifest_rows": len(rows),
            "unique_layouts": len({r["design_id"] for r in rows if r.get("design_id")}),
            "heatmap_files": _count_npy(heatmap_dir),
            "pi_freq_files": _count_npy(pifreq_dir),
            "layout_directories": _count_layout_dirs(layouts_dir),
        }
    elif is_legacy:
        storage = "per_sample"
        subdirs = [n for n in ("heatmap", "Imp", "Occ_map") if (root / n).is_dir()]
        n_hm = _count_npy(heatmap_dir)
        counts = {
            "manifest_rows": len(rows) if rows else n_hm,
            "unique_layouts": None,
            "heatmap_files": n_hm,
            "impedance_files": _count_npy(imp_dir),
            "occupancy_files": _count_npy(occ_dir),
        }
    else:
        storage = "unknown"
        subdirs = [p.name for p in root.iterdir() if p.is_dir()]
        counts = {"manifest_rows": len(rows)}

    freq_info = _freq_summary_from_manifest(rows)
    if not freq_info["pi_frequencies_mhz"] and pifreq_dir.is_dir():
        # Fallback: infer anchors from PI_freq vectors (first file)
        samples = sorted(pifreq_dir.glob("sample_*.npy"))[:50]
        if samples:
            import numpy as np

            mhz_set: set[float] = set()
            for p in samples:
                try:
                    hz = float(np.load(p).reshape(-1)[0])
                    mhz_set.add(round(hz / 1e6, 3))
                except Exception:
                    pass
            freq_info["pi_frequencies_mhz"] = sorted(mhz_set)

    total_bytes = _dataset_size_bytes(root, subdirs)
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "source_script": source_script,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "storage": storage,
        "counts": counts,
        "size": {
            "total_bytes": total_bytes,
            "total_mb": _mb(total_bytes),
            "by_subdirectory_mb": _subdir_sizes_mb(root, subdirs),
        },
        **freq_info,
    }

    stats_path = root / "normalization_stats.json"
    if stats_path.is_file():
        meta["normalization"] = {
            "stats_file": stats_path.name,
            "has_heatmap_stats": True,
        }
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            anchors = stats.get("PI_freq", {}).get("anchor_mhz")
            if anchors and not meta.get("pi_frequencies_mhz"):
                meta["pi_frequencies_mhz"] = [float(x) for x in anchors]
        except (json.JSONDecodeError, OSError):
            pass

    if extra:
        meta.update(extra)
    return meta


def write_dataset_meta(
    data_root: Path | str,
    *,
    stage: str,
    source_script: str,
    extra: dict[str, Any] | None = None,
    filename: str = META_FILENAME,
) -> Path:
    """Write ``dataset_meta.json`` under ``data_root``; return path."""
    root = Path(data_root)
    meta = build_dataset_meta(root, stage=stage, source_script=source_script, extra=extra)
    out = root / filename
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dataset meta: {out}")
    print(
        f"  layouts={meta['counts'].get('unique_layouts')} "
        f"samples={meta['counts'].get('manifest_rows') or meta['counts'].get('heatmap_files')} "
        f"size={meta['size']['total_mb']} MB "
        f"MHz={meta.get('pi_frequencies_mhz', [])}"
    )
    return out


RAW_MULTIFREQ_TRAIN = "datasets/data_multifreq_train"


def read_pi_frequencies_mhz(
    data_root: Path | str,
    *,
    fallback_raw: bool = True,
) -> tuple[float, ...]:
    """Read anchor MHz from ``dataset_meta.json`` (preferred) or ``manifest.csv``.

    When *fallback_raw* is True and *data_root* has no meta/manifest MHz list,
    falls back to ``datasets/data_multifreq_train/dataset_meta.json`` so appends
    to the raw pool stay in sync without editing YAML.
    """
    root = Path(data_root).resolve()
    mhz = _pi_frequencies_from_root(root)
    if mhz:
        return mhz
    if fallback_raw:
        from repo_paths import repo_path

        raw = repo_path(RAW_MULTIFREQ_TRAIN)
        if raw.resolve() != root:
            mhz = _pi_frequencies_from_root(raw)
            if mhz:
                return mhz
    return ()


def _pi_frequencies_from_root(root: Path) -> tuple[float, ...]:
    meta_path = root / META_FILENAME
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        mhz = meta.get("pi_frequencies_mhz") or []
        if mhz:
            return tuple(sorted({float(x) for x in mhz}))
        source = meta.get("source_dataset")
        if source:
            src = Path(source)
            if src.is_dir() and src.resolve() != root.resolve():
                nested = _pi_frequencies_from_root(src)
                if nested:
                    return nested

    manifest = root / "manifest.csv"
    if manifest.is_file():
        info = _freq_summary_from_manifest(_read_manifest_rows(manifest))
        mhz = info.get("pi_frequencies_mhz") or []
        if mhz:
            return tuple(sorted({float(x) for x in mhz}))
    return ()
