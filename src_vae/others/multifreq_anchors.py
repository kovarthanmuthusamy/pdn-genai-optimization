"""Shared PI frequency anchor MHz list and label helpers.

Anchor MHz are read from ``dataset_meta.json`` (``pi_frequencies_mhz``) on the
training/normalize dataset dir, with fallback to raw ``datasets/data_multifreq_train``.
``configs/multifreq_anchors.yaml`` is legacy fallback only.

Run: Imported by dataset build, normalization, and training scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import yaml

from libs.dataset_meta import read_pi_frequencies_mhz

# Last-resort if no JSON/manifest found anywhere
_DEFAULT_ANCHORS_MHZ: tuple[float, ...] = (
    10.0, 63.0, 80.0, 130.0, 150.0, 200.0, 250.0, 270.0, 330.0, 400.0, 500.0,
)


def _project_root() -> Path:
    from repo_paths import REPO_ROOT
    return REPO_ROOT


def _yaml_anchors(path: Path) -> tuple[float, ...]:
    if not path.is_file():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vals = raw.get("anchors_mhz") or raw.get("anchors") or []
    if not vals:
        return ()
    return tuple(sorted({float(x) for x in vals}))


def _resolve_dataset_dir(dataset_dir: str | Path | None) -> Path | None:
    if dataset_dir is not None:
        return Path(dataset_dir)
    for key in ("VAE_DATA_DIR", "MULTIFREQ_DATA_DIR", "MULTIFREQ_RAW_DIR"):
        val = os.environ.get(key)
        if val:
            return Path(val)
    return None


def load_anchors_mhz(
    dataset_dir: str | Path | None = None,
    *,
    path: Path | None = None,
) -> tuple[float, ...]:
    """Load sorted anchor MHz from dataset JSON (preferred) or legacy YAML.

    Resolution order:
      1. ``dataset_dir`` / ``dataset_meta.json`` (and manifest.csv on that dir)
      2. Env ``VAE_DATA_DIR`` / ``MULTIFREQ_*``
      3. Raw pool ``datasets/data_multifreq_train/dataset_meta.json``
      4. ``configs/multifreq_anchors.yaml`` (or explicit *path* to that YAML)
      5. Built-in defaults
    """
    root = _resolve_dataset_dir(dataset_dir)
    if root is not None:
        mhz = read_pi_frequencies_mhz(root, fallback_raw=True)
        if mhz:
            return mhz

    if root is None:
        mhz = read_pi_frequencies_mhz(
            _project_root() / "datasets" / "data_multifreq_train",
            fallback_raw=False,
        )
        if mhz:
            return mhz

    yaml_path = path if path is not None else _project_root() / "configs" / "multifreq_anchors.yaml"
    mhz = _yaml_anchors(yaml_path)
    if mhz:
        return mhz
    return _DEFAULT_ANCHORS_MHZ


def mhz_to_label(mhz: float) -> str:
    m = int(round(float(mhz)))
    return f"{m}MHz"


def mhz_to_heatmap_subdir(mhz: float) -> str:
    return f"heatmap_{mhz_to_label(mhz)}"


def anchors_to_freq_labels(
    anchors: Sequence[float] | None = None,
    *,
    dataset_dir: str | Path | None = None,
) -> list[str]:
    a = anchors or load_anchors_mhz(dataset_dir)
    return [mhz_to_label(m) for m in a]


def anchors_to_freq_hz(
    anchors: Sequence[float] | None = None,
    *,
    dataset_dir: str | Path | None = None,
) -> dict[str, float]:
    a = anchors or load_anchors_mhz(dataset_dir)
    return {mhz_to_label(m): float(m) * 1e6 for m in a}


def nearest_anchor_mhz(
    mhz: float,
    anchors: Sequence[float] | None = None,
    *,
    dataset_dir: str | Path | None = None,
) -> float:
    a = anchors or load_anchors_mhz(dataset_dir)
    return min(a, key=lambda x: abs(float(mhz) - float(x)))


def anchor_bin_index(
    mhz: float,
    anchors: Sequence[float] | None = None,
    tol_mhz: float = 0.5,
    *,
    dataset_dir: str | Path | None = None,
) -> int:
    """Index into anchors for exact (within tol) or nearest anchor."""
    a = list(anchors or load_anchors_mhz(dataset_dir))
    mhz = float(mhz)
    for i, anchor in enumerate(a):
        if abs(mhz - float(anchor)) <= tol_mhz:
            return i
    return int(min(range(len(a)), key=lambda i: abs(mhz - float(a[i]))))
