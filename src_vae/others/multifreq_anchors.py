"""Shared PI frequency anchor MHz list and label helpers.

Run: Imported by dataset build, normalization, and training scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

# Fallback if configs/multifreq_anchors.yaml is missing
_DEFAULT_ANCHORS_MHZ: tuple[float, ...] = (
    10.0, 63.0, 80.0, 130.0, 150.0, 200.0, 250.0, 270.0, 330.0, 400.0, 500.0,
)


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src_vae").is_dir() and (p / "configs").is_dir():
            return p
    return here.parents[2]


def load_anchors_mhz(path: Path | None = None) -> tuple[float, ...]:
    """Load sorted unique anchor MHz from YAML (or defaults)."""
    if path is None:
        path = _project_root() / "configs" / "multifreq_anchors.yaml"
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        vals = raw.get("anchors_mhz") or raw.get("anchors") or []
        if vals:
            return tuple(sorted({float(x) for x in vals}))
    return _DEFAULT_ANCHORS_MHZ


def mhz_to_label(mhz: float) -> str:
    m = int(round(float(mhz)))
    return f"{m}MHz"


def mhz_to_heatmap_subdir(mhz: float) -> str:
    return f"heatmap_{mhz_to_label(mhz)}"


def anchors_to_freq_labels(anchors: Sequence[float] | None = None) -> list[str]:
    return [mhz_to_label(m) for m in (anchors or load_anchors_mhz())]


def anchors_to_freq_hz(anchors: Sequence[float] | None = None) -> dict[str, float]:
    return {mhz_to_label(m): float(m) * 1e6 for m in (anchors or load_anchors_mhz())}


def nearest_anchor_mhz(mhz: float, anchors: Sequence[float] | None = None) -> float:
    a = anchors or load_anchors_mhz()
    return min(a, key=lambda x: abs(float(mhz) - float(x)))


def anchor_bin_index(mhz: float, anchors: Sequence[float] | None = None, tol_mhz: float = 0.5) -> int:
    """Index into anchors for exact (within tol) or nearest anchor."""
    a = list(anchors or load_anchors_mhz())
    mhz = float(mhz)
    for i, anchor in enumerate(a):
        if abs(mhz - float(anchor)) <= tol_mhz:
            return i
    return int(min(range(len(a)), key=lambda i: abs(mhz - float(a[i]))))
