"""Frequencies: the 231-point spectrum grid, the model's frequency input, and map frequencies.

What: the spectrum is sampled at 231 points from 1 to 600 MHz (inputs/Frequency_data_hz.npy). The
    model takes a frequency as log10(f) scaled to [0, 1] over 1-600 MHz. Map frequencies of a dataset
    are read from its dataset_meta.json (pi_frequencies_mhz) or, if absent, from its manifest.csv.
Usage (import only):
    freqs_mhz()                          (231,) spectrum grid in MHz
    pi_freq_mhz_to_norm(mhz), pi_freq_hz_to_norm(hz)   model frequency input in [0, 1]
    read_pi_frequencies_mhz(root), load_anchors_mhz(dataset_dir)   sorted map frequencies (MHz)
    anchor_bin_index, nearest_anchor_mhz
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

INPUTS = Path(__file__).resolve().parent / "inputs"

PI_FREQ_MIN_HZ = 1e6
PI_FREQ_MAX_HZ = 600e6
_LOG10_MIN = math.log10(PI_FREQ_MIN_HZ)
_LOG10_RANGE = math.log10(PI_FREQ_MAX_HZ) - _LOG10_MIN


def freqs_mhz() -> np.ndarray:
    """(231,) spectrum frequency grid in MHz."""
    return np.load(INPUTS / "Frequency_data_hz.npy").reshape(-1) / 1e6


def pi_freq_hz_to_norm(freq_hz: float) -> float:
    hz = max(float(freq_hz), 1.0)
    return float(np.clip((math.log10(hz) - _LOG10_MIN) / _LOG10_RANGE, 0.0, 1.0))


def pi_freq_mhz_to_norm(freq_mhz: float) -> float:
    """Convert a frequency in MHz to the model's frequency input in [0, 1]."""
    return pi_freq_hz_to_norm(float(freq_mhz) * 1e6)


def read_pi_frequencies_mhz(root: Path | str) -> tuple[float, ...]:
    """Sorted map frequencies (MHz) of a dataset folder: dataset_meta.json, else manifest.csv."""
    root = Path(root).resolve()
    meta_path = root / "dataset_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        mhz = meta.get("pi_frequencies_mhz") or []
        if mhz:
            return tuple(sorted({float(x) for x in mhz}))
    manifest = root / "manifest.csv"
    if manifest.is_file():
        vals = set()
        with manifest.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    vals.add(float(r["freq_mhz"]))
                except (KeyError, TypeError, ValueError):
                    pass
        if vals:
            return tuple(sorted(vals))
    return ()


def load_anchors_mhz(dataset_dir: str | Path) -> tuple[float, ...]:
    """Map frequencies (MHz) of ``dataset_dir``; stops if the folder has none."""
    mhz = read_pi_frequencies_mhz(dataset_dir)
    if not mhz:
        raise SystemExit(f"no map frequencies in {dataset_dir} (dataset_meta.json or manifest.csv)")
    return mhz


def nearest_anchor_mhz(mhz: float, anchors: Sequence[float]) -> float:
    return min(anchors, key=lambda x: abs(float(mhz) - float(x)))


def anchor_bin_index(mhz: float, anchors: Sequence[float], tol_mhz: float = 0.5) -> int:
    """Index of the anchor within ``tol_mhz`` of ``mhz``, else of the nearest anchor."""
    a = list(anchors)
    mhz = float(mhz)
    for i, anchor in enumerate(a):
        if abs(mhz - float(anchor)) <= tol_mhz:
            return i
    return int(min(range(len(a)), key=lambda i: abs(mhz - float(a[i]))))
