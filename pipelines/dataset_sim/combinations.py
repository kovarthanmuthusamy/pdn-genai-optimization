"""Load decap combination rows from CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_DECAPS = 52


def load_combinations_csv(path: Path) -> np.ndarray:
    """Return (N, 52) int8 occupancy rows."""
    rows: list[list[int]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != N_DECAPS:
                raise ValueError(f"{path}: expected {N_DECAPS} columns, got {len(parts)}")
            rows.append([int(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    arr = np.array(rows, dtype=np.int8)
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(f"{path}: values must be 0 or 1")
    return arr


def layout_dir_name(index: int) -> str:
    return f"layout_{index:05d}"
