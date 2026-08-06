"""Load multi-type decap combination rows from CSV (type codes 0/1/2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_DECAPS = 52
# Allowed type codes (0 = empty). Extend when the catalog grows.
VALID_CODES = {0, 1, 2}


def load_multitype_csv(path: Path) -> np.ndarray:
    """Return (N, 52) int8 rows of type codes in ``VALID_CODES``."""
    rows: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as f:
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
    bad = set(np.unique(arr).tolist()) - VALID_CODES
    if bad:
        raise ValueError(f"{path}: values must be in {sorted(VALID_CODES)}; found extra {sorted(bad)}")
    return arr


def layout_dir_name(index: int) -> str:
    return f"layout_{index:05d}"
