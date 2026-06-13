from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def build_peb_from_selection(
    selected: list[dict[str, Any]],
    peb_path: Path,
    *,
    powerbus: str,
    heatmap_only: bool,
    components: str,
    groot: Path,
) -> Path:
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))
    from scrap.generation.generate_peb import generate_peb  # noqa: E402

    occ = np.array([s["occupancy"] for s in selected], dtype=np.int8)
    freqs = [f"{int(round(s['mhz']))}e6" for s in selected]
    peb_path.parent.mkdir(parents=True, exist_ok=True)
    generate_peb(
        occupancy=occ,
        output_path=str(peb_path),
        powerbus=powerbus,
        freq=freqs[0],
        per_sample_freqs=freqs,
        components=components,
        include_distribution=True,
        include_spectrum=not heatmap_only,
    )
    return peb_path
