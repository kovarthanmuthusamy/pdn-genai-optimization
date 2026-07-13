#!/usr/bin/env python3
"""One-off smoke test for libs.dataset_meta (not part of main pipelines)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.dataset_meta import build_dataset_meta

for name in ("datasets/data_multifreq", "datasets/data_multifreq_norm"):
    p = Path(name)
    if not p.exists():
        print("skip", name)
        continue
    m = build_dataset_meta(p, stage="smoke", source_script="tools/smoke_dataset_meta.py")
    print(name, m["counts"], m["size"]["total_mb"], "MB", m.get("pi_frequencies_mhz", [])[:5], "...")
