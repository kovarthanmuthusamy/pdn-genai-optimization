#!/usr/bin/env python3
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mhz_counts(root: Path) -> Counter:
    m = root / "manifest.csv"
    c: Counter = Counter()
    labels: Counter = Counter()
    if not m.is_file():
        return c
    for r in csv.DictReader(m.open(encoding="utf-8")):
        if r.get("freq_mhz"):
            c[float(r["freq_mhz"])] += 1
        if r.get("freq_label"):
            labels[r["freq_label"]] += 1
    return c, labels


for name in ("datasets/data_multifreq", "datasets/data_multifreq_norm"):
    p = Path(name)
    c, labels = mhz_counts(p)
    print(f"\n=== {name} ===")
    for mhz in sorted(c):
        print(f"  {mhz:g} MHz: {c[mhz]:,}")
    if not c:
        print("  (no freq_mhz in manifest)")
    print("  unique MHz:", sorted(c.keys()))
