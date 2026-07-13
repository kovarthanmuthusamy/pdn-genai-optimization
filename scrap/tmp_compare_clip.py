import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / "data_multi_norm_robust/normalization_stats.json"
raw = json.loads(p.read_text())
hm = raw["Heatmap"]
print("mode:", hm.get("norm_mode"))
print()
print(f"{'MHz':>8}  {'clip_max z':>10}  {'ceiling@clip':>14}  {'z_max':>8}  {'data_peak':>12}")
print("-" * 58)
for k in sorted(hm["by_mhz"].keys(), key=float):
    b = hm["by_mhz"][k]
    z = b["clip_max"]
    med, iqr = b["median"], b["iqr"]
    phys = math.expm1(z * iqr + med)
    zmax = b.get("z_max", z)
    zmax_phys = math.expm1(zmax * iqr + med)
    print(f"{k:>8}  {z:10.3f}  {phys:14.1f}  {zmax:8.2f}  {zmax_phys:12.1f}")

old = root / "data_multi_norm/normalization_stats.json"
if old.is_file():
    o = json.loads(old.read_text())["Heatmap"]
    lm, ls = o["log_mean"], o["log_std"]
    z = o["clip_max"]
    zmax = o.get("z_max", z)
    print()
    print(
        f"OLD global  clip_max z={z:.3f} -> {math.expm1(z*ls+lm):.1f} ohm  "
        f"z_max={zmax:.2f} -> {math.expm1(zmax*ls+lm):.1f} ohm"
    )
