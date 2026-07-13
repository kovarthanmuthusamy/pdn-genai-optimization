import json
from pathlib import Path

from repo_paths import REPO_ROOT

root = REPO_ROOT / "datasets"
for name in ["data_multifreq_norm_z_score", "data_multifreq_gmax", "data_multifreq"]:
    p = root / name / "normalization_stats.json"
    if not p.exists():
        print(f"{name}: missing stats")
        continue
    s = json.loads(p.read_text())
    h = s.get("Heatmap", {})
    print(f"\n{name}:")
    print(f"  norm_mode: {h.get('norm_mode')}")
    print(f"  log_mean: {h.get('log_mean')}")
    print(f"  log_std: {h.get('log_std')}")
    print(f"  global_max_ohm: {h.get('global_max_ohm')}")
    print(f"  background_value: {s.get('background_value')}")

symlink = root / "data_multifreq_norm"
if symlink.is_symlink():
    print(f"\nsymlink data_multifreq_norm -> {symlink.resolve().name}")
