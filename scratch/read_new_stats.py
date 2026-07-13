import json
from pathlib import Path

from repo_paths import REPO_ROOT

s = json.loads((REPO_ROOT / "datasets/data_multifreq_norm_z_score/normalization_stats.json").read_text())
h = s["Heatmap"]
print("background_value:", s["background_value"])
print("clip_min:", h["clip_min"])
print("clip_max:", h["clip_max"])
print("log_mean:", h["log_mean"])
print("log_std:", h["log_std"])
