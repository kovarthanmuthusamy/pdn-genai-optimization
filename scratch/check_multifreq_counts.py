from pathlib import Path

from repo_paths import REPO_ROOT

root = REPO_ROOT / "datasets"
for name in ["data_multifreq", "data_multifreq_train", "data_multifreq_norm_z_score", "data_multifreq_gmax", "data_multifreq_norm"]:
    p = root / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    hm = len(list((p / "heatmap").glob("*.npy"))) if (p / "heatmap").is_dir() else 0
    pf = len(list((p / "PI_freq").glob("*.npy"))) if (p / "PI_freq").is_dir() else 0
    print(f"{name}: heatmap={hm} PI_freq={pf}")
