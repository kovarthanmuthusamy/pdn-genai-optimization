from pathlib import Path

from repo_paths import REPO_ROOT

p = REPO_ROOT / "datasets/data_multifreq_norm_z_score"
print("Top-level:", sorted(x.name for x in p.iterdir()))
layouts = p / "layouts"
imp = list(layouts.glob("*/imp.npy"))
occ = list(layouts.glob("*/occ.npy"))
print(f"layouts dirs: {sum(1 for x in layouts.iterdir() if x.is_dir())}")
print(f"imp.npy files: {len(imp)}")
print(f"occ.npy files: {len(occ)}")
if imp:
    import numpy as np
    s = imp[0]
    arr = np.load(s)
    print(f"Sample {s.parent.name}/imp.npy shape={arr.shape} dtype={arr.dtype} range=[{arr.min():.3f},{arr.max():.3f}]")
print(f"Imp/ folder exists: {(p / 'Imp').exists()}")
print(f"Occ_map/ exists: {(p / 'Occ_map').exists()}")

# compare raw train
raw = REPO_ROOT / "datasets/data_multifreq_train"
print("\nRaw train top-level:", sorted(x.name for x in raw.iterdir()))
print(f"Raw Imp/ exists: {(raw / 'Imp').exists()}")
print(f"Raw layouts imp count: {len(list((raw / 'layouts').glob('*/imp.npy')))}")
