import numpy as np
import os, glob

from repo_paths import REPO_ROOT

root = str(REPO_ROOT / "data_multi_norm" / "layouts")
dirs = sorted(os.listdir(root))[:1]
for d in dirs:
    dp = os.path.join(root, d)
    print("LAYOUT DIR:", d)
    for f in sorted(glob.glob(os.path.join(dp, "*"))):
        name = os.path.basename(f)
        if f.endswith(".npy"):
            a = np.load(f)
            print(f"  {name}: shape={a.shape} dtype={a.dtype} min={float(a.min()):.3f} max={float(a.max()):.3f} uniq={len(np.unique(a))} size={a.size}")
            if a.size <= 60:
                print("    values:", np.round(a.flatten(), 2))
        else:
            print(f"  {name}: ({os.path.getsize(f)} bytes)")
