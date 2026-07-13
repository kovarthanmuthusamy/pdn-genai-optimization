import json, random, csv
from pathlib import Path
import numpy as np

DATA = Path("datasets/data_multifreq_gmax")
stats = json.loads((DATA / "normalization_stats.json").read_text())["Heatmap"]
gmax = stats["global_max_ohm"]
thr = stats["fg_norm_threshold"]
rows = list(csv.DictReader((DATA / "manifest.csv").open()))
pick = random.Random(0).sample(rows, 400)
fg_lin, fg_log, fg_sqrt = [], [], []
for row in pick:
    x = np.load(DATA / "heatmap" / row["sample_name"], mmap_mode="r").reshape(-1)
    fg = x > thr
    if not fg.any():
        continue
    v = x[fg]
    fg_lin.append(v)
    fg_log.append(np.log1p(v * gmax) / np.log1p(gmax))
    fg_sqrt.append(np.sqrt(v))
for name, chunks in [("linear_norm", fg_lin), ("log1p_phys/gmax", fg_log), ("sqrt_norm", fg_sqrt)]:
    a = np.concatenate(chunks)
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    sk = ((a - a.mean()) ** 3).mean() / a.std() ** 3
    print(f"{name:18s} skew={sk:.2f}  p10={p10:.5f}  p50={p50:.5f}  p90={p90:.5f}  p90/p10={p90 / max(p10, 1e-9):.1f}x")
