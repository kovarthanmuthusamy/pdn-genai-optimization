import json
from pathlib import Path
p = Path("data_multi_norm_unbounded/normalization_stats.json")
if p.exists():
    d = json.loads(p.read_text())
    print("top keys:", list(d.keys())[:15])
    hm = d.get("heatmap", d)
    if isinstance(hm, dict):
        print("norm_mode:", hm.get("norm_mode"))
        print("unbounded:", hm.get("unbounded"))
        bins = hm.get("per_mhz_bins", hm.get("bins", {}))
        if bins:
            for k in sorted(bins, key=float):
                b = bins[k]
                print(f"  {k} MHz: median={b.get('median',0):.3f} iqr={b.get('iqr',0):.3f}")
        else:
            print("bins keys sample:", str(hm)[:500])
else:
    print("no stats file")

m = Path("data_multi_norm_unbounded/manifest_train.csv")
if m.exists():
    print("train_rows:", sum(1 for _ in m.open()) - 1)
