#!/usr/bin/env python3
"""Verify log1p train-space roundtrip + skew improvement."""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in _ROOT.parents if (p / "src_vae").is_dir()),
    _ROOT.parents[1],
)
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp043.codes.train_vae_simple import Config
from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, transform_disk_heatmap
from src_vae.others.heatmap_gmax_norm import (
    disk_to_train_space,
    load_stats,
    log1p_train_to_linear_norm,
    train_to_disk_space,
)

DATA = REPO_ROOT / "datasets/data_multifreq_gmax"
raw = load_stats(DATA)
gmax = raw["Heatmap"]["global_max_ohm"]
thr = raw["Heatmap"]["fg_norm_threshold"]

c = Config()
c.data_dir = str(DATA)
c.heatmap_train_space = "log1p_gmax"
c.heatmap_fg_threshold = None
assert apply_gmax_config(c, raw)

# roundtrip on one map
stem = "sample_100036"
lin = torch.from_numpy(np.load(DATA / "heatmap" / f"{stem}.npy").astype(np.float32)).unsqueeze(0).unsqueeze(0)
tr = disk_to_train_space(lin, c)
back = train_to_disk_space(tr, c)
err = (back - lin).abs().max().item()
print(f"roundtrip max err: {err:.2e}")

fg = lin.flatten() > thr
lin_fg = lin.flatten()[fg].numpy()
tr_fg = tr.flatten()[fg].numpy()
sk_lin = ((lin_fg - lin_fg.mean()) ** 3).mean() / lin_fg.std() ** 3
sk_tr = ((tr_fg - tr_fg.mean()) ** 3).mean() / tr_fg.std() ** 3
print(f"FG skew linear={sk_lin:.2f}  log1p_train={sk_tr:.2f}")
print(f"FG p10/p90 linear: {np.percentile(lin_fg,10):.5f} / {np.percentile(lin_fg,90):.5f}")
print(f"FG p10/p90 log1p:  {np.percentile(tr_fg,10):.5f} / {np.percentile(tr_fg,90):.5f}")

batch = transform_disk_heatmap(lin, c)
assert batch.shape == lin.shape
print("OK log1p train space")
