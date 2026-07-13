"""Per-term heatmap loss on one real train batch (untrained model)."""
import sys
from pathlib import Path
import yaml

from repo_paths import setup_path

setup_path()

import torch
from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer, _prepare_batch_gmax
from experiments.exp043.codes.train_vae_simple import _prepare_batch as orig_prepare
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.gmax_heatmap_loss import heatmap_loss_gmax
from experiments.exp043.codes.train_vae_simple import Config, build_vae_model
from src_vae.others.heatmap_gmax_norm import load_stats

raw = load_stats("datasets/data_multifreq_gmax")
cfg = yaml.safe_load(Path("experiments/exp043/config.yaml").read_text())
c = Config()
for k, v in cfg.items():
    if hasattr(c, k):
        setattr(c, k, v)
c.data_dir = "datasets/data_multifreq_gmax"
apply_gmax_config(c, raw)
patch_trainer(tr, c)

from experiments.exp043.codes.train_vae_simple import _build_dataloaders  # noqa: E402

_, train_dl, _ = _build_dataloaders(c)
batch = next(iter(train_dl))
hm, _, occ, imp, K, pi = _prepare_batch_gmax(batch, c, orig_prepare)

model = build_vae_model(c).eval()
with torch.no_grad():
    rh, _, _, _, _, _ = model(hm, occ, imp, K, pi)

ps = tr._penalty_scale(1, c)
full = heatmap_loss_gmax(rh, hm, c, ps).mean().item()

# toggle terms
terms = {}
base_cfg = dict(
    heatmap_pattern_weight=0, heatmap_spread_weight=0, heatmap_hotspot_weight=0,
    heatmap_peak_centroid_weight=0, heatmap_grad_weight=0, heatmap_lap_weight=0,
    heatmap_contrast_weight=0, heatmap_bg_weight=0, heatmap_dynrange_weight=0,
    heatmap_peak_weight=0,
)
cc = Config()
for k, v in cfg.items():
    if hasattr(cc, k):
        setattr(cc, k, v)
cc.data_dir = c.data_dir
apply_gmax_config(cc, raw)
for k, v in base_cfg.items():
    setattr(cc, k, v)
terms["base_only"] = heatmap_loss_gmax(rh, hm, cc, ps).mean().item()

for name, keys in [
    ("+pattern", ["heatmap_pattern_weight"]),
    ("+spread", ["heatmap_spread_weight"]),
    ("+hotspot", ["heatmap_hotspot_weight"]),
    ("+centroid", ["heatmap_peak_centroid_weight"]),
    ("+peak", ["heatmap_peak_weight"]),
    ("+grad", ["heatmap_grad_weight"]),
    ("+lap_dyn", ["heatmap_lap_weight", "heatmap_contrast_weight", "heatmap_bg_weight", "heatmap_dynrange_weight"]),
]:
    cc2 = Config()
    for k, v in cfg.items():
        if hasattr(cc2, k):
            setattr(cc2, k, v)
    cc2.data_dir = c.data_dir
    apply_gmax_config(cc2, raw)
    for k, v in base_cfg.items():
        setattr(cc2, k, v)
    for k in keys:
        setattr(cc2, k, getattr(c, k))
    terms[name] = heatmap_loss_gmax(rh, hm, cc2, ps).mean().item() - terms.get("base_only", 0)

print(f"real batch full hm loss (ep1 ps={ps:.3f}): {full:.2f}")
for k, v in terms.items():
    print(f"  {k:12s} {v:.2f}")

from experiments.exp043.codes.gmax_training_patch import heatmap_phys_amplitude_loss_gmax
phys = heatmap_phys_amplitude_loss_gmax(rh, hm, c.global_max_ohm, c).item()
print(f"  phys_p99      {phys:.2f}  (×{c.heatmap_phys_p99_weight} = {phys*c.heatmap_phys_p99_weight:.1f})")
