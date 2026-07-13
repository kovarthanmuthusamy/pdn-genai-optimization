"""Break down heatmap loss components at init."""
import sys
from pathlib import Path

from repo_paths import setup_path

setup_path()

import torch
from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer, transform_disk_heatmap
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.gmax_heatmap_loss import heatmap_loss_gmax
from experiments.exp043.codes.train_vae_simple import Config
from experiments.exp043.codes.vae_poe_freq import MultiInputVAEPoeFreq
from src_vae.others.heatmap_gmax_norm import load_stats

raw = load_stats("datasets/data_multifreq_gmax")
c = Config()
c.data_dir = "datasets/data_multifreq_gmax"
c.heatmap_train_space = "log1p_gmax"
apply_gmax_config(c, raw)
patch_trainer(tr, c)

m = MultiInputVAEPoeFreq(latent_dim=42, heatmap_private_dim=12, cond_dim=8).eval()
B = 8
hm_lin = torch.rand(B, 1, 64, 64).clamp(0, 1.02)
hm = transform_disk_heatmap(hm_lin, c)
occ = (torch.rand(B, 52) > 0.5).float()
imp = torch.randn(B, 1, 231)
K = occ.sum(1).long().clamp(0, 52)
pi = torch.linspace(0.1, 0.9, B)
with torch.no_grad():
    rh, _, _, _, _, _ = m(hm, occ, imp, K, pi)

# component probe via toggling weights
base_cfg = dict(
    heatmap_pattern_weight=0, heatmap_spread_weight=0, heatmap_hotspot_weight=0,
    heatmap_peak_centroid_weight=0, heatmap_grad_weight=0, heatmap_lap_weight=0,
    heatmap_contrast_weight=0, heatmap_bg_weight=0, heatmap_dynrange_weight=0,
    heatmap_peak_weight=0,
)
for name, extra in [
    ("base_only", {}),
    ("+pattern", {"heatmap_pattern_weight": 5.0}),
    ("+spread", {"heatmap_spread_weight": 3.5}),
    ("+hotspot", {"heatmap_hotspot_weight": 4.0}),
    ("+centroid", {"heatmap_peak_centroid_weight": 2.5}),
    ("+grad_lap", {"heatmap_grad_weight": 3.5, "heatmap_lap_weight": 2.0}),
    ("full", {}),
]:
    cc = Config()
    for k, v in base_cfg.items():
        setattr(cc, k, v)
    if name == "full":
        import yaml
        cfg = yaml.safe_load(Path("experiments/exp043/config.yaml").read_text())
        for k, v in cfg.items():
            if hasattr(cc, k):
                setattr(cc, k, v)
    else:
        for k, v in extra.items():
            setattr(cc, k, v)
    apply_gmax_config(cc, raw)
    cc.heatmap_fg_threshold = c.heatmap_fg_threshold
    cc.heatmap_z_clip_min = c.heatmap_z_clip_min
    cc.heatmap_z_clip_max = c.heatmap_z_clip_max
    loss = heatmap_loss_gmax(rh, hm, cc, ps=1.0).mean().item()
    print(f"{name:12s} {loss:.2f}")
