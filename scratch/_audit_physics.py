"""Audit PhysicsCritic + radius_influence for exp043 log1p/gmax heatmaps."""
import sys
from pathlib import Path

import torch
import yaml

from repo_paths import setup_path

setup_path()

from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer, _prepare_batch_gmax
from experiments.exp043.codes.train_vae_simple import Config
from experiments.exp043.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.physics_loss import PhysicsLoss
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

phys = PhysicsLoss(c.background_value, c.physics_fg_clip_min)
print("=== thresholds ===")
print(f"bg_threshold      {phys.bg_threshold:.6f}")
print(f"low_hm_thresh     {phys.low_hm_thresh:.6f}  (fg_clip_min+0.4)")
print(f"heatmap_fg_thr    {c.heatmap_fg_threshold:.6f}")

# real batch
train_ld, _ = create_multifreq_data_loaders(
    data_dir=c.data_dir, batch_size=8, num_workers=0, train_split=c.train_split,
    cache_in_ram=False, train_samples_per_epoch=8, cross_freq_pairs=False,
    split_by_design=c.split_by_design, balance_k=False, balance_freq=False,
)
batch = next(iter(train_ld))
hm, _, occ, imp, K, pi = _prepare_batch_gmax(batch, c, tr._prepare_batch)
hm, occ, pi = hm.cpu(), occ.cpu(), pi.cpu()
phys = phys.cpu()

fg = (hm > phys.bg_threshold).float()
fg_vals = hm[fg > 0]
print(f"\n=== target hm (train space) FG stats n={fg_vals.numel()} ===")
for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
    print(f"  p{int(q*100):02d} {torch.quantile(fg_vals, q).item():.4f}")

tgt_eff = phys._heatmap_to_critic_target(hm)
print(f"\n=== critic target (high=low-Z / decap effective) ===")
te = tgt_eff[fg > 0]
print(f"  mean {te.mean().item():.4f}  p50 {torch.quantile(te, 0.5).item():.4f}  p90 {torch.quantile(te, 0.9).item():.4f}")

with torch.no_grad():
    cs = phys.critic_supervision_loss(occ, hm, pi)
    # random recon
    recon = torch.rand_like(hm) * float(c.heatmap_z_clip_max)
    ri_rand = phys.radius_influence_loss(recon, occ.float(), pi)
    ri_tgt = phys.radius_influence_loss(hm, occ.float(), pi)

print(f"\n=== losses (untrained critic) ===")
print(f"  critic_supervision {cs.item():.4f}")
print(f"  RI random recon    {ri_rand.item():.4f}")
print(f"  RI perfect recon   {ri_tgt.item():.4f}")

# decap locality: at decap grid cells, is target eff higher?
scatter = phys.occ_scatter
occ_sp = (occ.float() @ scatter).view(-1, 1, 7, 8)
occ_up = torch.nn.functional.interpolate(occ_sp, size=(64, 64), mode="bilinear", align_corners=False)
# sample where occ_up > 0.5 vs < 0.1
high_occ = occ_up[0, 0] > 0.5
low_occ = occ_up[0, 0] < 0.1
fg0 = fg[0, 0] > 0
if high_occ.any() and low_occ.any():
    print(f"\n=== sample0: hm at high-occ vs low-occ (target) ===")
    print(f"  hm mean high-occ grid {hm[0,0][high_occ & fg0].mean().item():.4f}")
    print(f"  hm mean low-occ  grid {hm[0,0][low_occ & fg0].mean().item():.4f}")
    print(f"  tgt_eff mean high-occ {tgt_eff[0,0][high_occ & fg0].mean().item():.4f}")
    print(f"  tgt_eff mean low-occ  {tgt_eff[0,0][low_occ & fg0].mean().item():.4f}")

# weighted train contribution at ep 40
wri, wcs, war = tr._physics_weights(39, c)
print(f"\n=== weighted contribution ep 40 (epoch idx 39) ===")
print(f"  weights ri={wri:.3f} cs={wcs:.1f} ar={war:.3f}")
print(f"  ~train add  ri*w*0.04 + cs*2*0.03 + ar*w*0.001 ≈ {wri*0.04 + wcs*0.03 + war*0.001:.3f}")
print(f"  vs hm*4     ~18*4 = 72")
