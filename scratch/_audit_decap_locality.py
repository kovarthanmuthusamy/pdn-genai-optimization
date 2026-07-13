"""Check if GT occupancy correlates with local low-Z in exp043 train space."""
import sys
from pathlib import Path
import torch
import yaml

from repo_paths import setup_path

setup_path()

from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer, _prepare_batch_gmax
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.train_vae_simple import Config
from experiments.exp043.codes.dataloader_multifreq import create_multifreq_data_loaders
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
train_ld, _ = create_multifreq_data_loaders(
    data_dir=c.data_dir, batch_size=64, num_workers=0, train_split=c.train_split,
    cache_in_ram=False, train_samples_per_epoch=512, cross_freq_pairs=False,
    split_by_design=True, balance_k=False, balance_freq=False,
)

hm_sum = occ_sum = n = 0.0
hm_at_decap = hm_far = cnt_d = cnt_f = 0.0
scatter = phys.occ_scatter

for batch in train_ld:
    hm, _, occ, _, _, _ = _prepare_batch_gmax(batch, c, tr._prepare_batch)
    fg = hm > c.heatmap_fg_threshold
    # per-sample: pixels nearest each active decap (7x8 cell center on 64 grid)
    for b in range(hm.shape[0]):
        for i in range(52):
            if occ[b, i] < 0.5:
                continue
            pos = (scatter[i] > 0).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            r, col = divmod(int(pos.item()), 8)
            cy, cx = int(r * 64 / 7 + 32 / 7), int(col * 64 / 8 + 32 / 8)
            cy, cx = min(63, max(0, cy)), min(63, max(0, cx))
            if fg[b, 0, cy, cx]:
                hm_at_decap += hm[b, 0, cy, cx].item()
                cnt_d += 1
        far = hm[b, 0][fg[b, 0]]
        if far.numel():
            hm_far += far.mean().item()
            cnt_f += 1

print(f"GT hm at decap cell centers: mean={hm_at_decap/cnt_d:.4f}  (n={int(cnt_d)})")
print(f"GT hm FG spatial mean:        mean={hm_far/cnt_f:.4f}  (n={int(cnt_f)} samples)")
print(f"delta (far - decap):          {hm_far/cnt_f - hm_at_decap/cnt_d:.4f}  (positive => decap sites lower Z)")
