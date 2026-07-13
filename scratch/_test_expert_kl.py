import sys
from pathlib import Path

from repo_paths import setup_path

setup_path()

import torch
from experiments.exp043.codes.train_vae_simple import _iter_expert_stats, _expert_mu_lv_pairs
from experiments.exp043.codes.vae_poe_freq import MultiInputVAEPoeFreq

m = MultiInputVAEPoeFreq(latent_dim=42, heatmap_private_dim=12, cond_dim=8)
B = 2
hm = torch.rand(B, 1, 64, 64)
occ = (torch.rand(B, 52) > 0.5).float()
imp = torch.randn(B, 1, 231)
K = occ.sum(1).long().clamp(0, 52)
pi = torch.tensor([100.0, 200.0])
_, mu, lv, stats = m.encode(hm, occ, imp, K, pi)
pairs = _expert_mu_lv_pairs(stats)
assert stats.get("heatmap_skips") is not None
assert len(pairs) >= 3
exp_kl = sum((-0.5 * (1.0 + lv - m.pow(2) - lv.exp())).mean() for m, lv in pairs) / len(pairs)
for name, em, elv in _iter_expert_stats(stats):
    _ = (elv / 2).exp()
print("OK", len(pairs), len(list(_iter_expert_stats(stats))))
