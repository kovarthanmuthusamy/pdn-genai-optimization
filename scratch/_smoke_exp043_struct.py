"""Smoke test: 8×8 heatmap bottleneck + pattern/spread losses."""
from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path

setup_path()

import torch
from experiments.exp043.codes.vae_poe_freq import MultiInputVAEPoeFreq
from experiments.exp043.codes.gmax_heatmap_loss import heatmap_loss_gmax
from experiments.exp043.codes.train_vae_simple import Config

c = Config()
c.heatmap_pattern_weight = 5.0
c.heatmap_spread_weight = 3.5
B = 4
m = MultiInputVAEPoeFreq(
    latent_dim=42,
    heatmap_private_dim=12,
    cond_dim=8,
    freq_fourier_features=12,
    use_heatmap_film=True,
)
hm = torch.rand(B, 1, 64, 64)
occ = (torch.rand(B, 52) > 0.5).float()
imp = torch.randn(B, 1, 231)
K = occ.sum(1).long().clamp(0, 52)
pi = torch.tensor([100.0, 200.0, 300.0, 400.0])
rh, ro, ri, mu, logvar, _ = m(hm, occ, imp, K, pi)
assert rh.shape == (B, 1, 64, 64), rh.shape
assert float(rh.max()) <= 1.02 + 1e-4, float(rh.max())
loss = heatmap_loss_gmax(rh, hm, c, ps=1.0)
assert loss.shape == (B,), loss.shape
print("OK", rh.shape, float(loss.mean()))
print("params", sum(p.numel() for p in m.parameters()))
