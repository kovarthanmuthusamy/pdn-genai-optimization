"""Rough per-batch timing breakdown for exp043 training step."""
import sys
import time
from pathlib import Path

import torch
import yaml

from repo_paths import setup_path

setup_path()

from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer, _prepare_batch_gmax
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.train_vae_simple import Config, build_vae_model, _forward_train_batch, vae_loss
from experiments.exp043.codes.gmax_heatmap_loss import cross_freq_heatmap_loss_gmax
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

device = "cuda" if torch.cuda.is_available() else "cpu"
c.device = device
model = build_vae_model(c).to(device).train()
B = c.batch_size
hm = torch.rand(B, 1, 64, 64, device=device) * 0.5
occ = (torch.rand(B, 52, device=device) > 0.5).float()
imp = torch.randn(B, 1, 231, device=device)
K = occ.sum(1).long().clamp(1, 52)
pi = torch.rand(B, device=device)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

# warmup
for _ in range(3):
    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(model, hm, occ, imp, K, pi, c, train=True, epoch=15)
    losses = vae_loss(rh, ro, ri, hm, occ, imp, mu, lv, 0.0, c, ex, epoch=15, K=K, physics=None, pw=None, pi_freq=pi, imp_log_std=1.0, ps=0.5, apply_k=True)
    losses["total_loss"].backward()
    model.zero_grad(set_to_none=True)
sync()

N = 20
t0 = time.perf_counter()
for _ in range(N):
    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(model, hm, occ, imp, K, pi, c, train=True, epoch=15)
    losses = vae_loss(rh, ro, ri, hm, occ, imp, mu, lv, 0.0, c, ex, epoch=15, K=K, physics=None, pw=None, pi_freq=pi, imp_log_std=1.0, ps=0.5, apply_k=True)
    losses["total_loss"].backward()
    model.zero_grad(set_to_none=True)
sync()
base = (time.perf_counter() - t0) / N

# + cross-freq full
t0 = time.perf_counter()
for _ in range(N):
    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(model, hm, occ, imp, K, pi, c, train=True, epoch=15)
    losses = vae_loss(rh, ro, ri, hm, occ, imp, mu, lv, 0.0, c, ex, epoch=15, K=K, physics=None, pw=None, pi_freq=pi, imp_log_std=1.0, ps=0.5, apply_k=True)
    z_cf = model.encode_layout_latent(occ, imp, K, pi)
    cf = cross_freq_heatmap_loss_gmax(model, z_cf, K, pi, hm, c, 0.5, occupancy=occ)
    (losses["total_loss"] + cf).backward()
    model.zero_grad(set_to_none=True)
sync()
with_cf = (time.perf_counter() - t0) / N

# v3 off
from experiments.exp043.codes.vae_poe_freq import MultiInputVAEPoeFreq
m2 = MultiInputVAEPoeFreq(
    latent_dim=42, heatmap_private_dim=12, cond_dim=8,
    use_heatmap_film=True, use_heatmap_unet_skips=False, use_occ_spatial_decoder=False,
).to(device).train()
t0 = time.perf_counter()
for _ in range(N):
    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(m2, hm, occ, imp, K, pi, c, train=True, epoch=15)
    losses = vae_loss(rh, ro, ri, hm, occ, imp, mu, lv, 0.0, c, ex, epoch=15, K=K, physics=None, pw=None, pi_freq=pi, imp_log_std=1.0, ps=0.5, apply_k=True)
    losses["total_loss"].backward()
    m2.zero_grad(set_to_none=True)
sync()
no_v3 = (time.perf_counter() - t0) / N

print(f"params v3={sum(p.numel() for p in model.parameters()):,}  no_v3={sum(p.numel() for p in m2.parameters()):,}")
print(f"batch={B}  device={device}")
print(f"main step only:     {base*1000:.0f} ms")
print(f"+ cross_freq full:  {with_cf*1000:.0f} ms  (+{(with_cf-base)*1000:.0f} ms)")
print(f"no v3 main only:    {no_v3*1000:.0f} ms  ({base/no_v3:.2f}x vs v3)")
print(f"est epoch sec (125 batches): main={base*125:.1f}  +cf={with_cf*125:.1f}")
