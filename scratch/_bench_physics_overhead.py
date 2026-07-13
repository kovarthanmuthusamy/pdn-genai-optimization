"""Compare one training step with vs without PhysicsLoss."""
import sys
import time
from pathlib import Path

import torch
import yaml

from repo_paths import setup_path

setup_path()

from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer
from experiments.exp043.codes import train_vae_simple as tr
from experiments.exp043.codes.train_vae_simple import Config, build_vae_model, _forward_train_batch, vae_loss
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

device = "cuda" if torch.cuda.is_available() else "cpu"
c.device = device
model = build_vae_model(c).to(device).train()
phys = PhysicsLoss(c.background_value, c.physics_fg_clip_min).to(device).train()

B = c.batch_size
hm = torch.rand(B, 1, 64, 64, device=device) * float(c.heatmap_z_clip_max)
occ = (torch.rand(B, 52, device=device) > 0.5).float()
imp = torch.randn(B, 1, 231, device=device)
K = occ.sum(1).long().clamp(1, 52)
pi = torch.rand(B, device=device)
epoch = 40
ps = tr._penalty_scale(epoch, c)
pw = tr._physics_weights(epoch, c)
beta = 0.01

params_no = list(model.parameters())
params_yes = params_no + list(phys.parameters())
opt_no = torch.optim.AdamW(params_no, lr=1e-4)
opt_yes = torch.optim.AdamW(params_yes, lr=1e-4)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def bench(use_phys: bool, n: int = 30) -> float:
    opt = opt_yes if use_phys else opt_no
    for _ in range(5):
        rh, ro, ri, mu, lv, ex, z = _forward_train_batch(model, hm, occ, imp, K, pi, c, train=True, epoch=epoch)
        losses = vae_loss(
            rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
            epoch=epoch, K=K, physics=phys if use_phys else None, pw=pw if use_phys else None,
            pi_freq=pi, imp_log_std=1.0, ps=ps, apply_k=True,
        )
        losses["total_loss"].backward()
        opt.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        rh, ro, ri, mu, lv, ex, z = _forward_train_batch(model, hm, occ, imp, K, pi, c, train=True, epoch=epoch)
        losses = vae_loss(
            rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
            epoch=epoch, K=K, physics=phys if use_phys else None, pw=pw if use_phys else None,
            pi_freq=pi, imp_log_std=1.0, ps=ps, apply_k=True,
        )
        losses["total_loss"].backward()
        opt.zero_grad(set_to_none=True)
    sync()
    return (time.perf_counter() - t0) / n


t_no = bench(False)
t_yes = bench(True)
print(f"device={device}  batch={B}  epoch={epoch}  physics_weights ri={pw[0]:.2f} cs={pw[1]:.1f} ar={pw[2]:.2f}")
print(f"per step no physics:  {t_no*1000:.1f} ms")
print(f"per step with physics: {t_yes*1000:.1f} ms")
print(f"overhead: {(t_yes-t_no)*1000:.1f} ms  ({100*(t_yes/t_no-1):.1f}%)")
print(f"est epoch save @125 batches: {(t_yes-t_no)*125:.1f} s")
