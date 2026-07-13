#!/usr/bin/env python3
"""Joint latent optimization: heatmap + impedance (exp045, log1p z-score norm).

Optimizes latent z so decode matches target heatmap and impedance in **train norm space**
(same as ``data_multifreq_norm``). Denorm to physical Ω only for reporting.

Run:
    .venv/bin/python experiments/exp045/codes/latent_optimization_joint.py \\
        --checkpoint experiments/exp045/checkpoints/last_model.pt \\
        --sample-index 0 --mhz 200 --steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.exp045.codes.exp045_eval_common import load_model, resolve_paths  # noqa: E402
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm  # noqa: E402


def _load_sample(data_dir: Path, index: int, device: torch.device) -> dict:
    from src_vae.others.dataloader import VAEDataset

    ds = VAEDataset(str(data_dir))
    if index < 0 or index >= len(ds):
        raise IndexError(f"sample index {index} out of range [0, {len(ds)})")
    row = ds[index]
    out = {}
    for k, v in row.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if k == "heatmap_norm" and t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif k == "heatmap_norm" and t.dim() == 3:
                t = t.unsqueeze(0)
            elif k == "impedance" and t.dim() == 1:
                t = t.unsqueeze(0).unsqueeze(0)
            elif k == "impedance" and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k == "occupancy" and t.dim() == 1:
                t = t.unsqueeze(0)
            elif k in ("K", "PI_freq") and t.dim() == 0:
                t = t.unsqueeze(0)
            out[k] = t
        else:
            out[k] = v
    return out


@torch.enable_grad()
def optimize_latent(
    model,
    *,
    hm_tgt: torch.Tensor,
    occ: torch.Tensor,
    imp_tgt: torch.Tensor,
    K: torch.Tensor,
    pi: torch.Tensor,
    steps: int,
    lr: float,
    w_hm: float,
    w_imp: float,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        z0, mu, logvar, _ = model.encode(hm_tgt, occ, imp_tgt, K, pi)
    z = mu.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    last: dict[str, float] = {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        hm_pred = model.decode_heatmap(z, K, pi, occupancy=occ)
        imp_pred = model.decode_impedance(z, K, pi)
        if imp_pred.dim() == 2:
            imp_pred = imp_pred.unsqueeze(1)
        loss_hm = F.huber_loss(hm_pred, hm_tgt, delta=huber_delta)
        loss_imp = F.mse_loss(imp_pred, imp_tgt)
        loss = w_hm * loss_hm + w_imp * loss_imp
        loss.backward()
        opt.step()
        last = {
            "loss": float(loss.item()),
            "hm": float(loss_hm.item()),
            "imp": float(loss_imp.item()),
        }
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss={last['loss']:.5f}  hm={last['hm']:.5f}  imp={last['imp']:.5f}")
    return z.detach(), last


def main() -> None:
    ap = argparse.ArgumentParser(description="exp045 joint heatmap+impedance latent optimization")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-hm", type=float, default=1.0)
    ap.add_argument("--w-imp", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=0.2)
    ap.add_argument("--out", type=str, default=None, help="Optional JSON metrics path")
    args = ap.parse_args()

    paths = resolve_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(paths["checkpoint"])
    model, _, _, _ = load_model(ckpt, device)

    batch = _load_sample(Path(paths["data_dir"]), args.sample_index, device)
    pi = torch.full((1,), pi_freq_mhz_to_norm(args.mhz), device=device, dtype=torch.float32)
    hm = batch["heatmap_norm"]
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)
    occ = batch["occupancy"]
    imp = batch["impedance"]
    if imp.dim() == 2:
        imp = imp.unsqueeze(1)
    K = batch["K"].long()
    if K.dim() == 0:
        K = K.unsqueeze(0)

    print(f"Checkpoint: {ckpt}")
    print(f"Sample {args.sample_index}  K={int(K.item())}  MHz={args.mhz}  steps={args.steps}")

    z_opt, metrics = optimize_latent(
        model,
        hm_tgt=hm,
        occ=occ,
        imp_tgt=imp,
        K=K,
        pi=pi,
        steps=args.steps,
        lr=args.lr,
        w_hm=args.w_hm,
        w_imp=args.w_imp,
        huber_delta=args.huber_delta,
    )

    with torch.no_grad():
        hm_final = model.decode_heatmap(z_opt, K, pi, occupancy=occ)
        imp_final = model.decode_impedance(z_opt, K, pi)
        metrics["hm_mse"] = float(F.mse_loss(hm_final, hm).item())
        metrics["imp_mse"] = float(F.mse_loss(imp_final.unsqueeze(1) if imp_final.dim() == 2 else imp_final, imp).item())

    print(f"Final hm MSE={metrics['hm_mse']:.5f}  imp MSE={metrics['imp_mse']:.5f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
