"""Diagnose exp040 sweep artifacts: fixed hotspot, K/MHz invariance, residual vs factorized.

Run from repo root (with project venv + CUDA optional):
    python experiments/exp040/codes/inspect_sweep_artifacts.py
    python experiments/exp040/codes/inspect_sweep_artifacts.py --ckpt experiments/exp040/checkpoints/checkpoint_epoch_250.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.exp040.codes.inference_vae import VAEInference
from src_vae.others.pi_freq_utils import pi_freq_norm_for_model


def _hm_phys(hm_z: torch.Tensor, engine) -> np.ndarray:
    from src_vae.others.heatmap_z_clip import heatmap_z_to_physical

    clip = getattr(engine, "hm_z_clip", None)
    lo, hi = (clip[0], clip[1]) if clip is not None else (None, None)
    p = heatmap_z_to_physical(
        hm_z, engine.hm_log_mean, engine.hm_log_std, clip_lo=lo, clip_hi=hi,
    )
    return p[0, 0].detach().cpu().numpy()


def _peak_yx(plane: np.ndarray, mask: np.ndarray) -> tuple[int, int, float]:
    m = plane.copy()
    m[~mask] = 0.0
    iy, ix = np.unravel_index(m.argmax(), m.shape)
    return int(iy), int(ix), float(m[mask].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/exp040/checkpoints/last_model.pt")
    ap.add_argument("--pi-ref", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = VAEInference(str(ckpt), device=device)
    stats = _ROOT / "experiments/exp040/metrics/latent_stats.json"
    if stats.is_file():
        eng.load_latent_stats(str(stats))

    mask = eng.binary_mask.astype(bool)
    model = eng.model
    model.eval()
    torch.manual_seed(args.seed)

    K_list = [5, 25, 45]
    mhz_list = [70.0, 270.0, 400.0]

    print(f"Checkpoint: {ckpt}")
    print(f"pi_ref_mhz={args.pi_ref}  device={device}\n")

    with torch.no_grad():
        _, occ0, imp0 = model.inference(
            1,
            device,
            K=25,
            PI_freq=args.pi_ref,
            mode="marginal",
            latent_stats=eng.latent_stats,
            per_K_latent_stats=eng.per_K_latent_stats,
            shared_temp=1.5,
        )

    print("=== z / bases vs K (same occ/imp, pi_ref encode, 400 MHz decode) ===")
    pi_ref = pi_freq_norm_for_model(args.pi_ref, 1, unit="mhz", device=device)
    z_by_k: dict[int, torch.Tensor] = {}
    b_by_k: dict[int, torch.Tensor] = {}
    for K in K_list:
        K_t = torch.tensor([K], device=device)
        with torch.no_grad():
            z_by_k[K] = model.encode_layout_latent(occ0, imp0, K_t, pi_ref)
            b_by_k[K] = model._decode_bases(z_by_k[K], K_t)
    z25 = z_by_k[25]
    b25 = b_by_k[25]
    for K in K_list:
        dz = (z_by_k[K] - z25).norm().item()
        db = (b_by_k[K] - b25).norm().item()
        print(f"  K={K:2d}  ||z-z_K25||={dz:.4f}  ||bases-bases_K25||={db:.4f}")
    print("  (alpha is MHz-only in exp040 — identical alpha across K is expected.)\n")

    model.inference_factorized_only = True
    print("=== anchor_blend, factorized_only=True (shared occ/imp from K=25 marginal) ===")
    print("K     MHz    peak(y,x)   fg_max   alpha (6 modes)")
    for K in K_list:
        for mhz in mhz_list:
            with torch.no_grad():
                hm, _, _ = model.inference(
                    1,
                    device,
                    K=K,
                    PI_freq=mhz,
                    mode="anchor_blend",
                    occupancy=occ0,
                    impedance=imp0,
                    pi_ref_mhz=args.pi_ref,
                    latent_stats=eng.latent_stats,
                    per_K_latent_stats=eng.per_K_latent_stats,
                    shared_temp=1.5,
                )
            plane = _hm_phys(hm, eng)
            iy, ix, mx = _peak_yx(plane, mask)
            alpha = model._last_alpha
            a = alpha[0].cpu().numpy().round(3).tolist() if alpha is not None else []
            print(f"{K:3d}  {mhz:5.0f}  ({iy:2d},{ix:2d})   {mx:7.2f}  {a}")

    print(
        "\n=== encode ablation: PI_freq in encoder = target MHz vs pi_ref (sweep uses pi_ref) ===",
    )
    K_t = torch.tensor([25], device=device)
    for mhz in mhz_list:
        pi_tgt = pi_freq_norm_for_model(mhz, 1, unit="mhz", device=device)
        z_tgt = model.encode_layout_latent(occ0, imp0, K_t, pi_tgt)
        z_ref = model.encode_layout_latent(occ0, imp0, K_t, pi_ref)
        dz = (z_tgt - z_ref).norm().item()
        print(f"  encode@{mhz:5.0f} vs pi_ref@{args.pi_ref:.0f}: ||Δz|| = {dz:.4f}")

    print("\n=== anchor_blend per-K marginal occ/imp (real sweep path) ===")
    print("K     MHz    peak(y,x)   fg_max")
    for K in K_list:
        with torch.no_grad():
            _, occ_k, imp_k = model.inference(
                1,
                device,
                K=K,
                PI_freq=args.pi_ref,
                mode="marginal",
                latent_stats=eng.latent_stats,
                per_K_latent_stats=eng.per_K_latent_stats,
                shared_temp=1.5,
            )
        for mhz in mhz_list:
            with torch.no_grad():
                hm, _, _ = model.inference(
                    1,
                    device,
                    K=K,
                    PI_freq=mhz,
                    mode="anchor_blend",
                    occupancy=occ_k,
                    impedance=imp_k,
                    pi_ref_mhz=args.pi_ref,
                    latent_stats=eng.latent_stats,
                    per_K_latent_stats=eng.per_K_latent_stats,
                    shared_temp=1.5,
                )
            plane = _hm_phys(hm, eng)
            iy, ix, mx = _peak_yx(plane, mask)
            print(f"{K:3d}  {mhz:5.0f}  ({iy:2d},{ix:2d})   {mx:7.2f}")

    model.inference_factorized_only = False
    print("\n=== factorized vs residual @ 400 MHz, K=25 ===")
    with torch.no_grad():
        pi = pi_freq_norm_for_model(400.0, 1, unit="mhz", device=device)
        K_t = torch.tensor([25], device=device)
        z = model.encode_layout_latent(occ0, imp0, K_t, pi_ref)
        bases = model._decode_bases(z, K_t)
        hm_fac, _ = model._mix_bases(bases, pi)
        hm_res, _, _ = super(type(model), model).decode(z, K_t, pi)
        rg = float(model.residual_gain.item()) if model.residual_gain is not None else 0.0
        hm_full = hm_fac + rg * hm_res
        for name, t in [("factorized", hm_fac), ("residual", hm_res), ("full", hm_full)]:
            plane = _hm_phys(t, eng)
            iy, ix, mx = _peak_yx(plane, mask)
            print(f"  {name:12s} peak=({iy},{ix}) fg_max={mx:.2f}  residual_gain={rg:.3f}")

    print(
        "\nIf all peaks share the same (y,x) across K and MHz, the decoder has collapsed to a "
        "fixed spatial mode. If residual fg_max >> factorized, lower residual_gain or disable "
        "use_freq_residual in config. For sweep, set CALIBRATE_FG_MAX=False in "
        "run_multifreq_heatmap_sweep.py to avoid scaling a blob to anchor statistics."
    )


if __name__ == "__main__":
    main()
