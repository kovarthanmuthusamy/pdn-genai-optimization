"""Find feasible configs — Monte Carlo sampling and filtering via VAE surrogate.

Run:
    python pipelines/latent/find_feasible.py"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/latent/find_feasible.py
# =============================================================================

CHECKPOINT_PATH = "experiments/exp029_heat_private/checkpoints/checkpoint_epoch_400.pt"

# Which K values to test
K_LIST = [1, 2, 3, 4, 5]

# Sampling budget per K
NUM_SAMPLES_PER_K = 2000

# Save up to N best feasible samples per K
SAVE_TOP_N = 20

# If True, ensure the saved feasible set contains unique occupancy top-K patterns.
# This is useful when there are many feasible solutions and you want multiple
# distinct configurations rather than near-duplicates.
ENSURE_UNIQUE_OCC_TOPK = True

# If you want to bias sampling toward the data manifold, use aggregate-posterior stats.
# - "agg_posterior": sample from checkpoint latent_stats/per_K_latent_stats (recommended)
# - "standard_normal": sample from N(0,1)
SAMPLE_MODE = "agg_posterior"  # "agg_posterior" or "standard_normal"

# Temperature multiplier for sampling
SHARED_TEMP = 1.0

# Constraint / loss space
LOSS_SPACE = "log"  # "log" (normalized log space) or "ohm"
BOUNDARY_MARGIN = 0.0

# IO
NORMALIZATION_STATS_PATH = "datasets/data_norm/normalization_stats.json"
TARGET_IMPEDANCE_PATH = "configs/target_impedance.npy"
OUTPUT_ROOT = "data/latent_runs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# =============================================================================
# Helpers
# =============================================================================


def _project_root() -> Path:
    from repo_paths import REPO_ROOT
    return REPO_ROOT


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class NormStats:
    imp_log_mean: float
    imp_log_std: float


def _load_norm_stats(stats_path: Path) -> NormStats:
    ns = _load_json(stats_path)
    imp = ns["Impedance"]
    return NormStats(
        imp_log_mean=float(imp["log_mean"]),
        imp_log_std=float(imp["log_std"]),
    )


def _blend_impedance_norm_log(imp_norm: torch.Tensor) -> torch.Tensor:
    if not (imp_norm.dim() == 3 and imp_norm.shape[1] >= 2 and imp_norm.shape[2] == 231):
        raise ValueError(f"Expected imp_norm shape (B,>=2,231), got {tuple(imp_norm.shape)}")

    z_raw = imp_norm[:, 0]
    d1 = imp_norm[:, 1]

    z_integ1 = torch.cat([z_raw[:, :1], z_raw[:, :1] + torch.cumsum(d1, dim=-1)[:, :-1]], dim=-1)

    if imp_norm.shape[1] >= 3:
        d2 = imp_norm[:, 2]
        peak_w = torch.abs(d2) / (torch.abs(d2).amax(dim=-1, keepdim=True) + 1e-8)
        pad = 2
        peak_w = F.pad(peak_w, (pad, pad), mode="reflect").unfold(-1, 2 * pad + 1, 1).mean(dim=-1)
        z_blend = peak_w * z_raw + (1.0 - peak_w) * z_integ1
    else:
        z_blend = 0.5 * (z_raw + z_integ1)

    return z_blend


def _denorm_impedance_log(imp_log_norm: torch.Tensor, stats: NormStats) -> torch.Tensor:
    return imp_log_norm * float(stats.imp_log_std) + float(stats.imp_log_mean)


def _latent_prior_params(
    *,
    latent_dim: int,
    device: torch.device,
    K: int,
    latent_stats: dict | None,
    per_K_latent_stats: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = latent_stats or {}

    if per_K_latent_stats is not None:
        k_key = str(int(K))
        if k_key in per_K_latent_stats:
            stats = per_K_latent_stats[k_key]
        else:
            try:
                available = [int(k) for k in per_K_latent_stats.keys()]
            except Exception:
                available = []
            if available:
                nearest = str(min(available, key=lambda x: abs(x - int(K))))
                stats = per_K_latent_stats[nearest]

    s = stats.get("latent", {})
    if "mu_mean_per_dim" in s and "agg_std_per_dim" in s:
        mu = torch.tensor(s["mu_mean_per_dim"], dtype=DTYPE, device=device).view(-1)
        std = torch.tensor(s["agg_std_per_dim"], dtype=DTYPE, device=device).view(-1)
        mu = mu[:latent_dim]
        std = std[:latent_dim]
        if mu.numel() < latent_dim:
            mu = F.pad(mu, (0, latent_dim - mu.numel()))
        if std.numel() < latent_dim:
            std = F.pad(std, (0, latent_dim - std.numel()), value=1.0)
        return mu, std.clamp(min=1e-6)

    mu_val = float(s.get("mu_mean", 0.0))
    agg_std = (float(s.get("mu_std", 1.0)) ** 2 + float(s.get("sigma_mean", 1.0)) ** 2) ** 0.5
    mu = torch.full((latent_dim,), mu_val, dtype=DTYPE, device=device)
    std = torch.full((latent_dim,), float(agg_std), dtype=DTYPE, device=device).clamp(min=1e-6)
    return mu, std


def _sample_z(
    *,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    K: int,
    latent_stats: dict | None,
    per_K_latent_stats: dict | None,
    shared_temp: float,
) -> torch.Tensor:
    mode = str(SAMPLE_MODE).lower().strip()

    if mode == "standard_normal":
        return torch.randn(num_samples, latent_dim, device=device, dtype=DTYPE) * float(shared_temp)

    if mode != "agg_posterior":
        raise SystemExit(f"SAMPLE_MODE must be 'agg_posterior' or 'standard_normal', got {SAMPLE_MODE!r}")

    mu, std = _latent_prior_params(
        latent_dim=latent_dim,
        device=device,
        K=int(K),
        latent_stats=latent_stats,
        per_K_latent_stats=per_K_latent_stats,
    )
    return torch.randn(num_samples, latent_dim, device=device, dtype=DTYPE) * (std * float(shared_temp)) + mu


def _topk_occupancy(occ_prob: torch.Tensor, K: int) -> torch.Tensor:
    if occ_prob.dim() != 2 or occ_prob.shape[1] != 52:
        raise ValueError(f"Expected occ_prob shape (B,52), got {tuple(occ_prob.shape)}")
    occ_bin = torch.zeros_like(occ_prob)
    if int(K) > 0:
        idx = occ_prob.topk(min(int(K), occ_prob.shape[-1]), dim=-1).indices
        occ_bin.scatter_(-1, idx, 1.0)
    return occ_bin


@torch.no_grad()
def _decode_occupancy_prob(
    *,
    model: torch.nn.Module,
    z: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode occupancy probabilities (B,52) without decoding heatmap."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.dim() != 2:
        raise ValueError(f"Expected z shape (B,D) or (D,), got {tuple(z.shape)}")
    B = z.shape[0]
    K_tensor = torch.full((B,), int(K), dtype=torch.long, device=device)

    z_shared = z[:, : int(getattr(model, "shared_latent_dim"))]
    k_emb = getattr(model, "k_embedding")(K_tensor)
    dec = torch.cat([z_shared, k_emb], dim=1)
    occ_logits = getattr(model, "occupancy_decoder")(dec)
    return torch.sigmoid(occ_logits)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    project_root = _project_root()
    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.exp029_heat_private.codes.vae_multi_input_simple import MultiInputVAE  # noqa: WPS433

    device = torch.device(DEVICE)

    stats = _load_norm_stats(project_root / NORMALIZATION_STATS_PATH)

    target_imp_ohm = np.load(project_root / TARGET_IMPEDANCE_PATH).astype(np.float32).squeeze()
    if target_imp_ohm.shape != (231,):
        raise ValueError(f"Expected target impedance shape (231,), got {target_imp_ohm.shape}")

    target_imp_ohm_t = torch.tensor(target_imp_ohm, device=device, dtype=DTYPE).view(1, -1)
    target_imp_log_t = torch.log(target_imp_ohm_t.clamp(min=1e-12))
    target_imp_log_norm_t = (target_imp_log_t - float(stats.imp_log_mean)) / max(float(stats.imp_log_std), 1e-12)

    ckpt_path = project_root / CHECKPOINT_PATH
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    latent_dim = int(cfg.get("latent_dim", 32))
    cond_dim = int(cfg.get("cond_dim", 8))
    heatmap_private_dim = int(cfg.get("heatmap_private_dim", 8))

    model = MultiInputVAE(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        heatmap_private_dim=heatmap_private_dim,
        modality_dropout=0.0,
    ).to(device=device, dtype=DTYPE)

    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    compat = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    latent_stats = ckpt.get("latent_stats", None)
    per_K_latent_stats = ckpt.get("per_K_latent_stats", None)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        project_root
        / OUTPUT_ROOT
        / f"feasible_samples_{Path(CHECKPOINT_PATH).stem}_{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "checkpoint": CHECKPOINT_PATH,
        "device": DEVICE,
        "latent_dim": latent_dim,
        "K_list": K_LIST,
        "num_samples_per_k": NUM_SAMPLES_PER_K,
        "save_top_n": SAVE_TOP_N,
        "sample_mode": SAMPLE_MODE,
        "shared_temp": SHARED_TEMP,
        "loss_space": LOSS_SPACE,
        "boundary_margin": BOUNDARY_MARGIN,
        "normalization_stats": asdict(stats),
    }
    (out_root / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    for K in K_LIST:
        if not (0 <= int(K) <= 52):
            raise SystemExit(f"Invalid K={K} (must be 0..52)")

        with torch.no_grad():
            z = _sample_z(
                num_samples=int(NUM_SAMPLES_PER_K),
                latent_dim=latent_dim,
                device=device,
                K=int(K),
                latent_stats=latent_stats,
                per_K_latent_stats=per_K_latent_stats,
                shared_temp=float(SHARED_TEMP),
            )

            K_tensor = torch.full((z.shape[0],), int(K), device=device, dtype=torch.long)
            imp_norm = model.decode_impedance(z, K_tensor)  # (B,3,231)
            imp_log_norm = _blend_impedance_norm_log(imp_norm)  # (B,231)
            imp_log = _denorm_impedance_log(imp_log_norm, stats)  # (B,231)
            imp_ohm = torch.exp(imp_log).clamp(min=1e-12)

            if LOSS_SPACE == "ohm":
                imp_curve = imp_ohm
                target_curve = target_imp_ohm_t.expand(z.shape[0], -1)
            elif LOSS_SPACE == "log":
                imp_curve = imp_log_norm
                target_curve = target_imp_log_norm_t.expand(z.shape[0], -1)
            else:
                raise SystemExit(f"LOSS_SPACE must be 'log' or 'ohm', got {LOSS_SPACE!r}")

            boundary = target_curve - float(BOUNDARY_MARGIN)
            violation = (imp_curve - boundary).clamp(min=0.0)
            max_violation = violation.amax(dim=1)
            feasible = max_violation == 0.0

            # Rank feasible samples by latent prior penalty (MAP-ish).
            if str(SAMPLE_MODE).lower().strip() == "agg_posterior":
                prior_mu, prior_std = _latent_prior_params(
                    latent_dim=latent_dim,
                    device=device,
                    K=int(K),
                    latent_stats=latent_stats,
                    per_K_latent_stats=per_K_latent_stats,
                )
                prior_pen = ((z - prior_mu) / prior_std).pow(2).mean(dim=1)
            else:
                prior_pen = z.pow(2).mean(dim=1)

            feasible_idx = torch.where(feasible)[0]
            num_feas = int(feasible_idx.numel())

            print(
                f"K={int(K):02d} | feasible {num_feas}/{int(NUM_SAMPLES_PER_K)} "
                f"({100.0 * num_feas / max(1, int(NUM_SAMPLES_PER_K)):.2f}%)"
            )

            if num_feas == 0:
                # Still save a small diagnostic of the best (least violating) sample.
                best_i = int(torch.argmin(max_violation).item())
                diag_dir = out_root / f"K{int(K):02d}" / "diagnostic_best_violation"
                diag_dir.mkdir(parents=True, exist_ok=True)

                hm, occ_logits, _ = model.decode(z[best_i : best_i + 1], torch.tensor([int(K)], device=device, dtype=torch.long))
                occ_prob = torch.sigmoid(occ_logits)
                occ_topk = _topk_occupancy(occ_prob, int(K))

                np.save(diag_dir / "latent.npy", z[best_i : best_i + 1].cpu().numpy())
                np.save(diag_dir / "impedance_log.npy", imp_log[best_i : best_i + 1].cpu().numpy())
                np.save(diag_dir / "impedance_ohm.npy", imp_ohm[best_i : best_i + 1].cpu().numpy())
                np.save(diag_dir / "heatmap.npy", hm.cpu().numpy())
                np.save(diag_dir / "occupancy_prob.npy", occ_prob.cpu().numpy())
                np.save(diag_dir / "occupancy_topk.npy", occ_topk.cpu().numpy().astype(np.int8))

                metrics = {
                    "K": int(K),
                    "max_violation": float(max_violation[best_i].item()),
                    "prior_pen": float(prior_pen[best_i].item()),
                }
                (diag_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                continue

            # Decode occupancy (cheap) for diversity filtering / saving.
            occ_prob_all = _decode_occupancy_prob(model=model, z=z, K=int(K), device=device)
            occ_topk_all = _topk_occupancy(occ_prob_all, int(K)).to(dtype=torch.int8)

            feas_prior = prior_pen[feasible]
            order_all = torch.argsort(feas_prior)

            picked_list: list[int] = []
            seen_topk: set[tuple[int, ...]] = set()
            for ord_i in order_all.tolist():
                idx = int(feasible_idx[ord_i].item())
                if bool(ENSURE_UNIQUE_OCC_TOPK):
                    on = torch.where(occ_topk_all[idx].to(dtype=torch.bool))[0].tolist()
                    key = tuple(on)
                    if key in seen_topk:
                        continue
                    seen_topk.add(key)
                picked_list.append(idx)
                if len(picked_list) >= int(SAVE_TOP_N):
                    break

            picked = torch.tensor(picked_list, device=device, dtype=torch.long)

            k_dir = out_root / f"K{int(K):02d}"
            k_dir.mkdir(parents=True, exist_ok=True)

            for j, idx in enumerate(picked.tolist()):
                sol_dir = k_dir / f"sample_{j:03d}"
                sol_dir.mkdir(parents=True, exist_ok=True)

                hm, occ_logits, _ = model.decode(z[idx : idx + 1], torch.tensor([int(K)], device=device, dtype=torch.long))
                # Use occupancy from the cheap path to keep it consistent with the selection.
                occ_prob = occ_prob_all[idx : idx + 1]
                occ_topk = occ_topk_all[idx : idx + 1]

                np.save(sol_dir / "latent.npy", z[idx : idx + 1].cpu().numpy())
                np.save(sol_dir / "impedance_log.npy", imp_log[idx : idx + 1].cpu().numpy())
                np.save(sol_dir / "impedance_ohm.npy", imp_ohm[idx : idx + 1].cpu().numpy())
                np.save(sol_dir / "heatmap.npy", hm.cpu().numpy())
                np.save(sol_dir / "occupancy_prob.npy", occ_prob.cpu().numpy())
                np.save(sol_dir / "occupancy_topk.npy", occ_topk.cpu().numpy().astype(np.int8))

                metrics = {
                    "K": int(K),
                    "max_violation": float(max_violation[idx].item()),
                    "prior_pen": float(prior_pen[idx].item()),
                }
                (sol_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nDone. Results in: {out_root}")


if __name__ == "__main__":
    main()
