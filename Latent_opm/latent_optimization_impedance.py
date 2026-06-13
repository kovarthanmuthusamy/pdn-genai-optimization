"""Latent optimization: find z (frozen VAE decoder) so impedance stays below target.

Uses **exp038_true_multi** checkpoint (multifreq-trained VAE). Occ + impedance decoders
are K-only; PI_freq is not used here (layout spectrum optimization only).

    python Latent_opm/latent_optimization_impedance.py

Outputs → Latent_opm/runs/<experiment>/0/, 1/, 2/, … (K##_/best_* or no_solution.json)
"""

from __future__ import annotations

import importlib.util
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.exp038_true_multi.codes.vae_multi_input_simple import MultiInputVAE  # noqa: E402

# ── Config (exp038_true_multi) ───────────────────────────────────────────────

EXPERIMENT = "exp038_true_multi"
CHECKPOINT_PATH = f"experiments/{EXPERIMENT}/checkpoints/last_model.pt"
SURROGATE_CHECKPOINT_PATH = f"experiments/{EXPERIMENT}/checkpoints/surrogate_best.pt"


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


USE_SURROGATE = _env_flag("LATENT_OPT_USE_SURROGATE", "1")

K_LIST = list(range(1, 26))
NUM_CANDIDATE_SEEDS = 32
# "random": new integer seeds each run (saved in run_config.json). "fixed": 0..N-1 (reproducible).
SEED_MODE = os.getenv("LATENT_OPT_SEED_MODE", "random").strip().lower()
# If set, random mode draws seeds from np.random.default_rng(int(MASTER_SEED)). Unset = random pool per run.
def _parse_master_seed() -> int | None:
    raw = os.getenv("LATENT_OPT_MASTER_SEED", "").strip()
    if not raw:
        return None
    return int(raw)


MASTER_SEED: int | None = _parse_master_seed()
OPTIMIZE_BATCH_PER_K = True

NUM_STEPS = 1200
LR = 5e-2
GRAD_CLIP = 5.0
INIT_SHARED_TEMP = 0.8

OBJECTIVE_MODE = "gap_max"  # "gap_max" | "feasible_map"
LOSS_SPACE = "log"          # "log" | "ohm"

Z_L2_WEIGHT = 0.05
Z_PRIOR_MODE = "agg_posterior"
SHAPE_REG_WEIGHT = 2.0
SHAPE_MIN_STD_DLOG = 0.12
SHAPE_TRACK_WEIGHT = 2.0
POSTERIOR_BOUNDARY_WEIGHT = 10.0
POSTERIOR_SIGMA_LIMIT = 2.5
PHYSICS_AR_WEIGHT = 0.5

# Peak-aware spectrum terms (aligned with exp038 impedance_spectrum_loss)
PEAK_LOSS_WEIGHT = float(os.getenv("LATENT_OPT_PEAK_LOSS_WEIGHT", "1.0"))
PEAK_INDEX_WEIGHT = float(os.getenv("LATENT_OPT_PEAK_INDEX_WEIGHT", "3.0"))
PEAK_MAG_WEIGHT = float(os.getenv("LATENT_OPT_PEAK_MAG_WEIGHT", "2.0"))
DUAL_TOPK_WEIGHT = float(os.getenv("LATENT_OPT_DUAL_TOPK_WEIGHT", "2.5"))
IMPEDANCE_NUM_PEAKS = int(os.getenv("LATENT_OPT_IMPEDANCE_NUM_PEAKS", "8"))
IMPEDANCE_TOPK_K = int(os.getenv("LATENT_OPT_IMPEDANCE_TOPK_K", "20"))

DIVERSITY_WEIGHT = 0.35
OCC_CONFIDENCE_WEIGHT = 0.5

BOUNDARY_MARGIN = 0.1
GAP_REWARD_WEIGHT = 1.0
EXCEED_WEIGHT = 25.0
EXCEED_POWER = 2.0
SELECT_METRIC = "max_ohm"

NORMALIZATION_STATS_PATH = "datasets/data_multifreq_norm/normalization_stats.json"
TARGET_IMPEDANCE_PATH = "configs/target_impedance.npy"
OUTPUT_ROOT = "Latent_opm/runs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

HIST_KEYS = ("total", "exceed", "gap_mean", "best_score")


# ── Helpers ───────────────────────────────────────────────────────────────────


SOLUTION_PREFIX = "best_"
SOLUTION_LATENT_NAME = f"{SOLUTION_PREFIX}latent.npy"
SOLUTION_METRICS_NAME = f"{SOLUTION_PREFIX}metrics.json"


def experiment_runs_dir(root: Path, experiment: str = EXPERIMENT) -> Path:
    return root / OUTPUT_ROOT / experiment


def resolve_run_dir(root: Path, explicit: str | Path | None = None, experiment: str = EXPERIMENT) -> Path:
    """Resolve a latent-opt run folder (explicit path or latest under runs/<experiment>/)."""
    if explicit is not None:
        p = Path(explicit)
        return (root / p).resolve() if not p.is_absolute() else p.resolve()

    base = experiment_runs_dir(root, experiment)
    if not base.is_dir():
        raise FileNotFoundError(f"No runs under {base}")

    run_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        cfgs = [p.parent for p in base.rglob("run_config.json")]
        if cfgs:
            return max(cfgs, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(f"No run folders under {base}")

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def norm_stats_from_run_config(run_cfg: dict[str, Any], root: Path) -> NormStats:
    if isinstance(run_cfg.get("imp_log_norm"), dict):
        d = run_cfg["imp_log_norm"]
        return NormStats(float(d["imp_log_mean"]), float(d["imp_log_std"]))
    rel = str(run_cfg.get("normalization_stats_path", NORMALIZATION_STATS_PATH))
    path = Path(rel)
    if not path.is_absolute():
        path = (root / path).resolve()
    return _load_norm_stats(path)


def allocate_run_dir(root: Path, experiment: str) -> tuple[Path, int]:
    """Next run folder: Latent_opm/runs/<experiment>/<0|1|2|...>."""
    base = root / OUTPUT_ROOT / experiment
    base.mkdir(parents=True, exist_ok=True)
    indices = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    run_idx = (max(indices) + 1) if indices else 0
    run_dir = base / str(run_idx)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_idx


def resolve_candidate_seeds() -> tuple[list[int], dict[str, Any]]:
    """Build per-run candidate seeds and metadata for run_config.json."""
    n = int(os.getenv("LATENT_OPT_NUM_SEEDS", str(NUM_CANDIDATE_SEEDS)))
    if SEED_MODE in ("fixed", "legacy"):
        seeds = list(range(n))
        return seeds, {"mode": "fixed", "master_seed": None, "num_seeds": n, "seeds": seeds}
    pool_master = MASTER_SEED if MASTER_SEED is not None else int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(pool_master)
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]
    return seeds, {"mode": "random", "master_seed": pool_master, "num_seeds": n, "seeds": seeds}


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "experiments").is_dir() and (p / "datasets").is_dir():
            return p
    return here.parents[1]


@dataclass(frozen=True)
class NormStats:
    imp_log_mean: float
    imp_log_std: float


def _load_norm_stats(path: Path) -> NormStats:
    imp = json.loads(path.read_text(encoding="utf-8"))["Impedance"]
    return NormStats(float(imp["log_mean"]), float(imp["log_std"]))


def _imp_ch0(imp_norm: torch.Tensor) -> torch.Tensor:
    if imp_norm.dim() == 3:
        return imp_norm[:, 0]
    if imp_norm.dim() == 2:
        return imp_norm
    raise ValueError(f"bad imp shape {tuple(imp_norm.shape)}")


def _curves_from_norm(imp_norm: torch.Tensor, stats: NormStats) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_norm = _imp_ch0(imp_norm)
    log_ohm = log_norm * stats.imp_log_std + stats.imp_log_mean
    return log_norm, log_ohm, torch.exp(log_ohm).clamp(min=1e-12)


def _k_prior(
    K: int, latent_dim: int, device: torch.device,
    latent_stats: dict | None, per_K: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s: dict = latent_stats or {}
    if per_K:
        k_key = str(K)
        if k_key not in per_K:
            keys = [int(k) for k in per_K]
            k_key = str(min(keys, key=lambda x: abs(x - K))) if keys else k_key
        s = per_K.get(k_key, s)
    lat = s.get("latent", {})
    if "mu_mean_per_dim" in lat and "agg_std_per_dim" in lat:
        mu = torch.tensor(lat["mu_mean_per_dim"], dtype=DTYPE, device=device).view(-1)[:latent_dim]
        std = torch.tensor(lat["agg_std_per_dim"], dtype=DTYPE, device=device).view(-1)[:latent_dim]
        if mu.numel() < latent_dim:
            mu = F.pad(mu, (0, latent_dim - mu.numel()))
        if std.numel() < latent_dim:
            std = F.pad(std, (0, latent_dim - std.numel()), value=1.0)
    else:
        mu_v = float(lat.get("mu_mean", 0.0))
        std_v = (float(lat.get("mu_std", 1.0)) ** 2 + float(lat.get("sigma_mean", 1.0)) ** 2) ** 0.5
        mu = torch.full((latent_dim,), mu_v, dtype=DTYPE, device=device)
        std = torch.full((latent_dim,), std_v, dtype=DTYPE, device=device)
    return mu, std.clamp(min=1e-6)


def _sample_z(
    K: int, n: int, latent_dim: int, device: torch.device,
    latent_stats: dict | None, per_K: dict | None,
) -> torch.Tensor:
    mu, std = _k_prior(K, latent_dim, device, latent_stats, per_K)
    return torch.randn(n, latent_dim, device=device, dtype=DTYPE) * (std * INIT_SHARED_TEMP) + mu


def _decode_occ(model: torch.nn.Module, z: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    if z.dim() == 1:
        z = z.unsqueeze(0)
    K_t = torch.full((z.shape[0],), K, dtype=torch.long, device=device)
    z_sh = z[:, : model.shared_latent_dim]
    dec = torch.cat([z_sh, model.k_embedding(K_t)], dim=1)
    return torch.sigmoid(model.occupancy_decoder(dec))


def _ste_binary(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.5).to(x.dtype) + x - x.detach()


def _ste_topk(occ_prob: torch.Tensor, K: int) -> torch.Tensor:
    """Hard top-K mask in the forward pass, identity gradient to ``occ_prob``.

    Forward value equals the discrete top-K binary vector (exactly K ones, the
    surrogate's training distribution); backward flows ``d/d occ_prob = 1`` so the
    impedance/peak losses can reshape which slots get selected. Without this the
    top-K argsort severs the gradient and z never moves on the spectrum target.
    """
    hard = _topk_occ(occ_prob, K)
    return hard + occ_prob - occ_prob.detach()


def _forward_imp(
    model: torch.nn.Module,
    surrogate: torch.nn.Module | None,
    z: torch.Tensor,
    K: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    K_t = torch.full((z.shape[0],), K, dtype=torch.long, device=device)
    if surrogate is not None:
        occ = _decode_occ(model, z, K, device)
        return surrogate(_ste_topk(occ, K)), occ
    return model.decode_impedance(z, K_t), _decode_occ(model, z, K, device)


def _peak_spectrum_loss(imp_curve: torch.Tensor, target_row: torch.Tensor) -> torch.Tensor:
    """Peak index + magnitude + dual top-k vs target (log or ohm space)."""
    from experiments.exp038_true_multi.codes.impedance_spectrum_loss import (
        dual_topk_loss,
        peak_alignment_loss,
    )

    r = imp_curve.unsqueeze(0) if imp_curve.dim() == 1 else imp_curve
    t = target_row.unsqueeze(0).expand(r.shape[0], -1) if target_row.dim() == 1 else target_row
    k = min(IMPEDANCE_TOPK_K, t.shape[-1])
    loss = r.new_zeros(r.shape[0])
    if DUAL_TOPK_WEIGHT > 0 and k > 0:
        loss = loss + DUAL_TOPK_WEIGHT * dual_topk_loss(r, t, k, under_penalty=1.0)
    if PEAK_INDEX_WEIGHT > 0 or PEAK_MAG_WEIGHT > 0:
        p_idx, p_mag = peak_alignment_loss(r, t, IMPEDANCE_NUM_PEAKS)
        loss = loss + PEAK_INDEX_WEIGHT * p_idx + PEAK_MAG_WEIGHT * p_mag
    return loss


def _anti_resonance_loss(imp: torch.Tensor) -> torch.Tensor:
    ch0 = imp[:, 0:1, :]
    left, center, right = ch0[:, :, :-2], ch0[:, :, 1:-1], ch0[:, :, 2:]
    peak = torch.minimum(F.relu(center - left), F.relu(center - right))
    dip = torch.minimum(F.relu(left - center), F.relu(right - center))
    prior = torch.cat([torch.zeros_like(dip[:, :, :1]), torch.cummax(dip, dim=-1).values[:, :, :-1]], dim=-1)
    return (peak * torch.exp(-10.0 * prior)).mean()


def _base_loss(imp_curve: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary = target - BOUNDARY_MARGIN
    if target.dim() == 1:
        boundary = boundary.unsqueeze(0)
    if imp_curve.dim() == 1:
        imp_curve = imp_curve.unsqueeze(0)
    boundary = boundary.expand(imp_curve.shape[0], -1)
    above = (imp_curve - boundary).clamp(min=0.0)
    viol = above.pow(EXCEED_POWER)
    below = (boundary - imp_curve).clamp(min=0.0)
    exceed = viol.mean(dim=1) + viol.amax(dim=1)
    gap = below.mean(dim=1)
    if OBJECTIVE_MODE == "gap_max":
        base = -GAP_REWARD_WEIGHT * gap + EXCEED_WEIGHT * exceed
    else:
        base = EXCEED_WEIGHT * exceed
    return base, exceed, gap


def _z_prior_loss(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if Z_PRIOR_MODE == "agg_posterior":
        return ((z - mu) / std).pow(2).mean(dim=-1)
    return z.pow(2).mean(dim=-1)


def _shape_reg(imp_log: torch.Tensor) -> torch.Tensor:
    dlog = imp_log[:, 1:] - imp_log[:, :-1]
    return (SHAPE_MIN_STD_DLOG - dlog.std(dim=1)).clamp(min=0.0).pow(2)


def _posterior_boundary(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    r = ((z - mu) / std).pow(2).mean(dim=-1).sqrt()
    return (r - POSTERIOR_SIGMA_LIMIT).clamp(min=0.0).pow(2)


def _diversity_loss(occ: torch.Tensor) -> torch.Tensor:
    B = occ.shape[0]
    if B <= 1:
        return occ.new_zeros(())
    x = occ / (occ.norm(dim=1, keepdim=True) + 1e-8)
    sim = x @ x.t()
    return (sim.sum() - sim.diagonal().sum()) / (B * (B - 1))


def _is_feasible(imp_curve: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boundary = target - BOUNDARY_MARGIN
    if imp_curve.dim() == 1:
        imp_curve = imp_curve.unsqueeze(0)
    if boundary.dim() == 1:
        boundary = boundary.unsqueeze(0)
    return (imp_curve <= boundary.expand(imp_curve.shape[0], -1)).all(dim=1)


def _select_score(imp_ohm: torch.Tensor) -> torch.Tensor:
    if imp_ohm.dim() == 1:
        imp_ohm = imp_ohm.unsqueeze(0)
    return imp_ohm.max(dim=1).values if SELECT_METRIC == "max_ohm" else imp_ohm.mean(dim=1)


def _compute_loss(
    z: torch.Tensor,
    imp_norm: torch.Tensor,
    occ_prob: torch.Tensor,
    target_row: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_std: torch.Tensor,
    stats: NormStats,
    *,
    physics_ar_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns total (scalar), imp_curve (B,231), imp_log, imp_ohm."""
    log_norm, log_ohm, ohm = _curves_from_norm(imp_norm, stats)
    imp_curve = log_norm if LOSS_SPACE == "log" else ohm
    target = target_row.expand(imp_curve.shape[0], -1)

    base, exceed, gap = _base_loss(imp_curve, target)
    z_l2 = _z_prior_loss(z, prior_mu, prior_std)
    shape = _shape_reg(log_norm if LOSS_SPACE == "log" else log_ohm) if SHAPE_REG_WEIGHT > 0 else z_l2.new_zeros(z_l2.shape)
    post = _posterior_boundary(z, prior_mu, prior_std) if POSTERIOR_BOUNDARY_WEIGHT > 0 else z_l2.new_zeros(z_l2.shape)
    track = (
        F.mse_loss(imp_curve, target, reduction="none").mean(dim=1)
        if SHAPE_TRACK_WEIGHT > 0
        else z_l2.new_zeros(z_l2.shape)
    )
    ar = _anti_resonance_loss(imp_norm) if physics_ar_weight > 0 else z.new_zeros(())
    peak = (
        _peak_spectrum_loss(imp_curve, target_row).mean()
        if PEAK_LOSS_WEIGHT > 0
        else z.new_zeros(())
    )

    total = (
        base.mean()
        + Z_L2_WEIGHT * z_l2.mean()
        + SHAPE_REG_WEIGHT * shape.mean()
        + POSTERIOR_BOUNDARY_WEIGHT * post.mean()
        + SHAPE_TRACK_WEIGHT * track.mean()
        + physics_ar_weight * ar
        + PEAK_LOSS_WEIGHT * peak
    )
    if z.shape[0] > 1 and OPTIMIZE_BATCH_PER_K:
        total = total + DIVERSITY_WEIGHT * _diversity_loss(occ_prob) + OCC_CONFIDENCE_WEIGHT * (occ_prob * (1 - occ_prob)).mean()

    return total, imp_curve, log_ohm, ohm


def _topk_occ(occ_prob: torch.Tensor, K: int) -> torch.Tensor:
    out = torch.zeros_like(occ_prob)
    if K > 0:
        out.scatter_(-1, occ_prob.topk(min(K, occ_prob.shape[-1]), dim=-1).indices, 1.0)
    return out


def _save_solution(k_dir: Path, best: dict[str, Any], model: torch.nn.Module, K: int, device: torch.device) -> None:
    p = "best_"
    z, log_ohm, ohm = best["z"], best["imp_log"], best["imp_ohm"]
    np.save(k_dir / f"{p}latent.npy", z.cpu().numpy())
    np.save(k_dir / f"{p}impedance_log.npy", log_ohm.cpu().numpy())
    np.save(k_dir / f"{p}impedance_ohm.npy", ohm.cpu().numpy())
    with torch.no_grad():
        occ = _decode_occ(model, z, K, device)
    np.save(k_dir / f"{p}occupancy_prob.npy", occ.cpu().numpy())
    topk = _topk_occ(occ, K)
    topk_np = topk.cpu().numpy().astype(np.int8)
    np.save(k_dir / f"{p}occupancy_topk.npy", topk_np)
    from Data_Creation.csv_to_occupancy import labels_v1

    metrics = dict(best["metrics"])
    metrics["active_components"] = [
        labels_v1[i] for i in range(min(52, topk_np.size)) if int(topk_np.reshape(-1)[i])
    ]
    metrics["n_active"] = len(metrics["active_components"])
    (k_dir / f"{p}metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _optimize_k(
    K: int,
    model: torch.nn.Module,
    surrogate: torch.nn.Module | None,
    target_row: torch.Tensor,
    stats: NormStats,
    latent_dim: int,
    latent_stats: dict | None,
    per_K: dict | None,
    device: torch.device,
    physics_ar_weight: float,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, list], str, int]:
    """Returns best dict, history, mode, n_candidates."""
    mu, std = _k_prior(K, latent_dim, device, latent_stats, per_K)
    hist: dict[str, list] = {k: [] for k in HIST_KEYS}
    best: dict[str, Any] = {"score": float("inf"), "step": -1, "seed": -1, "z": None}

    def _update_best(feasible: torch.Tensor, scores: torch.Tensor, z: torch.Tensor, log_ohm: torch.Tensor, ohm: torch.Tensor, step: int, seeds: list[int]) -> None:
        for b in range(feasible.shape[0]):
            if not feasible[b].item():
                continue
            s = float(scores[b].item())
            if s < best["score"]:
                best.update(
                    score=s, step=step, seed=seeds[b],
                    z=z[b : b + 1].clone(), imp_log=log_ohm[b : b + 1].clone(), imp_ohm=ohm[b : b + 1].clone(),
                )

    def _step(z: torch.Tensor, opt: torch.optim.Optimizer, seeds: list[int], step: int) -> None:
        opt.zero_grad(set_to_none=True)
        imp_norm, occ = _forward_imp(model, surrogate, z, K, device)
        total, imp_curve, log_ohm, ohm = _compute_loss(
            z, imp_norm, occ, target_row, mu, std, stats, physics_ar_weight=physics_ar_weight,
        )
        total.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_([z], GRAD_CLIP)
        opt.step()
        with torch.no_grad():
            _, exceed, gap = _base_loss(imp_curve, target_row)
            feasible = _is_feasible(imp_curve, target_row)
            _update_best(feasible, _select_score(ohm), z, log_ohm, ohm, step, seeds)
        hist["total"].append(float(total.detach().cpu()))
        hist["exceed"].append(float(exceed.mean().cpu()))
        hist["gap_mean"].append(float(gap.mean().cpu()))
        hist["best_score"].append(best["score"] if best["z"] is not None else float("nan"))

    if OPTIMIZE_BATCH_PER_K:
        B = len(seeds)
        z0 = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            z0.append(_sample_z(K, 1, latent_dim, device, latent_stats, per_K))
        z = torch.nn.Parameter(torch.cat(z0))
        opt = torch.optim.Adam([z], lr=LR)
        seed_list = [int(s) for s in seeds]
        for step in range(NUM_STEPS):
            _step(z, opt, seed_list, step)
        return best, hist, "batch", B

    for seed in seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        z = torch.nn.Parameter(_sample_z(K, 1, latent_dim, device, latent_stats, per_K))
        opt = torch.optim.Adam([z], lr=LR)
        for step in range(NUM_STEPS):
            _step(z, opt, [int(seed)], step)
    return best, hist, "independent", len(seeds)


def _write_report(run_dir: Path) -> Path:
    spec = importlib.util.spec_from_file_location(
        "generate_run_report", Path(__file__).with_name("generate_run_report.py"),
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.write_report(run_dir)


def main() -> None:
    t0 = time.perf_counter()
    root = _project_root()
    os.chdir(root)
    sys.path.insert(0, str(root))
    device = torch.device(DEVICE)

    stats = _load_norm_stats(root / NORMALIZATION_STATS_PATH)
    target_ohm = np.load(root / TARGET_IMPEDANCE_PATH, allow_pickle=False).astype(np.float32).squeeze()
    if target_ohm.shape != (231,):
        raise SystemExit(f"target must be (231,), got {target_ohm.shape}")

    target_ohm_t = torch.tensor(target_ohm, device=device, dtype=DTYPE)
    target_log = torch.log(target_ohm_t.clamp(min=1e-12))
    target_row = (
        (target_log - stats.imp_log_mean) / max(stats.imp_log_std, 1e-12)
        if LOSS_SPACE == "log"
        else target_ohm_t
    )

    ckpt_path = root / CHECKPOINT_PATH
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    physics_ar_weight = float(cfg.get("physics_ar_weight", PHYSICS_AR_WEIGHT))
    latent_dim = int(cfg.get("latent_dim", 42))
    model = MultiInputVAE(
        latent_dim=latent_dim,
        cond_dim=int(cfg.get("cond_dim", 8)),
        heatmap_private_dim=int(cfg.get("heatmap_private_dim", 8)),
        modality_dropout=0.0,
    ).to(device=device, dtype=DTYPE)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict", ckpt)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    surrogate = None
    surr_path = root / SURROGATE_CHECKPOINT_PATH
    if USE_SURROGATE:
        if not surr_path.exists():
            raise SystemExit(
                f"Surrogate enabled but checkpoint missing:\n  {surr_path}\n"
                "Train it first:\n"
                f"  python experiments/{EXPERIMENT}/codes/surrogate_impedance.py\n"
                "Or disable: LATENT_OPT_USE_SURROGATE=0"
            )
        from experiments.exp038_true_multi.codes.surrogate_impedance import load_surrogate

        surrogate = load_surrogate(surr_path, device)
        print(f"Surrogate ON → {surr_path}")
    else:
        print("Surrogate OFF → VAE decode_impedance for optimization loss")

    latent_stats = ckpt.get("latent_stats")
    per_K = ckpt.get("per_K_latent_stats")
    seeds, seed_info = resolve_candidate_seeds()
    print(f"Candidate seeds: mode={seed_info['mode']}  n={len(seeds)}  master={seed_info.get('master_seed')}")

    out_root, run_idx = allocate_run_dir(root, EXPERIMENT)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"Run directory: {out_root}")

    run_cfg = {
        "experiment": EXPERIMENT,
        "run_index": run_idx,
        "run_dir": str(out_root),
        "started": started,
        "checkpoint": str(ckpt_path),
        "normalization_stats_path": str(root / NORMALIZATION_STATS_PATH),
        "imp_log_norm": asdict(stats),
        "device": DEVICE,
        "latent_dim": latent_dim,
        "K_list": K_LIST,
        "seed_info": seed_info,
        "seeds": seeds,
        "optimize_batch_per_k": OPTIMIZE_BATCH_PER_K,
        "num_steps": NUM_STEPS,
        "lr": LR,
        "grad_clip": GRAD_CLIP,
        "loss_space": LOSS_SPACE,
        "objective_mode": OBJECTIVE_MODE,
        "select_metric": SELECT_METRIC,
        "z_l2_weight": Z_L2_WEIGHT,
        "physics_ar_weight": physics_ar_weight,
        "peak_loss_weight": PEAK_LOSS_WEIGHT,
        "peak_index_weight": PEAK_INDEX_WEIGHT,
        "peak_mag_weight": PEAK_MAG_WEIGHT,
        "dual_topk_weight": DUAL_TOPK_WEIGHT,
        "impedance_num_peaks": IMPEDANCE_NUM_PEAKS,
        "surrogate": surrogate is not None,
        "surrogate_checkpoint": str(surr_path) if surrogate is not None else None,
    }
    (out_root / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    (out_root / "candidate_seeds.txt").write_text("\n".join(str(s) for s in seeds) + "\n", encoding="utf-8")

    timing: dict[str, Any] = {"started": started, "run_index": run_idx, "per_k": {}}
    for K in K_LIST:
        if not 0 <= int(K) <= 52:
            raise SystemExit(f"invalid K={K}")
        t_k = time.perf_counter()
        k_dir = out_root / f"K{int(K):02d}"
        k_dir.mkdir(parents=True, exist_ok=True)

        best, hist, mode, n_cand = _optimize_k(
            int(K), model, surrogate, target_row, stats, latent_dim,
            latent_stats, per_K, device, physics_ar_weight, seeds,
        )
        for key, arr in hist.items():
            np.save(k_dir / f"loss_{key}.npy", np.asarray(arr, dtype=np.float32))

        if best["z"] is None:
            payload = {"K": K, "has_solution": False, "n_candidates": n_cand, "mode": mode}
            (k_dir / "no_solution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            timing["per_k"][str(K)] = {"has_solution": False, "seconds": time.perf_counter() - t_k}
            print(f"K={K:02d} | no solution ({n_cand} candidates)")
        else:
            metrics = {
                "has_solution": True,
                "K": K,
                "max_imp_ohm": float(best["imp_ohm"].max().cpu()),
                "best_score": best["score"],
                "best_step": best["step"],
                "winning_seed": best["seed"],
                "mode": mode,
            }
            best["metrics"] = metrics
            _save_solution(k_dir, best, model, int(K), device)
            timing["per_k"][str(K)] = {
                "has_solution": True,
                "seconds": time.perf_counter() - t_k,
                "max_imp_ohm": metrics["max_imp_ohm"],
                "seed": best["seed"],
            }
            print(f"K={K:02d} | seed={best['seed']} step={best['step']} max_ohm={metrics['max_imp_ohm']:.4g}")

    timing["seconds_total"] = time.perf_counter() - t0
    (out_root / "timing_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"\nDone {timing['seconds_total'] / 60:.1f} min → {out_root}")
    print(f"Report → {_write_report(out_root)}")


if __name__ == "__main__":
    main()
