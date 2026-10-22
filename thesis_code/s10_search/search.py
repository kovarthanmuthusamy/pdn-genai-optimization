"""Guided latent search: for each decap budget K, latent vectors whose decoded placement meets a target spectrum.

What: everything is frozen; Adam moves 64 latent vectors per K in parallel for 1,200 steps.
    Start: the 64 best of 1,000 draws from N(mu_K, Sigma_K) of the training placements' latents
        (latent_cov_per_k.json), drawn with a fixed sweep seed and ranked by the claim below.
    Read-out: the placement decoder's probabilities become the hard typed placement CAD would build
        (the K most likely slots, each with its most likely part); the gradient passes straight
        through that hardening (_ste_topk).
    Claim (the guided search): two spectra of the hard placement -- the VAE decoder's at the searched
        latent vector and the placement-only encoder's -- are both pushed under the target, a term
        pulls them together, a point counts only if both pass, and its claim is the larger of the two.
    Loss (in the normalized log space): exceedance of the target minus 0.1 (squared mean + max,
        weight 25) minus the gap below it, + latent prior (0.05), spectrum roughness floor (2),
        Mahalanobis trust region at 2.5 sigma (10), tracking of the target (2), anti-resonance
        prior (0.5), diversity between the 64 vectors (0.35), confidence of the placement (0.5).
    Result per K: every vector's own best passing point (candidates.json, with the placement-alone
        ratio of each), and the lowest claim (best_*), or no_solution.json.
Usage: python -m thesis_code.s10_search.search
    Writes OUTPUT_DIR/s10_search/run_<n>/: run_config.json, candidate_seeds.txt, timing_summary.json
    and K02..K10/ (best_latent.npy, best_occupancy_codes.npy, best_impedance_ohm.npy, best_metrics.json,
    candidates.json, loss_*.npy).
Config keys (environment SEARCH_<name> overrides): TARGET_PATH, MASTER_SEED, SWEEP_SEED_BASE,
    NUM_STEPS, NUM_SEEDS, K_MAX.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import need, out_dir
from thesis_code.common.placement import LABELS_ORDERED
from thesis_code.s09_inference.inference import VAEInference
from thesis_code.s09_inference.placement_encoder import freeze, load, margin

# =============================================================================
# CONFIGURATION — edit these before running (or set SEARCH_<name> in the environment)
# =============================================================================
TARGET_PATH = os.getenv("SEARCH_TARGET_PATH", str(INPUTS / "target_impedance_0.8.npy"))
MASTER_SEED = int(os.getenv("SEARCH_MASTER_SEED")) if os.getenv("SEARCH_MASTER_SEED") else None
SWEEP_SEED_BASE = int(os.getenv("SEARCH_SWEEP_SEED_BASE", "90000"))   # the starting draws; MASTER_SEED only names the vectors
NUM_STEPS = int(os.getenv("SEARCH_NUM_STEPS", "1200"))
NUM_SEEDS = int(os.getenv("SEARCH_NUM_SEEDS", "64"))                  # parallel latent vectors per K
K_MAX = int(os.getenv("SEARCH_K_MAX", "10"))                          # budgets 2..K_MAX
# =============================================================================

K_LIST = list(range(2, K_MAX + 1))
SWEEP_SEED_N = 1000             # draws per K from which the starting vectors are taken
LR = 5e-2
GRAD_CLIP = 5.0
INIT_SHARED_TEMP = 0.8          # the draws use 0.8 x the latent spread
Z_L2_WEIGHT = 0.05
SHAPE_REG_WEIGHT = 2.0
SHAPE_MIN_STD_DLOG = 0.12
SHAPE_TRACK_WEIGHT = 2.0
POSTERIOR_BOUNDARY_WEIGHT = 10.0
POSTERIOR_SIGMA_LIMIT = 2.5     # trust region radius in units of the latent spread at that K
PHYSICS_AR_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.35
OCC_CONFIDENCE_WEIGHT = 0.5
BOUNDARY_MARGIN = 0.1           # in the normalized log space
GAP_REWARD_WEIGHT = 1.0
EXCEED_WEIGHT = 25.0
EXCEED_POWER = 2.0
CLAIM_AGREE_WEIGHT = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
HIST_KEYS = ("total", "exceed", "gap_mean", "best_score")
_JUDGE: dict[str, Any] = {}     # the placement-only encoder; set in main()
_COV: dict = {}                 # latent_cov_per_k.json; set in main()


@dataclass(frozen=True)
class NormStats:
    imp_log_mean: float
    imp_log_std: float


def allocate_run_dir() -> tuple[Path, int]:
    """Next run folder OUTPUT_DIR/s10_search/run_<n> (safe when two searches start together)."""
    base = out_dir("s10_search")
    idx = [int(p.name[4:]) for p in base.glob("run_*") if p.is_dir() and p.name[4:].isdigit()]
    n = max(idx) + 1 if idx else 0
    while True:
        try:
            (base / f"run_{n}").mkdir(exist_ok=False)
            return base / f"run_{n}", n
        except FileExistsError:
            n += 1


def resolve_candidate_seeds() -> tuple[list[int], dict[str, Any]]:
    """Integer labels of the parallel vectors (from MASTER_SEED, or a random master seed)."""
    master = MASTER_SEED if MASTER_SEED is not None else int(np.random.randint(0, 2**31 - 1))
    seeds = [int(x) for x in np.random.default_rng(master).integers(0, 2**31 - 1, size=NUM_SEEDS)]
    return seeds, {"mode": "random", "master_seed": master, "num_seeds": NUM_SEEDS, "seeds": seeds}


def _curves_from_norm(imp_norm: torch.Tensor, stats: NormStats) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, 1, 231) normalized log-spectrum -> (normalized log, log ohm, ohm), each (B, 231)."""
    log_norm = imp_norm[:, 0]
    log_ohm = log_norm * stats.imp_log_std + stats.imp_log_mean
    return log_norm, log_ohm, torch.exp(log_ohm).clamp(min=1e-12)


def _k_prior(K: int, latent_dim: int, device: torch.device, per_K: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and spread of the latents at budget K (saved in the checkpoint)."""
    lat = per_K[str(K)]["latent"]
    mu = torch.tensor(lat["mu_mean_per_dim"], dtype=DTYPE, device=device).view(-1)[:latent_dim]
    std = torch.tensor(lat["agg_std_per_dim"], dtype=DTYPE, device=device).view(-1)[:latent_dim]
    return mu, std.clamp(min=1e-6)


def _cov_for(K: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    e = _COV[str(K)]
    mu = torch.tensor(e["mu"], dtype=DTYPE, device=device)
    cov = torch.tensor(e["cov"], dtype=DTYPE, device=device)
    return mu, cov + 1e-5 * torch.eye(cov.shape[0], dtype=DTYPE, device=device)


def _sample_z(K: int, n: int, device: torch.device) -> torch.Tensor:
    mu_c, cov = _cov_for(K, device)
    L = torch.linalg.cholesky(cov)
    eps = torch.randn(n, mu_c.shape[0], device=device, dtype=DTYPE)
    return (eps @ L.T) * INIT_SHARED_TEMP + mu_c


def _decode_occ(model: torch.nn.Module, z: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """(B, 52, T) part probabilities of the placement z decodes to."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    K_t = torch.full((z.shape[0],), K, dtype=torch.long, device=device)
    dec = torch.cat([z[:, : model.shared_latent_dim], model.k_embedding(K_t)], dim=1)
    return torch.sigmoid(model.occupancy_decoder(dec))


def _topk_occ(occ_prob: torch.Tensor, K: int) -> torch.Tensor:
    """Hard typed placement: the K slots whose best part scores highest, each one-hot in that part."""
    score = occ_prob.max(dim=-1).values
    keep = torch.zeros_like(score)
    if K > 0:
        keep.scatter_(-1, score.topk(min(K, score.shape[-1]), dim=-1).indices, 1.0)
    part = F.one_hot(occ_prob.argmax(dim=-1), occ_prob.shape[-1]).to(occ_prob.dtype)
    return part * keep.unsqueeze(-1)


def _ste_topk(occ_prob: torch.Tensor, K: int) -> torch.Tensor:
    """Hard top-K placement in the forward pass, identity gradient to ``occ_prob``."""
    hard = _topk_occ(occ_prob, K)
    return hard + occ_prob - occ_prob.detach()


def _judge_imp(occ_hard: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """Normalized log-spectrum (B, 1, 231) of the placement-only encoder for a hard typed placement.
    Differentiable in ``occ_hard``: the straight-through read-out carries the gradient on to z."""
    K_t = torch.full((occ_hard.shape[0],), K, dtype=torch.long, device=device)
    return _JUDGE["net"]._impedance_norm(occ_hard, K_t).unsqueeze(1)


def _decoder_imp(model: torch.nn.Module, z: torch.Tensor, K: int, device: torch.device):
    """The VAE decoder's spectrum (B, 1, 231) at z with the hard placement z decodes to as context,
    and the placement probabilities."""
    K_t = torch.full((z.shape[0],), K, dtype=torch.long, device=device)
    occ = _decode_occ(model, z, K, device)
    occ_ctx = model.occupancy_encoder(_ste_topk(occ, K))
    dec_imp = torch.cat([model._layout_latent_slice(z), model.k_embedding(K_t), occ_ctx], dim=1)
    return model.impedance_decoder(dec_imp).unsqueeze(1), occ


def _is_feasible(imp_curve: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boundary = target - BOUNDARY_MARGIN
    if imp_curve.dim() == 1:
        imp_curve = imp_curve.unsqueeze(0)
    if boundary.dim() == 1:
        boundary = boundary.unsqueeze(0)
    return (imp_curve <= boundary.expand(imp_curve.shape[0], -1)).all(dim=1)


def _sweep_seed_z(model: torch.nn.Module, K: int, n_seeds: int, device: torch.device,
                  target_row: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """The ``n_seeds`` best of SWEEP_SEED_N draws: passing first (by the guided claim), then lowest peak."""
    torch.manual_seed(SWEEP_SEED_BASE + K)
    cand = _sample_z(K, SWEEP_SEED_N, device)
    peaks, feas, zs = [], [], []
    with torch.no_grad():
        for i in range(0, len(cand), 512):
            z = cand[i:i + 512]
            K_t = torch.full((len(z),), K, dtype=torch.long, device=device)
            occ_in = _ste_topk(_decode_occ(model, z, K, device), K)
            dec = torch.cat([model._layout_latent_slice(z), model.k_embedding(K_t), model.occupancy_encoder(occ_in)], dim=1)
            log_norm, _, ohm = _curves_from_norm(model.impedance_decoder(dec).unsqueeze(1), stats)
            ok_b = _is_feasible(log_norm, target_row)
            n_log, _, n_ohm = _curves_from_norm(_judge_imp(occ_in, K, device), stats)
            ohm, ok_b = torch.maximum(ohm, n_ohm), ok_b & _is_feasible(n_log, target_row)
            peaks.append(ohm.max(dim=-1).values)
            feas.append(ok_b)
            zs.append(z)
    peak = torch.cat(peaks).flatten()
    ok = torch.cat(feas).flatten().bool()
    score = torch.where(ok, peak, peak + 1e6)
    print(f"  K={K:02d} sweep: {int(ok.sum())}/{SWEEP_SEED_N} feasible, "
          f"best peak {peak[ok].min().item() if ok.any() else float('nan'):.3f} ohm", flush=True)
    return torch.cat(zs)[torch.argsort(score)[:n_seeds]].detach().clone()


def _anti_resonance_loss(imp: torch.Tensor) -> torch.Tensor:
    """Penalty on peaks that are not preceded by a dip (a peak needs an anti-resonance before it)."""
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
    viol = (imp_curve - boundary).clamp(min=0.0).pow(EXCEED_POWER)
    below = (boundary - imp_curve).clamp(min=0.0)
    exceed = viol.mean(dim=1) + viol.amax(dim=1)
    gap = below.mean(dim=1)
    return -GAP_REWARD_WEIGHT * gap + EXCEED_WEIGHT * exceed, exceed, gap


def _posterior_boundary_maha(z: torch.Tensor, K: int) -> torch.Tensor:
    """Trust region: Mahalanobis radius per dimension under the latents' covariance at K."""
    mu_c, cov = _cov_for(K, z.device)
    d = (z - mu_c).unsqueeze(-1)
    sol = torch.cholesky_solve(d, torch.linalg.cholesky(cov))
    r = torch.sqrt((d * sol).sum((1, 2)).clamp(min=0) / z.shape[-1])
    return (r - POSTERIOR_SIGMA_LIMIT).clamp(min=0.0).pow(2)


def _diversity_loss(occ: torch.Tensor) -> torch.Tensor:
    B = occ.shape[0]
    if B <= 1:
        return occ.new_zeros(())
    x = occ.flatten(1)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = x @ x.t()
    return (sim.sum() - sim.diagonal().sum()) / (B * (B - 1))


def _compute_loss(z: torch.Tensor, K: int, imp_norm: torch.Tensor, occ_prob: torch.Tensor, target_row: torch.Tensor,
                  prior_mu: torch.Tensor, prior_std: torch.Tensor, stats: NormStats):
    """Returns total (scalar), imp_curve (B, 231) normalized log, log ohm, ohm."""
    log_norm, log_ohm, ohm = _curves_from_norm(imp_norm, stats)
    target = target_row.expand(log_norm.shape[0], -1)
    base, _, _ = _base_loss(log_norm, target)
    z_l2 = ((z - prior_mu) / prior_std).pow(2).mean(dim=-1)
    dlog = log_norm[:, 1:] - log_norm[:, :-1]
    shape = (SHAPE_MIN_STD_DLOG - dlog.std(dim=1)).clamp(min=0.0).pow(2)
    post = _posterior_boundary_maha(z, K)
    track = F.mse_loss(log_norm, target, reduction="none").mean(dim=1)
    total = (base.mean() + Z_L2_WEIGHT * z_l2.mean() + SHAPE_REG_WEIGHT * shape.mean()
             + POSTERIOR_BOUNDARY_WEIGHT * post.mean() + SHAPE_TRACK_WEIGHT * track.mean()
             + PHYSICS_AR_WEIGHT * _anti_resonance_loss(imp_norm))
    if z.shape[0] > 1:
        total = total + DIVERSITY_WEIGHT * _diversity_loss(occ_prob) + OCC_CONFIDENCE_WEIGHT * (occ_prob * (1 - occ_prob)).mean()
    return total, log_norm, log_ohm, ohm


def _type_codes(hard: torch.Tensor) -> np.ndarray:
    """Hard (B, 52, T) placement -> (B, 52) codes: 0 = empty, 1..T = part."""
    return ((hard.argmax(-1) + 1) * (hard.sum(-1) > 0)).cpu().numpy().astype(np.int64)


def _codes_of(model, z, K, device) -> np.ndarray:
    with torch.no_grad():
        return _type_codes(_topk_occ(_decode_occ(model, z, K, device), K))[0]


def _save_solution(k_dir: Path, best: dict[str, Any], model: torch.nn.Module, K: int, device: torch.device) -> None:
    z = best["z"]
    np.save(k_dir / "best_latent.npy", z.cpu().numpy())
    np.save(k_dir / "best_impedance_log.npy", best["imp_log"].cpu().numpy())
    np.save(k_dir / "best_impedance_ohm.npy", best["imp_ohm"].cpu().numpy())
    with torch.no_grad():
        occ = _decode_occ(model, z, K, device)
    np.save(k_dir / "best_occupancy_prob.npy", occ.cpu().numpy())
    codes = _type_codes(_topk_occ(occ, K))
    np.save(k_dir / "best_occupancy_codes.npy", codes.astype(np.int8))
    topk_np = (codes > 0).astype(np.int8)
    np.save(k_dir / "best_occupancy_topk.npy", topk_np)
    metrics = dict(best["metrics"])
    metrics["active_components"] = [LABELS_ORDERED[i] for i in range(52) if int(topk_np.reshape(-1)[i])]
    metrics["n_active"] = len(metrics["active_components"])
    metrics["type_codes"] = codes.reshape(-1).tolist()
    (k_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _save_candidates(k_dir: Path, best: dict[str, Any], model: torch.nn.Module, K: int, device: torch.device,
                     target_ohm: np.ndarray, recheck: dict, pass_margin: float) -> None:
    """Every vector's own best passing placement, de-duplicated, with its placement-alone ratio."""
    rows, seen = [], set()
    for seed, ps in sorted(best["per_seed"].items(), key=lambda kv: kv[1]["score"]):
        codes = _codes_of(model, ps["z"], K, device)
        key = tuple(codes.tolist())
        if key in seen:
            continue
        seen.add(key)
        ohm = ps["imp_ohm"].cpu().numpy().reshape(-1)
        row = {"seed": int(seed), "step": int(ps["step"]), "codes": codes.tolist(),
               "claimed_max_ohm": float(ohm.max()),
               "claimed_worst_ratio": float((ohm / np.maximum(target_ohm, 1e-12)).max())}
        if key in recheck:
            row["recheck_worst_ratio"] = recheck[key]
            row["recheck_pass"] = bool(recheck[key] <= 1.0 - pass_margin)
        rows.append(row)
    (k_dir / "candidates.json").write_text(json.dumps(
        {"K": K, "n_seeds_feasible": len(best["per_seed"]), "n_distinct_layouts": len(rows), "rows": rows},
        indent=2), encoding="utf-8")


def _recheck(best: dict[str, Any], model: torch.nn.Module, K: int, device: torch.device,
             target_ohm: np.ndarray, checker) -> dict:
    """{placement codes: worst-case ratio of the placement-only encoder} of every vector's claim."""
    items = [_codes_of(model, ps["z"], K, device) for ps in best["per_seed"].values()]
    if not items:
        return {}
    ratios = (checker.predict_ohm(np.stack(items)) / np.maximum(target_ohm, 1e-12)).max(1)
    return {tuple(c.tolist()): float(r) for c, r in zip(items, ratios)}


def _optimize_k(K: int, model: torch.nn.Module, target_row: torch.Tensor, stats: NormStats, latent_dim: int,
                per_K: dict, device: torch.device, seeds: list[int]) -> tuple[dict[str, Any], dict[str, list]]:
    """Returns the best point (with every vector's own best passing point) and the loss history."""
    mu, std = _k_prior(K, latent_dim, device, per_K)
    hist: dict[str, list] = {k: [] for k in HIST_KEYS}
    best: dict[str, Any] = {"score": float("inf"), "step": -1, "seed": -1, "z": None, "per_seed": {}}

    def _update_best(feasible, scores, z, log_ohm, ohm, step):
        for b in range(feasible.shape[0]):
            if not feasible[b].item():
                continue
            s = float(scores[b].item())
            ps = best["per_seed"].get(seeds[b])
            if ps is None or s < ps["score"]:
                best["per_seed"][seeds[b]] = {"score": s, "step": step, "z": z[b:b + 1].clone(),
                                              "imp_log": log_ohm[b:b + 1].clone(), "imp_ohm": ohm[b:b + 1].clone()}
            if s < best["score"]:
                best.update(score=s, step=step, seed=seeds[b], z=z[b:b + 1].clone(),
                            imp_log=log_ohm[b:b + 1].clone(), imp_ohm=ohm[b:b + 1].clone())

    def _step(z, opt, step):
        opt.zero_grad(set_to_none=True)
        imp_norm, occ = _decoder_imp(model, z, K, device)
        total, imp_curve, log_ohm, ohm = _compute_loss(z, K, imp_norm, occ, target_row, mu, std, stats)
        n_curve, n_log_ohm, n_ohm = _curves_from_norm(_judge_imp(_ste_topk(occ, K), K, device), stats)
        total = total + _base_loss(n_curve, target_row)[0].mean() + CLAIM_AGREE_WEIGHT * F.mse_loss(imp_curve, n_curve)
        total.backward()
        torch.nn.utils.clip_grad_norm_([z], GRAD_CLIP)
        z_eval = z.detach().clone()          # the point that was scored, before the step moves it
        opt.step()
        with torch.no_grad():
            _, exceed, gap = _base_loss(imp_curve, target_row)
            feasible = _is_feasible(imp_curve, target_row) & _is_feasible(n_curve, target_row)
            log_ohm, ohm = torch.maximum(log_ohm, n_log_ohm), torch.maximum(ohm, n_ohm)
            _update_best(feasible, ohm.max(dim=1).values, z_eval, log_ohm, ohm, step)
        hist["total"].append(float(total.detach().cpu()))
        hist["exceed"].append(float(exceed.mean().cpu()))
        hist["gap_mean"].append(float(gap.mean().cpu()))
        hist["best_score"].append(best["score"] if best["z"] is not None else float("nan"))

    if len(seeds) > SWEEP_SEED_N:
        raise SystemExit(f"NUM_SEEDS ({len(seeds)}) must not exceed the {SWEEP_SEED_N} starting draws")
    z = torch.nn.Parameter(_sweep_seed_z(model, K, len(seeds), device, target_row, stats))
    opt = torch.optim.Adam([z], lr=LR)
    for step in range(NUM_STEPS):
        _step(z, opt, step)
    return best, hist


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device(DEVICE)
    weights = need(Path(settings.WEIGHTS_DIR), "WEIGHTS_DIR")
    cov_file = need(weights / "latent_cov_per_k.json", "WEIGHTS_DIR (latent_cov_per_k.json)")
    _COV.clear()
    _COV.update(json.loads(cov_file.read_text()))
    missing = [K for K in K_LIST if str(K) not in _COV]
    if missing:
        raise SystemExit(f"{cov_file}: no latent covariance for K={missing} -- run thesis_code.s10_search.latent_stats")
    target_ohm = np.load(TARGET_PATH, allow_pickle=False).astype(np.float32).squeeze()
    if target_ohm.shape != (231,):
        raise SystemExit(f"target must be (231,), got {target_ohm.shape}")
    pass_margin = margin(Path(TARGET_PATH))

    vae = VAEInference(weights / settings.CHECKPOINT, device=device)
    stats = NormStats(float(vae.imp_log_mean), float(vae.imp_log_std))
    target_row = (torch.log(torch.tensor(target_ohm, device=device, dtype=DTYPE).clamp(min=1e-12))
                  - stats.imp_log_mean) / max(stats.imp_log_std, 1e-12)
    model = vae.model.to(device=device, dtype=DTYPE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    checker = load(device=device)
    freeze(checker)
    _JUDGE["net"] = checker
    per_K = vae.per_K_latent_stats
    latent_dim = vae.latent_dim

    seeds, seed_info = resolve_candidate_seeds()
    out_root, run_idx = allocate_run_dir()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"Run directory: {out_root}")
    run_cfg = {"run_index": run_idx, "run_dir": str(out_root), "started": started,
               "checkpoint": str(weights / settings.CHECKPOINT), "imp_log_norm": asdict(stats), "device": DEVICE,
               "latent_dim": latent_dim, "n_decap_types": 3, "K_list": K_LIST, "seed_info": seed_info, "seeds": seeds,
               "num_steps": NUM_STEPS, "lr": LR, "grad_clip": GRAD_CLIP, "recheck_head": settings.HEAD,
               "recheck_margin": pass_margin, "recheck_filter": False, "claim_mode": "both",
               "sigma_limit": POSTERIOR_SIGMA_LIMIT, "claim_agree_weight": CLAIM_AGREE_WEIGHT,
               "sweep_seed_base": SWEEP_SEED_BASE, "sweep_seed_n": SWEEP_SEED_N, "latent_cov_k": sorted(map(int, _COV)),
               "target_impedance_path": str(TARGET_PATH)}
    (out_root / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    (out_root / "candidate_seeds.txt").write_text("\n".join(str(s) for s in seeds) + "\n", encoding="utf-8")

    timing: dict[str, Any] = {"started": started, "run_index": run_idx, "per_k": {}}
    for K in K_LIST:
        t_k = time.perf_counter()
        k_dir = out_root / f"K{K:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        best, hist = _optimize_k(K, model, target_row, stats, latent_dim, per_K, device, seeds)
        for key, arr in hist.items():
            np.save(k_dir / f"loss_{key}.npy", np.asarray(arr, dtype=np.float32))
        recheck = _recheck(best, model, K, device, target_ohm, checker) if best["z"] is not None else {}
        if best["z"] is not None:
            n_kept = sum(r <= 1.0 - pass_margin for r in recheck.values())
            print(f"K={K:02d} | placement-alone check: {n_kept}/{len(recheck)} claimed placements pass")
            best["recheck_worst_ratio"] = recheck.get(tuple(_codes_of(model, best["z"], K, device).tolist()))
        if best["per_seed"]:
            _save_candidates(k_dir, best, model, K, device, target_ohm, recheck, pass_margin)
        if best["z"] is None:
            (k_dir / "no_solution.json").write_text(json.dumps(
                {"K": K, "has_solution": False, "n_candidates": len(seeds), "mode": "batch"}, indent=2), encoding="utf-8")
            timing["per_k"][str(K)] = {"has_solution": False, "seconds": time.perf_counter() - t_k}
            print(f"K={K:02d} | no solution ({len(seeds)} candidates)")
            continue
        metrics = {"has_solution": True, "K": K, "max_imp_ohm": float(best["imp_ohm"].max().cpu()),
                   "best_score": best["score"], "best_step": best["step"], "winning_seed": best["seed"], "mode": "batch",
                   "recheck_head": settings.HEAD, "recheck_margin": pass_margin,
                   "recheck_worst_ratio": best["recheck_worst_ratio"], "n_claimed_layouts": len(recheck),
                   "n_recheck_pass": int(sum(r <= 1.0 - pass_margin for r in recheck.values()))}
        best["metrics"] = metrics
        _save_solution(k_dir, best, model, K, device)
        timing["per_k"][str(K)] = {"has_solution": True, "seconds": time.perf_counter() - t_k,
                                   "max_imp_ohm": metrics["max_imp_ohm"], "seed": best["seed"]}
        print(f"K={K:02d} | seed={best['seed']} step={best['step']} max_ohm={metrics['max_imp_ohm']:.4g}")

    timing["seconds_total"] = time.perf_counter() - t0
    (out_root / "timing_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"\nDone {timing['seconds_total'] / 60:.1f} min -> {out_root}")


if __name__ == "__main__":
    main()
