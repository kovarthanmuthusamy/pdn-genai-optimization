"""Training epoch loop for exp045 — encode z + U-Net skips on cross-freq decode."""

from __future__ import annotations

from typing import Any

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp038_true_multi.codes.train_vae_simple import (
    LOSS_KEYS,
    Config,
    PhysicsLoss,
    _amp_dtype,
    _cross_modal,
    _jitter_pi_freq_norm,
    _penalty_scale,
    _phase_weights,
    _prepare_batch,
)

import torch


def _forward_train_batch(
    model,
    hm_enc: torch.Tensor,
    occ: torch.Tensor,
    imp: torch.Tensor,
    K: torch.Tensor,
    pi: torch.Tensor,
    c: Config,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor]:
    """Encode for KL; decode with teacher U-Net skips on encode batches."""
    base = getattr(model, "_orig_mod", model)
    z_post, mu, logvar, expert_stats = base.encode(hm_enc, occ, imp, K, pi)
    encode_skips = base._last_heatmap_skips
    use_layout = (
        train
        and c.layout_train_prob > 0.0
        and bool((torch.rand((), device=pi.device) < c.layout_train_prob).item())
    )
    if use_layout:
        z = base.encode_layout_latent(occ, imp, K, pi)
        skips = None
    else:
        z = z_post
        skips = encode_skips if getattr(c, "encode_native_teacher_skips", True) else None
    pi_dec = _jitter_pi_freq_norm(pi, c, train=train)
    rh, ro, ri = base.decode(z, K, pi_dec, occupancy=occ, heatmap_skips=skips)
    return rh, ro, ri, mu, logvar, expert_stats, z


def _cross_freq_decode(
    c: Config,
    base,
    *,
    hm_enc,
    occ,
    imp,
    K,
    pi,
    z_decode,
    pi_alt,
    hm_alt,
    ps: float,
    dynrange_weight: float | None,
):
    """Encode z + teacher skips for cross-freq heatmap loss."""
    if getattr(c, "cross_freq_use_encode_z", True):
        z_cf, _, _, _ = base.encode(hm_enc, occ, imp, K, pi)
        skips = base._last_heatmap_skips
    elif c.cross_freq_layout_z_only:
        z_cf = base.encode_layout_latent(occ, imp, K, pi)
        skips = None
    else:
        z_cf = z_decode
        skips = None
    hm_alt_t = hm_alt.to(z_cf.device, non_blocking=c.is_cuda())
    if hm_alt_t.dim() == 3:
        hm_alt_t = hm_alt_t.unsqueeze(1)
    rh, _, _ = base.decode(z_cf, K, pi_alt, occupancy=occ, heatmap_skips=skips)
    return _tr.heatmap_loss(
        rh, hm_alt_t, c, ps, dynrange_weight=dynrange_weight, lite=True,
    ).mean()


def _run_epoch(
    model,
    loader,
    c: Config,
    epoch: int,
    beta: float,
    md: float,
    physics: PhysicsLoss | None,
    pw: tuple[float, float, float] | None,
    imp_log_std: float,
    hm_log_mean: float,
    hm_log_std: float,
    *,
    train: bool,
    optimizer=None,
    scaler=None,
    collect_per_k: bool = False,
) -> dict[str, Any]:
    model.train(train)
    if physics:
        physics.train(train)
    model.modality_dropout = md if train else 0.0
    base = getattr(model, "_orig_mod", model)
    base.modality_dropout_protect_heatmap = (
        train and epoch >= c.modality_dropout_protect_heatmap_epoch
    )

    gpu_acc = train and c.is_cuda()
    acc_dev = c.device if gpu_acc else "cpu"
    acc: dict[str, float | torch.Tensor] = (
        {k: torch.zeros((), device=acc_dev) for k in LOSS_KEYS}
        if gpu_acc
        else {k: 0.0 for k in LOSS_KEYS}
    )
    n_batches = 0
    mu_sum = mu_sq = sig_sum = None
    n_mu = 0
    mod_acc: dict[str, list[float]] = {}
    k_buckets: dict[int, dict] = {}

    amp = _amp_dtype(c)
    autocast = torch.autocast(
        device_type="cuda" if c.is_cuda() else "cpu", dtype=amp, enabled=amp is not None,
    )
    cm_every = max(1, c.cross_modal_update_freq)
    ps = _penalty_scale(epoch, c)
    pwts = _phase_weights(epoch, c)
    params = list(model.parameters()) + (list(physics.parameters()) if physics else [])

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for i, batch in enumerate(loader):
            hm, hm_enc, occ, imp, K, pi = _prepare_batch(batch, c)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast:
                if train:
                    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(
                        model, hm_enc, occ, imp, K, pi, c, train=True,
                    )
                else:
                    rh, ro, ri, mu, lv, ex = model(hm_enc, occ, imp, K, pi)
                    z = mu
                losses = _tr.vae_loss(
                    rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
                    epoch=epoch, K=K, physics=physics, pw=pw, pi_freq=pi,
                    imp_log_std=imp_log_std, ps=ps, apply_k=train,
                    weight_overrides=pwts,
                )
                if c.heatmap_phys_p99_weight > 0:
                    phys = _tr.heatmap_phys_amplitude_loss(rh, hm, hm_log_mean, hm_log_std, c)
                    losses["heatmap_phys_p99_loss"] = phys
                    losses["total_loss"] = losses["total_loss"] + c.heatmap_phys_p99_weight * phys
                if train and c.cross_modal_weight > 0 and i % cm_every == 0:
                    losses["total_loss"] = losses["total_loss"] + c.cross_modal_weight * _cross_modal(
                        model, hm_enc, occ, imp, hm, K, pi, c, imp_log_std, ps,
                    )
                cf_w = pwts["cross_freq_weight"]
                if (
                    train
                    and cf_w > 0
                    and epoch >= c.cross_freq_start_epoch
                    and "heatmap_norm_alt" in batch
                ):
                    pi_alt = batch["PI_freq_alt"].to(c.device, non_blocking=c.is_cuda())
                    hm_alt = batch["heatmap_norm_alt"].to(c.device, non_blocking=c.is_cuda())
                    base_cf = getattr(model, "_orig_mod", model)
                    cf = _cross_freq_decode(
                        c, base_cf,
                        hm_enc=hm_enc, occ=occ, imp=imp, K=K, pi=pi, z_decode=z,
                        pi_alt=pi_alt, hm_alt=hm_alt, ps=ps,
                        dynrange_weight=pwts["heatmap_dynrange_weight"],
                    )
                    losses["cross_freq_heatmap_loss"] = cf
                    losses["total_loss"] = losses["total_loss"] + cf_w * cf
                if train and c.kan_spline_l1_weight > 0 and hasattr(model, "spline_l1"):
                    spl = getattr(model, "_orig_mod", model).spline_l1()
                    losses["total_loss"] = losses["total_loss"] + c.kan_spline_l1_weight * spl
                    losses["spline_l1_loss"] = spl

            if train:
                if scaler and scaler.is_enabled():
                    scaler.scale(losses["total_loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
            else:
                dim = mu.shape[1]
                if mu_sum is None:
                    mu_sum = torch.zeros(dim)
                    mu_sq = torch.zeros(dim)
                    sig_sum = torch.zeros(dim)
                mc, sc = mu.cpu(), (0.5 * lv).exp().cpu()
                mu_sum += mc.sum(0)
                mu_sq += mc.pow(2).sum(0)
                sig_sum += sc.sum(0)
                n_mu += mc.shape[0]
                for name, (em, elv) in ex.items():
                    if name not in mod_acc:
                        mod_acc[name] = [0.0, 0.0, 0.0, 0.0]
                    es = (elv / 2).exp()
                    mod_acc[name][0] += em.mean().item()
                    mod_acc[name][1] += em.std().item()
                    mod_acc[name][2] += es.mean().item()
                    mod_acc[name][3] += es.std().item()
                if collect_per_k:
                    for kv in K.cpu().unique().tolist():
                        kv = int(kv)
                        mask = K.cpu() == kv
                        bk = k_buckets.setdefault(
                            kv,
                            {"mu": torch.zeros(dim), "sq": torch.zeros(dim), "sig": torch.zeros(dim), "n": 0},
                        )
                        m = mc[mask]
                        s = sc[mask]
                        bk["mu"] += m.sum(0)
                        bk["sq"] += m.pow(2).sum(0)
                        bk["sig"] += s.sum(0)
                        bk["n"] += mask.sum().item()

            for k in LOSS_KEYS:
                if k in losses:
                    if gpu_acc:
                        acc[k] = acc[k] + losses[k].detach()  # type: ignore[operator]
                    else:
                        acc[k] += losses[k].item()  # type: ignore[operator]
            n_batches += 1

    if gpu_acc:
        out: dict[str, Any] = {k: (v / n_batches).item() for k, v in acc.items()}  # type: ignore[union-attr]
    else:
        out = {k: v / n_batches for k, v in acc.items()}  # type: ignore[operator]
    out["beta"] = beta
    if not train and n_mu > 0:
        mu_pd = mu_sum / n_mu
        std_pd = ((mu_sq / n_mu - mu_pd.pow(2)).clamp(min=0) + (sig_sum / n_mu).pow(2)).sqrt()
        out.update(
            mu_mean=float(mu_pd.mean()),
            mu_std=float(mu_pd.std()),
            mu_min=float(mu_pd.min()),
            mu_max=float(mu_pd.max()),
            logvar_mean=0.0,
            logvar_std=0.0,
            logvar_min=0.0,
            logvar_max=0.0,
            std_mean=float(std_pd.mean()),
        )
        out["per_dim_stats"] = {
            "latent": {"mu_mean_per_dim": mu_pd.tolist(), "agg_std_per_dim": std_pd.tolist()},
        }
        out["modality_stats"] = {
            n: {
                "mu_mean": a[0] / n_batches,
                "mu_std": a[1] / n_batches,
                "std_mean": a[2] / n_batches,
                "std_std": a[3] / n_batches,
            }
            for n, a in mod_acc.items()
        }
        per_k = {}
        for kv, bk in k_buckets.items():
            if bk["n"] < 2:
                continue
            nk = bk["n"]
            mk = bk["mu"] / nk
            sk = ((bk["sq"] / nk - mk.pow(2)).clamp(min=0) + (bk["sig"] / nk).pow(2)).sqrt()
            per_k[str(kv)] = {
                "latent": {"mu_mean_per_dim": mk.tolist(), "agg_std_per_dim": sk.tolist()},
            }
        out["per_K_latent_stats"] = per_k
    else:
        out.setdefault("modality_stats", {})
        out.setdefault("per_K_latent_stats", {})
    return out
