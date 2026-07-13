"""Training epoch loop for exp046 — encode z + U-Net skips + spatial losses on bounded data."""

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
import torch.nn.functional as F

from experiments.exp047.codes.spatial_metrics import peak_loc_loss, pearson_fg_loss
from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE


def _pi_norm_to_mhz(pi_norm: torch.Tensor) -> torch.Tensor:
    """Batch PI norm → MHz."""
    log10_hz = pi_norm.float() * _LOG10_RANGE + _LOG10_MIN
    return torch.pow(10.0, log10_hz) / 1e6


def _high_freq_mult(pi_norm: torch.Tensor, c: Config) -> torch.Tensor:
    thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
    mult = float(getattr(c, "layout_high_freq_loss_mult", 2.0))
    mhz = _pi_norm_to_mhz(pi_norm)
    return torch.where(mhz >= thr, mhz.new_tensor(mult), mhz.new_tensor(1.0))


def _layout_sharpening_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    pi_norm: torch.Tensor,
    c: Config,
    ps: float,
) -> torch.Tensor:
    """Extra peak/grad/spatial terms on layout batches; boosted at high MHz."""
    from experiments.exp038_true_multi.codes.train_vae_simple import _clip_recon_heatmap_z

    recon = _clip_recon_heatmap_z(recon, c)
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    w = _high_freq_mult(pi_norm, c) * float(getattr(c, "layout_spatial_loss_mult", 1.5))

    huber = F.huber_loss(recon, target, delta=1.0, reduction="none")
    intens = (target - bg).clamp(min=0)
    peak = (huber * fg * intens).sum((1, 2, 3)) / (intens * fg).sum((1, 2, 3)).clamp(min=1e-6)

    fg_dx = fg[:, :, :, 1:] * fg[:, :, :, :-1]
    fg_dy = fg[:, :, 1:, :] * fg[:, :, :-1, :]
    gx = F.huber_loss(
        recon[:, :, :, 1:] - recon[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
        delta=0.5, reduction="none",
    )
    gy = F.huber_loss(
        recon[:, :, 1:, :] - recon[:, :, :-1, :],
        target[:, :, 1:, :] - target[:, :, :-1, :],
        delta=0.5, reduction="none",
    )
    grad = (gx * fg_dx).sum((1, 2, 3)) / fg_dx.sum((1, 2, 3)).clamp(min=1.0)
    grad = grad + (gy * fg_dy).sum((1, 2, 3)) / fg_dy.sum((1, 2, 3)).clamp(min=1.0)

    pear_w = float(getattr(c, "heatmap_pearson_weight", 0.0))
    peak_loc_w = float(getattr(c, "heatmap_peak_loc_weight", 0.0))
    pg_mult = float(getattr(c, "layout_peak_grad_mult", 1.5))
    peak_w = float(getattr(c, "heatmap_peak_weight", 0.0)) * pg_mult
    grad_w = float(getattr(c, "heatmap_grad_weight", 0.0)) * pg_mult

    per = peak_w * peak + grad_w * grad
    if pear_w > 0:
        per = per + pear_w * pearson_fg_loss(recon, target, fg)
    if peak_loc_w > 0:
        per = per + peak_loc_w * peak_loc_loss(recon, target, fg, bg)
    return (w * per).mean()


def _latent_distill_loss(
    mu_student: torch.Tensor,
    mu_teacher: torch.Tensor,
    c: Config,
    base,
    logvar_student: torch.Tensor | None = None,
    logvar_teacher: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pull the layout (student) latent toward the encode (teacher) latent.

    Private (heatmap-structure) dims are weighted more than shared dims — this is
    the split-latent core: layout must learn the spatial info that only the real
    heatmap provides during encode.  Optionally also distills logvar.
    """
    tgt_mu = mu_teacher.detach()
    shared = base.shared_latent_dim
    priv = base.heatmap_private_dim
    if priv > 0:
        d_shared = (mu_student[:, :shared] - tgt_mu[:, :shared]).pow(2).mean()
        d_priv = (mu_student[:, shared:] - tgt_mu[:, shared:]).pow(2).mean()
        pm = float(getattr(c, "latent_distill_private_mult", 3.0))
        loss = d_shared + pm * d_priv
    else:
        loss = (mu_student - tgt_mu).pow(2).mean()

    if logvar_student is not None and logvar_teacher is not None:
        tgt_lv = logvar_teacher.detach()
        lv_w = float(getattr(c, "latent_distill_logvar_weight", 0.3))
        if priv > 0:
            lv_shared = (logvar_student[:, :shared] - tgt_lv[:, :shared]).pow(2).mean()
            lv_priv = (logvar_student[:, shared:] - tgt_lv[:, shared:]).pow(2).mean()
            loss = loss + lv_w * (lv_shared + pm * lv_priv)
        else:
            loss = loss + lv_w * (logvar_student - tgt_lv).pow(2).mean()
    return loss


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Encode for KL; decode with teacher U-Net skips on encode batches.

    Also computes latent distillation and output-level distillation on layout
    batches (student-decoded heatmap vs teacher-decoded heatmap).
    """
    base = getattr(model, "_orig_mod", model)
    z_post, mu, logvar, expert_stats = base.encode(hm_enc, occ, imp, K, pi)
    encode_skips = base._last_heatmap_skips

    distill = torch.zeros((), device=pi.device)
    out_distill = torch.zeros((), device=pi.device)
    w_distill = float(getattr(c, "latent_distill_weight", 0.0))
    use_head = bool(getattr(base, "use_layout_private_head", False))

    z_lay = None
    lv_lay = None
    if train and w_distill > 0.0 and use_head:
        z_lay, mu_lay, lv_lay = base.encode_layout_latent_full(occ, imp, K, pi)
        distill = _latent_distill_loss(mu_lay, mu, c, base, lv_lay, logvar)

    use_layout = (
        train
        and c.layout_train_prob > 0.0
        and bool((torch.rand((), device=pi.device) < c.layout_train_prob).item())
    )
    pi_dec = _jitter_pi_freq_norm(pi, c, train=train)
    if use_layout:
        z = z_lay if z_lay is not None else base.encode_layout_latent(occ, imp, K, pi)
        skips = None
        rh, ro, ri = base.decode(z, K, pi_dec, occupancy=occ, heatmap_skips=skips)
        w_out_distill = float(getattr(c, "output_distill_weight", 0.0))
        if w_out_distill > 0.0:
            rh_teacher, _, _ = base.decode(
                z_post.detach(), K, pi_dec, occupancy=occ, heatmap_skips=encode_skips,
            )
            out_distill = torch.nn.functional.huber_loss(rh, rh_teacher.detach())
    else:
        z = z_post
        skips = encode_skips if getattr(c, "encode_native_teacher_skips", True) else None
        rh, ro, ri = base.decode(z, K, pi_dec, occupancy=occ, heatmap_skips=skips)
    return rh, ro, ri, mu, logvar, expert_stats, z, distill, out_distill, use_layout


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
    force_layout_z: bool = False,
):
    """Encode z + teacher skips for cross-freq heatmap loss."""
    mix_p = float(getattr(c, "cross_freq_layout_mix_prob", 0.0))
    use_layout_z = force_layout_z or (
        mix_p > 0.0
        and bool((torch.rand((), device=pi.device) < mix_p).item())
    )
    if use_layout_z:
        z_cf = base.encode_layout_latent(occ, imp, K, pi)
        skips = None
    elif getattr(c, "cross_freq_use_encode_z", True):
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
    cf = _tr.heatmap_loss(
        rh, hm_alt_t, c, ps, dynrange_weight=dynrange_weight, lite=True,
    )
    thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
    cf_mult = float(getattr(c, "cross_freq_high_freq_loss_mult", 2.0))
    mhz = _pi_norm_to_mhz(pi_alt)
    weight = torch.where(mhz >= thr, mhz.new_tensor(cf_mult), mhz.new_tensor(1.0))
    return (cf * weight).mean()


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
    distill_acc: float | torch.Tensor = torch.zeros((), device=acc_dev) if gpu_acc else 0.0
    out_distill_acc: float | torch.Tensor = torch.zeros((), device=acc_dev) if gpu_acc else 0.0
    layout_sharpen_acc: float | torch.Tensor = torch.zeros((), device=acc_dev) if gpu_acc else 0.0
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
                distill = None
                out_distill = None
                used_layout = False
                if train:
                    rh, ro, ri, mu, lv, ex, z, distill, out_distill, used_layout = _forward_train_batch(
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
                w_distill = float(getattr(c, "latent_distill_weight", 0.0))
                if train and distill is not None and w_distill > 0.0:
                    losses["latent_distill_loss"] = distill
                    losses["total_loss"] = losses["total_loss"] + w_distill * distill
                w_out_distill = float(getattr(c, "output_distill_weight", 0.0))
                if train and out_distill is not None and w_out_distill > 0.0:
                    losses["output_distill_loss"] = out_distill
                    losses["total_loss"] = losses["total_loss"] + w_out_distill * out_distill
                layout_sharpen_w = float(getattr(c, "layout_sharpening_weight", 1.0))
                if train and used_layout and layout_sharpen_w > 0.0:
                    pi_for_ls = _jitter_pi_freq_norm(pi, c, train=True)
                    ls = _layout_sharpening_loss(rh, hm, pi_for_ls, c, ps)
                    losses["layout_sharpen_loss"] = ls
                    losses["total_loss"] = losses["total_loss"] + layout_sharpen_w * ls
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
                for name, val in ex.items():
                    if not isinstance(val, tuple) or len(val) != 2:
                        continue
                    em, elv = val
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
                        acc[k] = acc[k] + losses[k].detach()
                    else:
                        acc[k] += losses[k].item()
            if "latent_distill_loss" in losses:
                if gpu_acc:
                    distill_acc = distill_acc + losses["latent_distill_loss"].detach()
                else:
                    distill_acc += float(losses["latent_distill_loss"])
            if "output_distill_loss" in losses:
                if gpu_acc:
                    out_distill_acc = out_distill_acc + losses["output_distill_loss"].detach()
                else:
                    out_distill_acc += float(losses["output_distill_loss"])
            n_batches += 1

    if gpu_acc:
        out: dict[str, Any] = {k: (v / n_batches).item() for k, v in acc.items()}
        out["latent_distill_loss"] = float(distill_acc / max(n_batches, 1)) if torch.is_tensor(distill_acc) else distill_acc / max(n_batches, 1)
        out["output_distill_loss"] = float(out_distill_acc / max(n_batches, 1)) if torch.is_tensor(out_distill_acc) else out_distill_acc / max(n_batches, 1)
    else:
        out = {k: v / n_batches for k, v in acc.items()}
        out["latent_distill_loss"] = float(distill_acc) / max(n_batches, 1)
        out["output_distill_loss"] = float(out_distill_acc) / max(n_batches, 1)
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
