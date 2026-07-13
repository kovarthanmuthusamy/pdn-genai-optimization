"""Training epoch loop for exp053 — pixel peak blob losses + layout path."""

from __future__ import annotations

from typing import Any

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp038_true_multi.codes.train_vae_simple import (
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

from experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses import heatmap_peak_blob_phys_loss
from experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight import (
    mhz_loss_weight,
    weighted_mean,
)
from experiments.exp053_peak_log1p_losses.codes.training_guard import (
    clamp_logvar,
    prepare_recon_for_loss,
    safe_train_step,
    sanitize_loss_dict,
)
from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE

_LAST_TRAIN_METRICS: dict[str, float] = {}


def _training_use_amp(c: Config, *, train: bool) -> bool:
    """exp053: full fp32 train step avoids AMP backward NaNs in decoder/conditioner."""
    if not train:
        return _amp_dtype(c) is not None
    if bool(getattr(c, "training_force_fp32", True)):
        return False
    return _amp_dtype(c) is not None



def _peak_phys_loss_weight(pi_norm: torch.Tensor, c: Config, base_weight: float) -> float:
    """Boost physical peak-blob loss at high MHz."""
    w = float(base_weight)
    if w <= 0.0:
        return 0.0
    thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
    hf_mult = float(getattr(c, "heatmap_peak_phys_hf_mult", 2.0))
    mhz = _pi_norm_to_mhz(pi_norm).mean().item()
    if mhz >= thr:
        w *= hf_mult
    return w


def _pi_norm_to_mhz(pi_norm: torch.Tensor) -> torch.Tensor:
    """Batch PI norm → MHz."""
    log10_hz = pi_norm.float() * _LOG10_RANGE + _LOG10_MIN
    return torch.pow(10.0, log10_hz) / 1e6


def _high_freq_mult(pi_norm: torch.Tensor, c: Config) -> torch.Tensor:
    thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
    mult = float(getattr(c, "layout_high_freq_loss_mult", 2.0))
    mhz = _pi_norm_to_mhz(pi_norm)
    return torch.where(mhz >= thr, mhz.new_tensor(mult), mhz.new_tensor(1.0))




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
    cf = _tr.heatmap_loss(rh, hm_alt_t, c, ps, lite=True)
    if bool(getattr(c, "heatmap_mhz_loss_weight_enabled", False)):
        weight = mhz_loss_weight(pi_alt, c)
    else:
        thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
        cf_mult = float(getattr(c, "cross_freq_high_freq_loss_mult", 2.0))
        mhz = _pi_norm_to_mhz(pi_alt)
        weight = torch.where(mhz >= thr, mhz.new_tensor(cf_mult), mhz.new_tensor(1.0))
    return weighted_mean(cf, weight)


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
        {k: torch.zeros((), device=acc_dev) for k in _tr.LOSS_KEYS}
        if gpu_acc
        else {k: 0.0 for k in _tr.LOSS_KEYS}
    )
    n_batches = 0
    nan_skips = 0
    distill_acc: float | torch.Tensor = torch.zeros((), device=acc_dev) if gpu_acc else 0.0
    out_distill_acc: float | torch.Tensor = torch.zeros((), device=acc_dev) if gpu_acc else 0.0
    mu_sum = mu_sq = sig_sum = None
    n_mu = 0
    mod_acc: dict[str, list[float]] = {}
    k_buckets: dict[int, dict] = {}

    use_amp = _training_use_amp(c, train=train)
    amp_dtype = _amp_dtype(c)
    autocast = torch.autocast(
        device_type="cuda" if c.is_cuda() else "cpu", dtype=amp_dtype, enabled=use_amp,
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
            distill = None
            out_distill = None
            used_layout = False
            if train:
                # Full fp32 forward + loss + backward (no AMP on train path).
                rh, ro, ri, mu, lv, ex, z, distill, out_distill, used_layout = _forward_train_batch(
                    model, hm_enc, occ, imp, K, pi, c, train=True,
                )
                rh = prepare_recon_for_loss(rh, c)
                ro = ro.float()
                ri = ri.float()
                hm_f = hm.float()
                occ_f = occ.float()
                imp_f = imp.float()
                mu = mu.float()
                lv = clamp_logvar(lv, c)
                pi_f = pi.float()
            else:
                with autocast:
                    rh, ro, ri, mu, lv, ex = model(hm_enc, occ, imp, K, pi)
                    z = mu
                rh = prepare_recon_for_loss(rh.float(), c)
                ro = ro.float()
                ri = ri.float()
                hm_f = hm.float()
                occ_f = occ.float()
                imp_f = imp.float()
                mu = mu.float()
                lv = clamp_logvar(lv.float(), c)
                pi_f = pi.float()
            losses = _tr.vae_loss(
                rh, ro, ri, hm_f, occ_f, imp_f, mu, lv, beta, c, ex,
                epoch=epoch, K=K, physics=physics, pw=pw, pi_freq=pi_f,
                imp_log_std=imp_log_std, ps=ps, apply_k=train,
                weight_overrides=pwts,
            )
            w_distill = float(getattr(c, "latent_distill_weight", 0.0))
            if train and distill is not None and w_distill > 0.0:
                d = distill.float()
                losses["latent_distill_loss"] = d
                losses["total_loss"] = losses["total_loss"] + w_distill * d
            w_out_distill = float(getattr(c, "output_distill_weight", 0.0))
            if train and out_distill is not None and w_out_distill > 0.0:
                od = out_distill.float()
                losses["output_distill_loss"] = od
                losses["total_loss"] = losses["total_loss"] + w_out_distill * od
            base_phys_w = float(getattr(c, "heatmap_peak_phys_weight", 0.0))
            if base_phys_w > 0.0:
                phys_per = heatmap_peak_blob_phys_loss(
                        rh, hm_f, c, pi_freq=pi_f, reduction="none",
                )
                hm_w = mhz_loss_weight(pi_f, c)
                if bool(getattr(c, "heatmap_mhz_loss_weight_enabled", False)):
                    hf_extra = float(getattr(c, "heatmap_peak_phys_hf_mult", 1.0))
                    if hf_extra != 1.0:
                        thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
                        mhz = _pi_norm_to_mhz(pi_f)
                        hm_w = hm_w * torch.where(
                            mhz >= thr,
                            mhz.new_tensor(hf_extra),
                            mhz.new_tensor(1.0),
                    )
                phys = weighted_mean(phys_per, hm_w)
                losses["heatmap_peak_phys_loss"] = phys
                losses["total_loss"] = losses["total_loss"] + base_phys_w * phys
            if train and c.cross_modal_weight > 0 and i % cm_every == 0:
                losses["total_loss"] = losses["total_loss"] + c.cross_modal_weight * _cross_modal(
                        model, hm_enc, occ, imp, hm_f, K, pi_f, c, imp_log_std, ps,
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
                )
                losses["cross_freq_heatmap_loss"] = cf
                losses["total_loss"] = losses["total_loss"] + cf_w * cf
            if train and c.kan_spline_l1_weight > 0 and hasattr(model, "spline_l1"):
                spl = getattr(model, "_orig_mod", model).spline_l1()
                losses["total_loss"] = losses["total_loss"] + c.kan_spline_l1_weight * spl
                losses["spline_l1_loss"] = spl
            losses = sanitize_loss_dict(losses, c)

            if train:
                skip_bad = bool(getattr(c, "training_skip_nonfinite_batches", True))
                stepped, skip_reason, bad_grad = safe_train_step(
                    losses["total_loss"],
                    params,
                    optimizer,
                    scaler,
                    model=model,
                    physics=physics,
                    grad_clip=float(getattr(c, "training_grad_clip_norm", 1.0)),
                    skip_nonfinite=skip_bad,
                    use_scaler=False,
                    sanitize_grads=bool(getattr(c, "training_sanitize_grads", False)),
                )
                if not stepped:
                    nan_skips += 1
                    if nan_skips <= 3 or nan_skips % 25 == 0:
                        rh_max = float(rh.detach().max()) if torch.is_tensor(rh) else 0.0
                        bad_s = ", ".join(bad_grad[:3]) if bad_grad else "?"
                        print(
                            f"  [nan-guard ep {epoch}] skip batch {i} ({skip_reason}) "
                            f"rh_max={rh_max:.3f} skips={nan_skips} grads={bad_s}",
                            flush=True,
                    )
                    continue
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

            for k in _tr.LOSS_KEYS:
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
    if train and n_batches > 0:
        _LAST_TRAIN_METRICS.clear()
        for _k in ("heatmap_loss", "heatmap_loss_tier_a", "heatmap_peak_log1p_loss", "heatmap_peak_phys_loss"):
            if _k in out:
                _LAST_TRAIN_METRICS[_k] = float(out[_k])
    out["beta"] = beta
    out["nan_skips"] = nan_skips
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
