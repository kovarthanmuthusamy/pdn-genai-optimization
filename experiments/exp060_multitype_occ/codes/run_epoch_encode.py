"""exp055 training epoch — layout path + log1p peak/valley."""

from __future__ import annotations
from typing import Any
import torch
import experiments.exp060_multitype_occ.codes.train_core as _tr
from experiments.exp060_multitype_occ.codes.train_core import (
    Config, _amp_dtype, _cross_modal, _jitter_pi_freq_norm, _penalty_scale, _phase_weights, _prepare_batch,
)
from experiments.exp060_multitype_occ.codes.distributed_train import get_world_size, unwrap_model
from experiments.exp060_multitype_occ.codes.training_guard import clamp_logvar, prepare_recon_for_loss, safe_train_step, sanitize_loss_dict

_LAST_TRAIN_METRICS: dict[str, float] = {}
_HM_KEYS = ("heatmap_loss", "heatmap_loss_tier_a", "heatmap_peak_log1p_loss", "heatmap_valley_log1p_loss")


def _use_amp(c: Config, *, train: bool) -> bool:
    if train:
        return False  # exp055: full fp32 train avoids AMP backward NaNs
    return _amp_dtype(c) is not None


def _latent_distill_loss(mu_s, mu_t, c, base, lv_s=None, lv_t=None):
    tgt = mu_t.detach()
    shared, priv = base.shared_latent_dim, base.heatmap_private_dim
    if priv > 0:
        loss = (mu_s[:, :shared] - tgt[:, :shared]).pow(2).mean() + c.latent_distill_private_mult * (mu_s[:, shared:] - tgt[:, shared:]).pow(2).mean()
    else:
        loss = (mu_s - tgt).pow(2).mean()
    if lv_s is not None and lv_t is not None:
        tlv = lv_t.detach()
        lv_w, pm = c.latent_distill_logvar_weight, float(c.latent_distill_private_mult)
        if priv > 0:
            loss = loss + lv_w * ((lv_s[:, :shared] - tlv[:, :shared]).pow(2).mean() + pm * (lv_s[:, shared:] - tlv[:, shared:]).pow(2).mean())
        else:
            loss = loss + lv_w * (lv_s - tlv).pow(2).mean()
    return loss


def _forward_train_batch(model, hm_enc, occ, imp, K, pi, c, *, train: bool):
    base = unwrap_model(model)
    z_post, mu, logvar, expert_stats = base.encode(hm_enc, occ, imp, K, pi)
    distill = torch.zeros((), device=pi.device)
    out_distill = torch.zeros((), device=pi.device)
    z_lay = lv_lay = None
    if train and c.latent_distill_weight > 0 and base.use_layout_private_head:
        z_lay, mu_lay, lv_lay = base.encode_layout_latent_full(occ, imp, K, pi)
        distill = _latent_distill_loss(mu_lay, mu, c, base, lv_lay, logvar)
    use_layout = train and c.layout_train_prob > 0 and bool(torch.rand((), device=pi.device) < c.layout_train_prob)
    pi_dec = _jitter_pi_freq_norm(pi, c, train=train)
    if use_layout:
        use_occ_only = (
            c.occ_only_encode_prob > 0.0
            and bool(torch.rand((), device=pi.device) < c.occ_only_encode_prob)
        )
        if use_occ_only:
            z = base.encode_occupancy_latent(occ, K, pi)
        else:
            z = z_lay if z_lay is not None else base.encode_layout_latent(occ, imp, K, pi)
        rh, ro, ri = base.decode(z, K, pi_dec, occupancy=occ)
        if c.output_distill_weight > 0:
            rh_t, _, _ = base.decode(z_post.detach(), K, pi_dec, occupancy=occ)
            out_distill = torch.nn.functional.huber_loss(rh, rh_t.detach())
    else:
        z = z_post
        rh, ro, ri = base.decode(z, K, pi_dec, occupancy=occ)
    return rh, ro, ri, mu, logvar, expert_stats, z, distill, out_distill, use_layout, z_post


def _cross_freq_decode(c, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps, z_encode=None):
    mix_p = float(c.cross_freq_layout_mix_prob)
    use_layout_z = mix_p > 0 and bool(torch.rand((), device=pi.device) < mix_p)
    if use_layout_z:
        if getattr(c, "occ_only_encode_prob", 0.0) >= 1.0:
            z_cf = base.encode_occupancy_latent(occ, K, pi)
        else:
            z_cf = base.encode_layout_latent(occ, imp, K, pi)
    elif c.cross_freq_use_encode_z:
        if z_encode is not None:
            z_cf = z_encode
        else:
            z_cf = base.encode(hm_enc, occ, imp, K, pi)[0]
    elif c.cross_freq_layout_z_only:
        if getattr(c, "occ_only_encode_prob", 0.0) >= 1.0:
            z_cf = base.encode_occupancy_latent(occ, K, pi)
        else:
            z_cf = base.encode_layout_latent(occ, imp, K, pi)
    else:
        z_cf = z_decode
    hm_alt_t = hm_alt.to(z_cf.device, non_blocking=c.is_cuda())
    if hm_alt_t.dim() == 3:
        hm_alt_t = hm_alt_t.unsqueeze(1)
    rh, _, _ = base.decode(z_cf, K, pi_alt, occupancy=occ)
    cf = _tr.heatmap_loss(rh, hm_alt_t, c, ps, lite=True)
    cf = cf.mean() if cf.ndim else cf
    return cf, rh, hm_alt_t


def _run_epoch(model, loader, c: Config, epoch, beta, md, physics, pw, imp_log_std, hm_log_mean, hm_log_std,
               *, train: bool, optimizer=None, scaler=None, collect_per_k: bool = False) -> dict[str, Any]:
    del hm_log_mean, hm_log_std, physics, pw
    model.train(train)
    model.modality_dropout = md if train else 0.0
    base = unwrap_model(model)
    base.modality_dropout_protect_heatmap = train and epoch >= c.modality_dropout_protect_heatmap_epoch
    gpu = train and c.is_cuda()
    dev = c.device if gpu else "cpu"
    acc = {k: torch.zeros((), device=dev) if gpu else 0.0 for k in _tr.LOSS_KEYS}
    n_batches = nan_skips = 0
    distill_acc = out_acc = torch.zeros((), device=dev) if gpu else 0.0
    mu_sum = mu_sq = sig_sum = None
    n_mu = 0
    mod_acc, k_buckets = {}, {}

    use_amp = _use_amp(c, train=train)
    amp_dtype = _amp_dtype(c)
    autocast = torch.autocast(device_type="cuda" if c.is_cuda() else "cpu", dtype=amp_dtype, enabled=use_amp)
    ps, pwts = _penalty_scale(epoch, c), _phase_weights(epoch, c)
    params = list(model.parameters())
    ctx = torch.enable_grad() if train else torch.inference_mode()

    with ctx:
        for i, batch in enumerate(loader):
            hm, hm_enc, occ, imp, K, pi = _prepare_batch(batch, c)
            distill = out_distill = None
            z_enc = None
            if train:
                optimizer.zero_grad(set_to_none=True)
                rh, ro, ri, mu, lv, ex, z, distill, out_distill, _, z_enc = _forward_train_batch(
                    model, hm_enc, occ, imp, K, pi, c, train=True)
                rh, ro, ri, hm_f, occ_f, imp_f = prepare_recon_for_loss(rh, c), ro.float(), ri.float(), hm.float(), occ.float(), imp.float()
                mu, lv, pi_f = mu.float(), clamp_logvar(lv, c), pi.float()
            else:
                with autocast:
                    rh, ro, ri, mu, lv, ex = model(hm_enc, occ, imp, K, pi)
                    z = mu
                rh, ro, ri = prepare_recon_for_loss(rh.float(), c), ro.float(), ri.float()
                hm_f, occ_f, imp_f, mu, lv, pi_f = hm.float(), occ.float(), imp.float(), mu.float(), clamp_logvar(lv.float(), c), pi.float()

            losses = _tr.vae_loss(rh, ro, ri, hm_f, occ_f, imp_f, mu, lv, beta, c, ex, epoch=epoch, K=K,
                physics=None, pw=None, pi_freq=pi_f, imp_log_std=imp_log_std, ps=ps, apply_k=train, weight_overrides=pwts)
            if train and distill is not None and c.latent_distill_weight > 0:
                d = distill.float()
                losses["latent_distill_loss"] = d
                losses["total_loss"] = losses["total_loss"] + c.latent_distill_weight * d
            if train and out_distill is not None and c.output_distill_weight > 0:
                od = out_distill.float()
                losses["output_distill_loss"] = od
                losses["total_loss"] = losses["total_loss"] + c.output_distill_weight * od
            if train and c.cross_modal_weight > 0 and i % max(1, c.cross_modal_update_freq) == 0:
                losses["total_loss"] = losses["total_loss"] + c.cross_modal_weight * _cross_modal(
                    model, hm_enc, occ, imp, hm_f, K, pi_f, c, imp_log_std, ps)
            cf_w = pwts["cross_freq_weight"]
            if train and cf_w > 0 and epoch >= c.cross_freq_start_epoch and "heatmap_norm_alt" in batch:
                pi_alt = batch["PI_freq_alt"].to(c.device, non_blocking=c.is_cuda())
                hm_alt = batch["heatmap_norm_alt"].to(c.device, non_blocking=c.is_cuda())
                cf, rh_cf, hm_alt_t = _cross_freq_decode(c, unwrap_model(model), hm_enc=hm_enc, occ=occ, imp=imp, K=K, pi=pi,
                    z_decode=z, pi_alt=pi_alt, hm_alt=hm_alt, ps=ps, z_encode=z_enc)
                losses["cross_freq_heatmap_loss"] = cf
                losses["total_loss"] = losses["total_loss"] + cf_w * cf
                cf_peak_w = float(getattr(c, "cross_freq_peak_loc_weight", 0.0))
                if cf_peak_w > 0:
                    from experiments.exp060_multitype_occ.codes.spatial_metrics import extrema_loc_sharp_loss
                    bg = c.background_value + 0.5
                    fg = (hm_alt_t > bg).float()
                    cf_peak = extrema_loc_sharp_loss(
                        rh_cf, hm_alt_t, fg, bg, side="peak",
                        temperature=float(getattr(c, "heatmap_peak_loc_temperature", 0.035)),
                    ).reshape(-1).mean()
                    losses["cross_freq_peak_loc_loss"] = cf_peak.detach()
                    losses["total_loss"] = losses["total_loss"] + cf_peak_w * cf_peak
            losses = sanitize_loss_dict(losses, c)

            if train:
                # DDP: all ranks must agree to skip BEFORE backward, or one rank
                # allreduces grads while the other does not → NCCL hang.
                finite = torch.tensor(
                    [1.0 if torch.isfinite(losses["total_loss"]).all() else 0.0],
                    device=losses["total_loss"].device,
                )
                if get_world_size() > 1:
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if c.training_skip_nonfinite_batches and finite.item() < 0.5:
                    optimizer.zero_grad(set_to_none=True)
                    nan_skips += 1
                    if nan_skips <= 2 or nan_skips % 50 == 0:
                        rh_max = float(rh.detach().max()) if torch.is_tensor(rh) else 0.0
                        print(f"  [nan-guard ep {epoch}] skip batch {i} (nonfinite_loss_sync) rh_max={rh_max:.3f} skips={nan_skips}", flush=True)
                    continue
                stepped, reason, bad = safe_train_step(losses["total_loss"], params, optimizer, scaler, model=model,
                    grad_clip=float(c.training_grad_clip_norm), skip_nonfinite=False,
                    use_scaler=False, sanitize_grads=c.training_sanitize_grads)
                if not stepped:
                    nan_skips += 1
                    if nan_skips <= 2 or nan_skips % 50 == 0:
                        rh_max = float(rh.detach().max()) if torch.is_tensor(rh) else 0.0
                        print(f"  [nan-guard ep {epoch}] skip batch {i} ({reason}) rh_max={rh_max:.3f} skips={nan_skips} grads={', '.join(bad[:3]) or '?'}", flush=True)
                    continue
            else:
                dim = mu.shape[1]
                if mu_sum is None:
                    mu_sum, mu_sq, sig_sum = torch.zeros(dim), torch.zeros(dim), torch.zeros(dim)
                mc, sc = mu.cpu(), (0.5 * lv).exp().cpu()
                mu_sum += mc.sum(0); mu_sq += mc.pow(2).sum(0); sig_sum += sc.sum(0); n_mu += mc.shape[0]
                for name, val in ex.items():
                    if isinstance(val, tuple) and len(val) == 2:
                        em, elv = val
                        mod_acc.setdefault(name, [0., 0., 0., 0.])
                        es = (elv / 2).exp()
                        mod_acc[name][0] += em.mean().item(); mod_acc[name][1] += em.std().item()
                        mod_acc[name][2] += es.mean().item(); mod_acc[name][3] += es.std().item()
                if collect_per_k:
                    for kv in K.cpu().unique().tolist():
                        kv = int(kv); mask = K.cpu() == kv
                        bk = k_buckets.setdefault(kv, {"mu": torch.zeros(dim), "sq": torch.zeros(dim), "sig": torch.zeros(dim), "n": 0})
                        m, s = mc[mask], sc[mask]
                        bk["mu"] += m.sum(0); bk["sq"] += m.pow(2).sum(0); bk["sig"] += s.sum(0); bk["n"] += mask.sum().item()

            for k in _tr.LOSS_KEYS:
                if k in losses:
                    acc[k] = acc[k] + losses[k].detach() if gpu else acc[k] + losses[k].item()
            if "latent_distill_loss" in losses:
                distill_acc = distill_acc + losses["latent_distill_loss"].detach() if gpu else distill_acc + float(losses["latent_distill_loss"])
            if "output_distill_loss" in losses:
                out_acc = out_acc + losses["output_distill_loss"].detach() if gpu else out_acc + float(losses["output_distill_loss"])
            n_batches += 1

    if gpu:
        out = {k: (v / n_batches).item() for k, v in acc.items()}
        out["latent_distill_loss"] = float(distill_acc / max(n_batches, 1))
        out["output_distill_loss"] = float(out_acc / max(n_batches, 1))
    else:
        out = {k: v / n_batches for k, v in acc.items()}
        out["latent_distill_loss"] = float(distill_acc) / max(n_batches, 1)
        out["output_distill_loss"] = float(out_acc) / max(n_batches, 1)
    if train and n_batches > 0:
        _LAST_TRAIN_METRICS.clear()
        _LAST_TRAIN_METRICS.update({k: float(out[k]) for k in _HM_KEYS if k in out})
    out["beta"], out["nan_skips"] = beta, nan_skips
    if not train and n_mu > 0:
        mu_pd = mu_sum / n_mu
        std_pd = ((mu_sq / n_mu - mu_pd.pow(2)).clamp(min=0) + (sig_sum / n_mu).pow(2)).sqrt()
        out.update(mu_mean=float(mu_pd.mean()), mu_std=float(mu_pd.std()), mu_min=float(mu_pd.min()), mu_max=float(mu_pd.max()),
                   logvar_mean=0., logvar_std=0., logvar_min=0., logvar_max=0., std_mean=float(std_pd.mean()))
        out["per_dim_stats"] = {"latent": {"mu_mean_per_dim": mu_pd.tolist(), "agg_std_per_dim": std_pd.tolist()}}
        out["modality_stats"] = {n: {"mu_mean": a[0]/n_batches, "mu_std": a[1]/n_batches, "std_mean": a[2]/n_batches, "std_std": a[3]/n_batches} for n, a in mod_acc.items()}
        per_k = {}
        for kv, bk in k_buckets.items():
            if bk["n"] < 2: continue
            nk = bk["n"]; mk = bk["mu"]/nk
            sk = ((bk["sq"]/nk - mk.pow(2)).clamp(min=0) + (bk["sig"]/nk).pow(2)).sqrt()
            per_k[str(kv)] = {"latent": {"mu_mean_per_dim": mk.tolist(), "agg_std_per_dim": sk.tolist()}}
        out["per_K_latent_stats"] = per_k
    else:
        out.setdefault("modality_stats", {}); out.setdefault("per_K_latent_stats", {})
    return out
