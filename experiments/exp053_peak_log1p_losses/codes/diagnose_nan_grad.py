"""Isolate which loss term / layer produces non-finite gradients (exp052)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

os.environ.setdefault("VAE_EXPERIMENT_DIR", "experiments/exp053_peak_log1p_losses")

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp038_true_multi.codes.physics_loss import PhysicsLoss
from experiments.exp053_peak_log1p_losses.codes import dataloader_multifreq as dm
from experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses import heatmap_peak_blob_phys_loss
from experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight import mhz_loss_weight, weighted_mean
from experiments.exp053_peak_log1p_losses.codes.run_epoch_encode import (
    _cross_freq_decode,
    _forward_train_batch,
)
from experiments.exp053_peak_log1p_losses.codes.train_vae_simple import (
    Config,
    _patch_training,
    build_vae_model,
)
from experiments.exp053_peak_log1p_losses.codes.training_guard import (
    clamp_logvar,
    find_nonfinite_grad_names,
    prepare_recon_for_loss,
)


def _load(c: Config, ckpt: Path, device: str):
    model = build_vae_model(c).to(device)
    physics = PhysicsLoss(c.background_value, c.physics_fg_clip_min).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    if ck.get("physics_state_dict"):
        physics.load_state_dict(ck["physics_state_dict"], strict=False)
    model.train()
    physics.train()
    return model, physics


def _build_terms(
    model,
    physics,
    batch,
    c: Config,
    *,
    epoch: int,
    beta: float,
    imp_log_std: float,
):
    hm, hm_enc, occ, imp, K, pi = _tr._prepare_batch(batch, c)
    dev = c.device
    hm, hm_enc, occ, imp, K, pi = [x.to(dev, non_blocking=True) for x in (hm, hm_enc, occ, imp, K, pi)]

    rh, ro, ri, mu, lv, ex, z, distill, out_distill, _ = _forward_train_batch(
        model, hm_enc, occ, imp, K, pi, c, train=True,
    )
    rh = prepare_recon_for_loss(rh, c)
    lv = clamp_logvar(lv, c)
    ps = _tr._penalty_scale(epoch, c)
    pwts = _tr._phase_weights(epoch, c)
    pw = _tr._physics_weights(epoch, c)

    losses = _tr.vae_loss(
        rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
        epoch=epoch, K=K, physics=physics, pw=pw, pi_freq=pi,
        imp_log_std=imp_log_std, ps=ps, apply_k=True,
        weight_overrides=pwts,
    )

    terms: dict[str, torch.Tensor] = {}
    terms["kl"] = beta * losses["kl_loss"]
    terms["heatmap"] = c.heatmap_weight * losses["heatmap_loss"] * (
        pwts["heatmap_weight"] / max(c.heatmap_weight, 1e-9)
    )
    # use actual weighted pieces from losses dict
    hw = pwts.get("heatmap_weight", c.heatmap_weight)
    terms["heatmap"] = hw * losses["heatmap_loss"]
    terms["occupancy"] = c.occupancy_weight * losses["occupancy_loss"]
    iw = pwts.get("impedance_weight", c.impedance_weight)
    terms["impedance"] = iw * losses["impedance_loss"]
    terms["physics_ri"] = pw[0] * losses["physics_ri_loss"]
    terms["physics_cs"] = pw[1] * losses["physics_critic_sup_loss"]
    terms["physics_ar"] = pw[2] * losses["physics_ar_loss"]

    w_distill = float(getattr(c, "latent_distill_weight", 0.0))
    if w_distill > 0.0 and torch.is_tensor(distill):
        terms["latent_distill"] = w_distill * distill

    w_out = float(getattr(c, "output_distill_weight", 0.0))
    if w_out > 0.0 and torch.is_tensor(out_distill):
        terms["output_distill"] = w_out * out_distill

    base_phys_w = float(getattr(c, "heatmap_peak_phys_weight", 0.0))
    if base_phys_w > 0.0:
        phys_per = heatmap_peak_blob_phys_loss(rh, hm, c, pi_freq=pi, reduction="none")
        hm_w = mhz_loss_weight(pi, c)
        phys = weighted_mean(phys_per, hm_w)
        terms["phys_blob"] = base_phys_w * phys

    cf_w = pwts["cross_freq_weight"]
    if cf_w > 0 and epoch >= c.cross_freq_start_epoch and "heatmap_norm_alt" in batch:
        pi_alt = batch["PI_freq_alt"].to(dev, non_blocking=True)
        hm_alt = batch["heatmap_norm_alt"].to(dev, non_blocking=True)
        base_cf = getattr(model, "_orig_mod", model)
        cf = _cross_freq_decode(
            c, base_cf,
            hm_enc=hm_enc, occ=occ, imp=imp, K=K, pi=pi, z_decode=z,
            pi_alt=pi_alt, hm_alt=hm_alt, ps=ps,
        )
        terms["cross_freq"] = cf_w * cf

    meta = {
        "rh_max": float(rh.detach().max()),
        "total": float(losses["total_loss"].detach()),
        "imp_peak": float(losses.get("impedance_peak_loss", torch.tensor(0.0)).detach()),
    }
    return terms, model, physics, meta


def _grad_probe(term_name: str, term: torch.Tensor, model, physics) -> list[str]:
    for p in list(model.parameters()) + list(physics.parameters()):
        p.grad = None
    if not torch.isfinite(term).all():
        return [f"{term_name}:nonfinite_loss"]
    try:
        term.backward(retain_graph=True)
    except Exception as exc:
        return [f"{term_name}:backward_exc:{exc}"]
    bad = find_nonfinite_grad_names(model, physics, max_names=3)
    return [f"{term_name}->{n}" for n in bad] if bad else []


def main() -> None:
    _patch_training()
    c = Config()
    _tr._apply_yaml_config(c)
    _tr._on_stats_loaded(c, json.load(open(Path(c.data_dir) / "normalization_stats.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c.device = device

    ckpt = Path(c.experiment_dir) / "checkpoints" / "checkpoint_epoch_425.pt"
    if not ckpt.is_file():
        ckpt = sorted((Path(c.experiment_dir) / "checkpoints").glob("checkpoint_epoch_*.pt"))[-1]
    print(f"checkpoint: {ckpt}")
    model, physics = _load(c, ckpt, device)

    epoch = 417
    beta = 0.102
    imp_log_std = 1.0
    train_loader, _ = dm.create_multifreq_data_loaders(
        c.data_dir,
        batch_size=c.batch_size,
        num_workers=0,
        train_samples_per_epoch=c.train_samples_per_epoch,
        append_tag_boost_tag=getattr(c, "append_tag_boost_tag", None),
        append_tag_boost_frac=getattr(c, "append_tag_boost_frac", 0.55),
        append_tag_boost_start_epoch=getattr(c, "append_tag_boost_start_epoch", 1),
        append_tag_boost_peak_epoch=getattr(c, "append_tag_boost_peak_epoch", 100),
        append_tag_boost_decay_end_epoch=getattr(c, "append_tag_boost_decay_end_epoch", 150),
        high_freq_pair_bias=getattr(c, "high_freq_pair_bias", 0.55),
    )

    amp = _tr._amp_dtype(c)
    autocast = torch.autocast(
        device_type="cuda" if c.is_cuda() else "cpu", dtype=amp, enabled=amp is not None,
    )

    found = 0
    for bi, batch in enumerate(train_loader):
        if bi > 800:
            break
        with autocast:
            terms, model, physics, meta = _build_terms(
                model, physics, batch, c, epoch=epoch, beta=beta, imp_log_std=imp_log_std,
            )

        # full backward probe
        for p in list(model.parameters()) + list(physics.parameters()):
            p.grad = None
        total = sum(terms.values())
        total.backward()
        bad_full = find_nonfinite_grad_names(model, physics, max_names=5)

        if bad_full:
            found += 1
            print(
                f"\n=== batch {bi} rh_max={meta['rh_max']:.3f} total={meta['total']:.3f} "
                f"imp_peak={meta['imp_peak']:.4f} ==="
            )
            print("full_bad:", bad_full)
            for name, term in terms.items():
                hits = _grad_probe(name, term, model, physics)
                if hits:
                    print(" ", hits[0] if len(hits) == 1 else hits)
            if found >= 8:
                break

    print(f"\nDone. nonfinite batches found: {found}")


if __name__ == "__main__":
    main()
