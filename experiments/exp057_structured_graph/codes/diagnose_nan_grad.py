"""Isolate which loss term / layer produces non-finite gradients (exp052)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

os.environ.setdefault("VAE_EXPERIMENT_DIR", "experiments/exp057_structured_graph")

import experiments.exp057_structured_graph.codes.train_core as _tr
from experiments.exp057_structured_graph.codes import dataloader_multifreq as dm
from experiments.exp057_structured_graph.codes.run_epoch_encode import (
    _cross_freq_decode,
    _forward_train_batch,
)
from experiments.exp057_structured_graph.codes.train_vae_simple import (
    Config,
    _patch_training,
    build_vae_model,
)
from experiments.exp057_structured_graph.codes.training_guard import (
    clamp_logvar,
    find_nonfinite_grad_names,
    prepare_recon_for_loss,
)


def _load(c: Config, ckpt: Path, device: str):
    model = build_vae_model(c).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.train()
    return model


def _build_terms(
    model,
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
    losses = _tr.vae_loss(
        rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
        epoch=epoch, K=K, physics=None, pw=None, pi_freq=pi,
        imp_log_std=imp_log_std, ps=ps, apply_k=False,
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
    w_distill = float(getattr(c, "latent_distill_weight", 0.0))
    if w_distill > 0.0 and torch.is_tensor(distill):
        terms["latent_distill"] = w_distill * distill

    w_out = float(getattr(c, "output_distill_weight", 0.0))
    if w_out > 0.0 and torch.is_tensor(out_distill):
        terms["output_distill"] = w_out * out_distill

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
    return terms, model, meta


def _grad_probe(term_name: str, term: torch.Tensor, model) -> list[str]:
    params = list(model.parameters())
    for p in params:
        p.grad = None
    if not torch.isfinite(term).all():
        return [f"{term_name}:nonfinite_loss"]
    try:
        term.backward(retain_graph=True)
    except Exception as exc:
        return [f"{term_name}:backward_exc:{exc}"]
    bad = find_nonfinite_grad_names(model, max_names=3)
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
    model = _load(c, ckpt, device)

    epoch = 417
    beta = 0.102
    imp_log_std = 1.0
    train_loader, _ = dm.create_multifreq_data_loaders(
        c.data_dir,
        batch_size=c.batch_size,
        num_workers=0,
        train_samples_per_epoch=c.train_samples_per_epoch,
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
            terms, model, meta = _build_terms(
                model, batch, c, epoch=epoch, beta=beta, imp_log_std=imp_log_std,
            )

        # full backward probe
        params = list(model.parameters())
        for p in params:
            p.grad = None
        total = sum(terms.values())
        total.backward()
        bad_full = find_nonfinite_grad_names(model, max_names=5)

        if bad_full:
            found += 1
            print(
                f"\n=== batch {bi} rh_max={meta['rh_max']:.3f} total={meta['total']:.3f} "
                f"imp_peak={meta['imp_peak']:.4f} ==="
            )
            print("full_bad:", bad_full)
            for name, term in terms.items():
                hits = _grad_probe(name, term, model)
                if hits:
                    print(" ", hits[0] if len(hits) == 1 else hits)
            if found >= 8:
                break

    print(f"\nDone. nonfinite batches found: {found}")


if __name__ == "__main__":
    main()
