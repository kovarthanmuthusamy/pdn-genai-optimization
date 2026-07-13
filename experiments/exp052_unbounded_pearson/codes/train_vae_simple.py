"""Train VAE for exp052 — Pearson + grad + phys top-k blob (no percentile/dynrange/z-clip)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses import heatmap_loss_pearson_grad
from experiments.exp052_unbounded_pearson.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp052_unbounded_pearson.codes.spatial_metrics import peak_loc_loss
from experiments.exp052_unbounded_pearson.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp052_unbounded_pearson.codes.vae_poe_freq import MultiInputVAEPoeFreq

_orig_prepare = _tr._prepare_batch


@dataclass
class Config(_tr.Config):
    synthetic_blend_prob: float = 0.08
    use_freq_poe_expert: bool = True
    use_heatmap_unet_skips: bool = True
    use_occ_spatial_decoder: bool = True
    use_occ_spatial_tower: bool = True
    occ_spatial_ch: int = 8
    cross_freq_use_encode_z: bool = True
    encode_native_teacher_skips: bool = True
    layout_train_prob: float = 0.55
    heatmap_private_dim: int = 15
    heatmap_peak_loc_weight: float = 1.0
    heatmap_pearson_weight: float = 2.5
    use_layout_private_head: bool = True
    layout_private_hidden: int = 384
    use_layout_private_freq_film: bool = True
    latent_distill_weight: float = 2.0
    latent_distill_private_mult: float = 3.0
    latent_distill_logvar_weight: float = 0.3
    output_distill_weight: float = 2.5
    freeze_occ_imp_epoch: int = 320
    # exp052: no legacy peak/dynrange/p99 heatmap terms or recon z-clip
    heatmap_peak_weight: float = 0.0
    heatmap_dynrange_weight: float = 0.0
    heatmap_focus_dynrange_weight: float = 0.0
    heatmap_phys_p99_weight: float = 0.0
    heatmap_clip_recon: bool = False
    heatmap_z_clip_min: float | None = None
    heatmap_z_clip_max: float | None = None
    heatmap_peak_topk_k: int = 24
    heatmap_peak_phys_weight: float = 2.0
    heatmap_peak_phys_overshoot_weight: float = 1.75
    heatmap_peak_phys_hf_mult: float = 1.5
    heatmap_grad_vector_weight: float = 2.0
    heatmap_grad_direction_weight: float = 1.0
    heatmap_grad_direction_min_mag: float = 0.08
    heatmap_grad_huber_delta: float = 0.5
    # High-freq / layout path
    layout_high_freq_mhz_threshold: float = 250.0
    layout_high_freq_loss_mult: float = 2.0
    layout_spatial_loss_mult: float = 1.75
    layout_peak_grad_mult: float = 1.5
    cross_freq_layout_mix_prob: float = 0.45
    cross_freq_high_freq_loss_mult: float = 2.5
    high_freq_pair_bias: float = 0.55
    # MHz-dependent heatmap loss weight (low MHz < high MHz)
    heatmap_mhz_loss_weight_enabled: bool = True
    heatmap_mhz_loss_weight_mode: str = "log"
    heatmap_mhz_loss_weight_min_mhz: float = 10.0
    heatmap_mhz_loss_weight_max_mhz: float = 500.0
    heatmap_mhz_loss_weight_min: float = 0.5
    heatmap_mhz_loss_weight_max: float = 2.5
    # NaN guards
    amp: str = "off"
    training_force_fp32: bool = True
    training_sanitize_grads: bool = False
    training_skip_nonfinite_batches: bool = True
    training_logvar_clamp_max: float = 10.0
    training_logvar_clamp_min: float = -10.0
    training_recon_z_soft_cap: float | None = None
    training_recon_z_cap_mult: float = 1.25
    training_recon_z_soft_floor: float = -6.0
    training_loss_finite_cap: float = 1e4
    training_grad_clip_norm: float = 1.0
    early_stop_min_epoch: int = 0
    high_freq_mhz_threshold: float = 250.0
    early_stop_encode_cross_mhz: float = 0.0
    early_stop_encode_cross_patience: int = 0
    _early_stopped: bool = False
    # Append-tag sampling curriculum (New_data_10000 boost early, then decay)
    append_tag_boost_tag: str = "New_data_10000"
    append_tag_boost_frac: float = 0.55
    append_tag_boost_start_epoch: int = 1
    append_tag_boost_peak_epoch: int = 100
    append_tag_boost_decay_end_epoch: int = 150
    # Layout-path curriculum (epochs 1–150)
    layout_train_prob_base: float = 0.55
    layout_train_prob_boost: float = 0.62
    cross_freq_layout_mix_prob_base: float = 0.45
    cross_freq_layout_mix_prob_boost: float = 0.50
    layout_path_boost_start_epoch: int = 1
    layout_path_boost_peak_epoch: int = 100
    layout_path_boost_decay_end_epoch: int = 150
    # Phase 3: careful impedance decoder unfreeze
    unfreeze_impedance_on_resume: bool = False
    unfreeze_impedance_min_start_epoch: int = 850
    impedance_separate_lr: bool = False

_ACTIVE_TRAIN_CONFIG: list[Config] = []
_LAST_CURRICULUM_LOG: int = 0


def _on_train_epoch_start(epoch_1based: int, c: Config, train_ld) -> None:
    global _LAST_CURRICULUM_LOG
    from experiments.exp052_unbounded_pearson.codes.sampler_curriculum import apply_layout_path_curriculum

    apply_layout_path_curriculum(c, epoch_1based)
    sampler = getattr(train_ld, "exp052_curriculum_sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_1based)

    if epoch_1based == 1 or epoch_1based % 25 == 0 or epoch_1based != _LAST_CURRICULUM_LOG:
        if epoch_1based <= int(getattr(c, "layout_path_boost_decay_end_epoch", 0)) or epoch_1based % 25 == 0:
            print(
                f"  [curriculum ep {epoch_1based}] layout_train_prob={c.layout_train_prob:.3f}  "
                f"cf_layout_mix={c.cross_freq_layout_mix_prob:.3f}",
                flush=True,
            )
            _LAST_CURRICULUM_LOG = epoch_1based


def build_vae_model(c: Config) -> MultiInputVAEPoeFreq:
    model = MultiInputVAEPoeFreq(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=bool(getattr(c, "use_heatmap_film", True)),
        use_freq_poe_expert=bool(getattr(c, "use_freq_poe_expert", True)),
        use_heatmap_unet_skips=bool(getattr(c, "use_heatmap_unet_skips", True)),
        use_occ_spatial_decoder=bool(getattr(c, "use_occ_spatial_decoder", True)),
        use_occ_spatial_tower=bool(getattr(c, "use_occ_spatial_tower", True)),
        occ_spatial_ch=int(getattr(c, "occ_spatial_ch", 8)),
        use_layout_private_head=bool(getattr(c, "use_layout_private_head", True)),
        layout_private_hidden=int(getattr(c, "layout_private_hidden", 384)),
        use_layout_private_freq_film=bool(getattr(c, "use_layout_private_freq_film", True)),
    )
    c._model_ref = model
    print(
        f"  exp052 model: private_h={c.layout_private_hidden}  freq_film={c.use_layout_private_freq_film}  "
        f"layout_prob={c.layout_train_prob}  cf_layout_mix={c.cross_freq_layout_mix_prob}\n"
        f"    mhz_loss={c.heatmap_mhz_loss_weight_min:.2f}-{c.heatmap_mhz_loss_weight_max:.2f} ({c.heatmap_mhz_loss_weight_mode})  "
        f"pearson_w={c.heatmap_pearson_weight}  grad_vec={c.heatmap_grad_vector_weight}  grad_dir={c.heatmap_grad_direction_weight}  "
        f"phys_w={c.heatmap_peak_phys_weight}  peak_loc={c.heatmap_peak_loc_weight}\n"
        f"    distill_w={c.latent_distill_weight}  freeze_occ_imp_epoch={c.freeze_occ_imp_epoch}",
    )
    return model


def _heatmap_loss_with_spatial(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    ps: float,
    *,
    lite: bool = False,
    **_,
) -> torch.Tensor:
    del ps
    loss = heatmap_loss_pearson_grad(recon, target, c, lite=lite)
    peak_loc_w = float(getattr(c, "heatmap_peak_loc_weight", 0.0))
    if peak_loc_w <= 0:
        return loss
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    return loss + peak_loc_w * peak_loc_loss(recon, target, fg, bg)


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


_FROZEN_OCC_IMP: dict = {"done": False}
_orig_phase_weights = _tr._phase_weights


def _phase_weights_with_freeze(epoch: int, c: Config) -> dict:
    w = _orig_phase_weights(epoch, c)
    freeze_ep = int(getattr(c, "freeze_occ_imp_epoch", 0))
    if freeze_ep > 0 and epoch >= freeze_ep and not _FROZEN_OCC_IMP["done"]:
        if _ACTIVE_TRAIN_CONFIG:
            _freeze_occ_imp_decoders(_ACTIVE_TRAIN_CONFIG[0])
    return w




_IMPEDANCE_TRAIN_TAGS = ("impedance_encoder", "impedance_mu", "impedance_logvar", "impedance_decoder", "imp_fc")


def _unfreeze_impedance_modules(c: Config) -> int:
    """Unfreeze impedance path; keep occupancy decoder frozen."""
    model = getattr(c, "_model_ref", None)
    if model is None:
        return 0
    n = 0
    for name, param in model.named_parameters():
        if any(tag in name for tag in _IMPEDANCE_TRAIN_TAGS):
            if param.requires_grad:
                continue
            param.requires_grad = True
            n += 1
    return n


def _on_after_checkpoint_load(c: Config, model, start: int) -> None:
    c._model_ref = model
    if not getattr(c, "unfreeze_impedance_on_resume", False):
        return
    if start < int(getattr(c, "unfreeze_impedance_min_start_epoch", 850)):
        print(f"  >> Skip impedance unfreeze (resume epoch {start} < min {c.unfreeze_impedance_min_start_epoch})")
        return
    n = _unfreeze_impedance_modules(c)
    print(
        f"  >> Phase-3: unfroze {n} impedance parameter tensor(s) "
        f"(encoder+decoder); occupancy decoder stays frozen",
        flush=True,
    )


def _freeze_occ_imp_decoders(c: Config) -> None:
    if _FROZEN_OCC_IMP["done"]:
        return
    model = getattr(c, "_model_ref", None)
    if model is None:
        return
    frozen = 0
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("occupancy_dec", "impedance_dec", "occ_fc", "imp_fc")):
            param.requires_grad = False
            frozen += 1
    _FROZEN_OCC_IMP["done"] = True
    print(f"  >> Froze {frozen} occ/imp decoder params at epoch {int(getattr(c, 'freeze_occ_imp_epoch', 0))}")


def _on_stats_loaded(c: Config, raw: dict) -> None:
    """Attach unbounded norm bundle (exp052 requires unbounded stats)."""
    from src_vae.others.norm_stats import NormStatsBundle

    bundle = NormStatsBundle.load(c.data_dir)
    c._norm_stats = bundle
    c._hm_log_mean = float(bundle.heatmap.log_mean)
    c._hm_log_std = float(bundle.heatmap.log_std)
    hm = bundle.heatmap
    if not hm.is_unbounded():
        raise ValueError(
            f"exp052 requires unbounded heatmap stats; got {hm.norm_mode!r} (unbounded={hm.unbounded})",
        )
    c.physics_fg_clip_min = float(hm.background_value)
    if bool(getattr(c, "training_force_fp32", True)):
        c.amp = "off"
    print(f"  Central norm: {hm.describe()}")



_orig_vae_loss = _tr.vae_loss


def _vae_loss_mhz_weighted(
    recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp,
    mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq,
    imp_log_std, ps, apply_k, weight_overrides=None,
):
    from experiments.exp052_unbounded_pearson.codes.mhz_loss_weight import (
        mhz_loss_weight,
        weighted_mean,
    )
    from experiments.exp052_unbounded_pearson.codes.training_guard import (
        clamp_logvar,
        sanitize_loss_dict,
    )

    wo = weight_overrides or {}
    hw = wo.get("heatmap_weight", c.heatmap_weight)
    hm_ps = _tr.heatmap_loss(recon_hm, tgt_hm, c, ps)

    sample_w = None
    if bool(getattr(c, "heatmap_mhz_loss_weight_enabled", False)) and pi_freq is not None:
        sample_w = mhz_loss_weight(pi_freq, c)
        if apply_k and c.use_k_weighting and K is not None:
            wh, _ = _tr._k_weights(K, c)
            sample_w = sample_w * wh.to(sample_w.dtype)

    logvar = clamp_logvar(logvar, c)

    losses = _orig_vae_loss(
        recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp,
        mu, logvar, beta, c, expert_stats, epoch=epoch, K=K, physics=physics,
        pw=pw, pi_freq=pi_freq, imp_log_std=imp_log_std, ps=ps, apply_k=False,
        weight_overrides=weight_overrides,
    )

    if sample_w is not None:
        loss_hm_new = weighted_mean(hm_ps, sample_w)
        old_hm = losses["heatmap_loss"]
        delta = hw * (loss_hm_new - old_hm)
        losses["heatmap_loss"] = loss_hm_new
        losses["recon_loss"] = losses["recon_loss"] + delta
        losses["total_loss"] = losses["total_loss"] + delta

    return sanitize_loss_dict(losses, c)

def _patch_training() -> None:
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._run_epoch = _run_epoch_encode
    _tr.heatmap_loss = _heatmap_loss_with_spatial
    _tr.vae_loss = _vae_loss_mhz_weighted
    _tr._phase_weights = _phase_weights_with_freeze
    _tr._on_stats_loaded = _on_stats_loaded
    _tr.LOSS_KEYS = _tr.LOSS_KEYS + ("heatmap_peak_phys_loss",)
    _patch_spatial_eval()
    _patch_dataloader()

    _orig_apply = _tr._apply_yaml_config

    def _apply_yaml_config_track(c: Config) -> None:
        _orig_apply(c)
        _ACTIVE_TRAIN_CONFIG.clear()
        _ACTIVE_TRAIN_CONFIG.append(c)

    _tr._apply_yaml_config = _apply_yaml_config_track

    def _bind_epoch_start(c: Config) -> None:
        c.on_train_epoch_start = _on_train_epoch_start
        c.on_after_checkpoint_load = _on_after_checkpoint_load

    _orig_apply2 = _tr._apply_yaml_config

    def _apply_yaml_with_epoch_hook(c: Config) -> None:
        _orig_apply2(c)
        _bind_epoch_start(c)

    _orig_build_optimizer = _tr._build_optimizer

    def _build_optimizer_careful_imp(
        model, physics, c: Config, epoch_start: int, *, use_grad_scaler: bool = False,
    ):
        base = float(c.learning_rate)
        mult = float(getattr(c, "impedance_decoder_lr_mult", 1.0))
        focus_ep = int(getattr(c, "impedance_peak_focus_epoch", 0))
        split = bool(getattr(c, "impedance_separate_lr", False)) or (
            getattr(c, "unfreeze_impedance_on_resume", False) and mult != 1.0
        )
        if split and epoch_start >= focus_ep:
            imp_params = _tr._impedance_module_params(model)
            imp_ids = {id(p) for p in imp_params}
            other_params = [
                p for _, p in model.named_parameters()
                if p.requires_grad and id(p) not in imp_ids
            ]
            imp_lr = base * mult
            groups = [
                {"params": other_params, "lr": base},
                {"params": imp_params, "lr": imp_lr},
            ]
            if physics:
                groups.append({"params": list(physics.parameters()), "lr": base})
            print(
                f"  Careful impedance LR: backbone={base:.2e}  "
                f"impedance={imp_lr:.2e} (mult={mult})",
                flush=True,
            )
            return _tr._adamw(groups, lr=base, c=c, use_grad_scaler=use_grad_scaler)
        return _orig_build_optimizer(model, physics, c, epoch_start, use_grad_scaler=use_grad_scaler)

    _tr._build_optimizer = _build_optimizer_careful_imp
    _tr._apply_yaml_config = _apply_yaml_with_epoch_hook


def _patch_dataloader() -> None:
    from experiments.exp052_unbounded_pearson.codes import dataloader_multifreq as _dm047
    import experiments.exp038_true_multi.codes.train_vae_simple as tr038

    def _create_with_hf_bias(*args, **kwargs):
        c = _ACTIVE_TRAIN_CONFIG[0] if _ACTIVE_TRAIN_CONFIG else None
        if c is not None:
            kwargs.setdefault("high_freq_pair_bias", float(c.high_freq_pair_bias))
            kwargs.setdefault("high_freq_mhz_threshold", float(c.high_freq_mhz_threshold))
            kwargs.setdefault("append_tag_boost_tag", getattr(c, "append_tag_boost_tag", None))
            kwargs.setdefault("append_tag_boost_frac", float(getattr(c, "append_tag_boost_frac", 0.55)))
            kwargs.setdefault("append_tag_boost_start_epoch", int(getattr(c, "append_tag_boost_start_epoch", 1)))
            kwargs.setdefault("append_tag_boost_peak_epoch", int(getattr(c, "append_tag_boost_peak_epoch", 100)))
            kwargs.setdefault("append_tag_boost_decay_end_epoch", int(getattr(c, "append_tag_boost_decay_end_epoch", 150)))
        return _dm047.create_multifreq_data_loaders(*args, **kwargs)

    tr038.create_multifreq_data_loaders = _create_with_hf_bias
    _tr.create_multifreq_data_loaders = _create_with_hf_bias


def _patch_spatial_eval() -> None:
    import experiments.exp038_true_multi.codes.eval_cross_freq as _eval_cf
    from experiments.exp052_unbounded_pearson.codes.eval_spatial_metrics import run_off_anchor_eval_spatial

    if getattr(_eval_cf, "_exp052_spatial_patched", False):
        return

    def _spatial_off_anchor_eval(
        model,
        val_loader,
        *,
        bg: float,
        off_anchor_mhz=(100.0, 175.0, 330.0, 400.0),
        max_batches: int = 30,
        device: str | torch.device = "cuda",
        out_csv=None,
    ):
        return run_off_anchor_eval_spatial(
            model,
            val_loader,
            bg=bg,
            off_anchor_mhz=off_anchor_mhz,
            max_batches=max_batches,
            device=device,
            out_csv=out_csv,
            early_stop_state=None,
        )

    _eval_cf.run_off_anchor_eval = _spatial_off_anchor_eval
    _eval_cf._exp052_spatial_patched = True


def _print_training_summary(c: _tr.Config) -> None:
    enc_pct = (1.0 - float(c.layout_train_prob)) * 100.0
    print(
        "  exp052 training (Tier A lean: Pearson global + grad local + phys blob + peak_loc):\n"
        f"    layout_train_prob={c.layout_train_prob} (~{enc_pct:.0f}% encode)  "
        f"cf_layout_mix={c.cross_freq_layout_mix_prob}  high_freq_pair_bias={c.high_freq_pair_bias}\n"
        f"    spatial: peak_loc_w={c.heatmap_peak_loc_weight}  "
        f"grad_vec_w={c.heatmap_grad_vector_weight}  grad_dir_w={c.heatmap_grad_direction_weight}\n"
        f"    peak_topk={getattr(c, 'heatmap_peak_topk_k', 24)}  "
        f"phys_w={getattr(c, 'heatmap_peak_phys_weight', 0)}  "
        f"phys_over={getattr(c, 'heatmap_peak_phys_overshoot_weight', 0)}\n"
        f"    hf_mult={c.layout_high_freq_loss_mult}  private_h={c.layout_private_hidden}  "
        f"freq_film={c.use_layout_private_freq_film}\n"
        f"    epochs={c.num_epochs}  nan_guard=on  z_cap_mult={c.training_recon_z_cap_mult}  "
        f"logvar_clamp=[{c.training_logvar_clamp_min},{c.training_logvar_clamp_max}]",
    )


def train_vae() -> None:
    _patch_training()
    c = Config()
    _tr._apply_yaml_config(c)
    _print_training_summary(c)
    _tr.train_vae()


def main() -> None:
    exp_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(exp_dir))
    cfg_path = exp_dir / "config.yaml"
    if cfg_path.is_file():
        lines = [
            ln for ln in cfg_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        data_dir = json.loads("\n".join(lines)).get("data_dir")
        if data_dir:
            os.environ.setdefault("VAE_DATA_DIR", str(data_dir))
    else:
        os.environ.setdefault("VAE_DATA_DIR", str(_PROJECT_ROOT / "datasets/data_multifreq_train_norm_unbounded"))
    train_vae()


if __name__ == "__main__":
    main()
