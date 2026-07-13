"""Train VAE for exp050 — pixel peak blob losses, no percentile training terms."""

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
from experiments.exp050.codes.heatmap_peak_losses import heatmap_loss_tier_a
from experiments.exp050.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp050.codes.spatial_metrics import peak_loc_loss
from experiments.exp050.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp050.codes.vae_poe_freq import MultiInputVAEPoeFreq

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
    heatmap_peak_loc_weight: float = 0.5
    use_layout_private_head: bool = True
    layout_private_hidden: int = 384
    use_layout_private_freq_film: bool = True
    latent_distill_weight: float = 2.0
    latent_distill_private_mult: float = 3.0
    latent_distill_logvar_weight: float = 0.3
    output_distill_weight: float = 2.5
    freeze_occ_imp_epoch: int = 320
    # Pixel peak blob (no percentile training)
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
    high_freq_mhz_threshold: float = 250.0
    early_stop_encode_cross_mhz: float = 0.0
    early_stop_encode_cross_patience: int = 0
    early_stop_min_epoch: int = 0
    _early_stopped: bool = False


_ACTIVE_TRAIN_CONFIG: list[Config] = []


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
        f"  exp050 model: private_h={c.layout_private_hidden}  freq_film={c.use_layout_private_freq_film}  "
        f"layout_prob={c.layout_train_prob}  cf_layout_mix={c.cross_freq_layout_mix_prob}\n"
        f"    grad_vec={c.heatmap_grad_vector_weight}  grad_dir={c.heatmap_grad_direction_weight}  "
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
    dynrange_weight: float | None = None,
    lite: bool = False,
) -> torch.Tensor:
    del ps, dynrange_weight, lite
    loss = heatmap_loss_tier_a(recon, target, c)
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
    """Attach norm bundle; disable z-clip for unbounded robust stats."""
    from src_vae.others.heatmap_z_clip import load_heatmap_z_clip_bounds
    from src_vae.others.norm_stats import NormStatsBundle

    bundle = NormStatsBundle.load(c.data_dir)
    c._norm_stats = bundle
    c._hm_log_mean = float(bundle.heatmap.log_mean)
    c._hm_log_std = float(bundle.heatmap.log_std)
    hm = bundle.heatmap
    if hm.is_unbounded():
        c.heatmap_z_clip_min = None
        c.heatmap_z_clip_max = None
        c.heatmap_clip_recon = False
        c.physics_fg_clip_min = float(hm.background_value)
        print(f"  Central norm: {hm.describe()}  (no z-clip)")
        return
    if hm.is_robust_per_mhz():
        cb = load_heatmap_z_clip_bounds(c.data_dir)
        if cb is not None:
            c.heatmap_z_clip_min, c.heatmap_z_clip_max = cb
            c.physics_fg_clip_min = cb[0]
    print(f"  Central norm: {hm.describe()}")


def _patch_training() -> None:
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._run_epoch = _run_epoch_encode
    _tr.heatmap_loss = _heatmap_loss_with_spatial
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


def _patch_dataloader() -> None:
    from experiments.exp050.codes import dataloader_multifreq as _dm047
    import experiments.exp038_true_multi.codes.train_vae_simple as tr038

    def _create_with_hf_bias(*args, **kwargs):
        c = _ACTIVE_TRAIN_CONFIG[0] if _ACTIVE_TRAIN_CONFIG else None
        if c is not None:
            kwargs.setdefault("high_freq_pair_bias", float(c.high_freq_pair_bias))
            kwargs.setdefault("high_freq_mhz_threshold", float(c.high_freq_mhz_threshold))
        return _dm047.create_multifreq_data_loaders(*args, **kwargs)

    tr038.create_multifreq_data_loaders = _create_with_hf_bias
    _tr.create_multifreq_data_loaders = _create_with_hf_bias


def _patch_spatial_eval() -> None:
    import experiments.exp038_true_multi.codes.eval_cross_freq as _eval_cf
    from experiments.exp050.codes.eval_spatial_metrics import run_off_anchor_eval_spatial

    if getattr(_eval_cf, "_exp050_spatial_patched", False):
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
    _eval_cf._exp050_spatial_patched = True


def _print_training_summary(c: _tr.Config) -> None:
    enc_pct = (1.0 - float(c.layout_train_prob)) * 100.0
    print(
        "  exp050 training (Tier A lean: FG huber + grad + phys blob + peak_loc):\n"
        f"    layout_train_prob={c.layout_train_prob} (~{enc_pct:.0f}% encode)  "
        f"cf_layout_mix={c.cross_freq_layout_mix_prob}  high_freq_pair_bias={c.high_freq_pair_bias}\n"
        f"    spatial: peak_loc_w={c.heatmap_peak_loc_weight}  "
        f"peak_w={c.heatmap_peak_weight}  grad_vec_w={c.heatmap_grad_vector_weight}  dyn_w={c.heatmap_dynrange_weight}\n"
        f"    peak_topk={getattr(c, 'heatmap_peak_topk_k', 24)}  "
        f"phys_w={getattr(c, 'heatmap_peak_phys_weight', 0)}  "
        f"phys_over={getattr(c, 'heatmap_peak_phys_overshoot_weight', 0)}  "
        f"p99_train=off\n"
        f"    hf_mult={c.layout_high_freq_loss_mult}  private_h={c.layout_private_hidden}  "
        f"freq_film={c.use_layout_private_freq_film}\n"
        f"    epochs={c.num_epochs}  early_stop=disabled",
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
        os.environ.setdefault("VAE_DATA_DIR", str(_PROJECT_ROOT / "data_multi_norm_unbounded"))
    train_vae()


if __name__ == "__main__":
    main()
