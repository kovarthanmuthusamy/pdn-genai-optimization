"""
Train exp040 FactorizedFreqVAE.

Patches exp038 training with factorized heatmap decode, synthetic mid-MHz
blends, and optional mode-orthogonality loss.
"""

from __future__ import annotations

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
from experiments.exp040.codes import vae_factorized_freq as _vf
from experiments.exp040.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp040.codes.vae_factorized_freq import FactorizedFreqVAE, modes_orthogonality_loss

_orig_prepare = _tr._prepare_batch
_orig_vae_loss = _tr.vae_loss
_orig_load_checkpoint = _tr.load_checkpoint

# Legacy exp039 heatmap decoder (residual path) — frozen when use_freq_residual=false.
_RESIDUAL_PREFIXES = (
    "heatmap_fc",
    "heatmap_dec_deconv1",
    "heatmap_dec_attn",
    "heatmap_dec_film",
    "heatmap_dec_deconv2",
    "residual_gain",
)


@dataclass
class Config(_tr.Config):
    num_freq_modes: int = 6
    freq_alpha_softmax: bool = False
    use_freq_residual: bool = True
    residual_gain_init: float = 0.15
    mode_orthogonality_weight: float = 0.05
    synthetic_blend_prob: float = 0.25
    inference_factorized_only: bool = True


def build_factorized_vae(c: Config) -> FactorizedFreqVAE:
    return FactorizedFreqVAE(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=c.use_heatmap_film,
        num_freq_modes=int(getattr(c, "num_freq_modes", 6)),
        freq_alpha_softmax=bool(getattr(c, "freq_alpha_softmax", False)),
        use_freq_residual=bool(getattr(c, "use_freq_residual", True)),
        residual_gain_init=float(getattr(c, "residual_gain_init", 0.15)),
    )


def configure_factorized_training(model: FactorizedFreqVAE, c: Config) -> None:
    """Apply yaml flags and freeze legacy residual decoder when disabled."""
    model.use_freq_residual = bool(getattr(c, "use_freq_residual", True))
    model.freq_alpha_softmax = bool(getattr(c, "freq_alpha_softmax", False))
    model.inference_factorized_only = bool(getattr(c, "inference_factorized_only", True))

    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if not model.use_freq_residual and name.startswith(_RESIDUAL_PREFIXES):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()

    print(
        f"  exp040 decode: use_freq_residual={model.use_freq_residual}, "
        f"freq_alpha_softmax={model.freq_alpha_softmax}, "
        f"inference_factorized_only={model.inference_factorized_only}",
    )
    if not model.use_freq_residual:
        print(
            f"  Legacy heatmap decoder frozen ({frozen:,} params); "
            f"training factorized path ({trainable:,} params).",
        )


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


def vae_loss(*args, **kwargs):
    losses = _orig_vae_loss(*args, **kwargs)
    c = kwargs.get("c") if "c" in kwargs else args[9]
    w = float(getattr(c, "mode_orthogonality_weight", 0.0))
    bases = _vf.LAST_BASES
    _vf.LAST_BASES = None
    if w > 0.0 and bases is not None:
        orth = modes_orthogonality_loss(bases)
        losses["mode_orth_loss"] = orth
        losses["total_loss"] = losses["total_loss"] + w * orth
    return losses


def _patch_training_module() -> None:
    _tr.Config = Config
    _tr.MultiInputVAE = FactorizedFreqVAE
    _tr.build_vae_model = build_factorized_vae
    _tr._prepare_batch = _prepare_batch
    _tr.vae_loss = vae_loss


def train_vae() -> None:
    _patch_training_module()

    _orig_train = _tr.train_vae
    _orig_lc = _tr.load_checkpoint

    def _load_checkpoint_with_configure(path, model, optimizer=None, device="cuda", physics=None):
        epoch, val, ckpt_cfg = _orig_lc(path, model, optimizer, device, physics)
        if isinstance(model, FactorizedFreqVAE):
            c = _tr.Config()
            _tr._apply_yaml_config(c)
            configure_factorized_training(model, c)
        return epoch, val, ckpt_cfg

    _tr.load_checkpoint = _load_checkpoint_with_configure

    # Run training; configure model after build via hook on build_vae_model
    _orig_build = _tr.build_vae_model

    def _build_and_configure(c: Config) -> FactorizedFreqVAE:
        model = build_factorized_vae(c)
        configure_factorized_training(model, c)
        return model

    _tr.build_vae_model = _build_and_configure
    try:
        _orig_train()
    finally:
        _tr.build_vae_model = _orig_build
        _tr.load_checkpoint = _orig_lc


def main() -> None:
    exp_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(exp_dir))
    os.environ.setdefault("VAE_DATA_DIR", str(_PROJECT_ROOT / "datasets" / "data_multifreq_norm"))
    train_vae()


if __name__ == "__main__":
    main()
