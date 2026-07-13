"""Train VAE for exp045 — unbounded norm + U-Net skips + encode-first + robust losses."""

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
from experiments.exp045.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp045.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp045.codes.unbounded_heatmap_loss import patch_unbounded_losses
from experiments.exp045.codes.vae_poe_freq import MultiInputVAEPoeFreq

_orig_prepare = _tr._prepare_batch
_orig_on_stats = _tr._on_stats_loaded


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
    layout_train_prob: float = 0.30
    heatmap_unbounded: bool = True
    heatmap_clip_recon: bool = False
    heatmap_huber_delta: float = 2.0
    heatmap_tail_z_threshold: float = 3.0
    heatmap_tail_weight_boost: float = 2.0
    heatmap_soft_cap_z: float = 6.0
    heatmap_private_dim: int = 12
    heatmap_pearson_weight: float = 1.0
    heatmap_peak_loc_weight: float = 0.5
    heatmap_grad_weight: float = 2.5
    heatmap_lap_weight: float = 3.0
    early_stop_encode_cross_mhz: float = 330.0
    early_stop_encode_cross_patience: int = 3


_ACTIVE_TRAIN_CONFIG: list[Config] = []
_SPATIAL_EARLY_STOP: dict = {}


def build_vae_model(c: Config) -> MultiInputVAEPoeFreq:
    model = MultiInputVAEPoeFreq(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=c.use_heatmap_film,
        use_freq_poe_expert=bool(getattr(c, "use_freq_poe_expert", True)),
        use_heatmap_unet_skips=bool(getattr(c, "use_heatmap_unet_skips", True)),
        use_occ_spatial_decoder=bool(getattr(c, "use_occ_spatial_decoder", True)),
        use_occ_spatial_tower=bool(getattr(c, "use_occ_spatial_tower", True)),
        occ_spatial_ch=int(getattr(c, "occ_spatial_ch", 8)),
    )
    print(
        f"  exp045 model: U-Net skips={getattr(c, 'use_heatmap_unet_skips', True)}  "
        f"freq PoE private d={c.heatmap_private_dim}  "
        f"occ_spatial={getattr(c, 'use_occ_spatial_decoder', True)}  "
        f"occ_tower={getattr(c, 'use_occ_spatial_tower', True)}",
    )
    return model


def _on_stats_loaded(c: Config, raw: dict) -> None:
    _orig_on_stats(c, raw)
    hm = raw.get("Heatmap") or {}
    if hm.get("unbounded") or getattr(c, "heatmap_unbounded", False):
        c.heatmap_unbounded = True
        c.heatmap_clip_recon = False
        c.heatmap_z_clip_min = None
        c.heatmap_z_clip_max = None
        c.background_value = float(raw.get("background_value", hm.get("background_value", c.background_value)))
        zmin, zmax = hm.get("z_min"), hm.get("z_max")
        zrng = f"z∈[{zmin:.4f}, {zmax:.4f}]" if zmin is not None and zmax is not None else "z unbounded"
        print(
            f"  unbounded heatmap norm: bg={c.background_value:.4f}  {zrng}  no load/recon clip",
        )


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


def _patch_training() -> None:
    patch_unbounded_losses()
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._on_stats_loaded = _on_stats_loaded
    _tr._run_epoch = _run_epoch_encode
    _patch_spatial_eval()
    _orig_apply = _tr._apply_yaml_config

    def _apply_yaml_config_track(c: Config) -> None:
        _orig_apply(c)
        _ACTIVE_TRAIN_CONFIG.clear()
        _ACTIVE_TRAIN_CONFIG.append(c)
        _SPATIAL_EARLY_STOP.clear()
        _SPATIAL_EARLY_STOP.update({
            "config": c,
            "best_mse": float("inf"),
            "patience": 0,
            "stopped": False,
        })

    _tr._apply_yaml_config = _apply_yaml_config_track


def _patch_spatial_eval() -> None:
    import experiments.exp038_true_multi.codes.eval_cross_freq as _eval_cf
    from experiments.exp045.codes.eval_spatial_metrics import run_off_anchor_eval_spatial

    if getattr(_eval_cf, "_exp045_spatial_patched", False):
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
        c = _ACTIVE_TRAIN_CONFIG[0] if _ACTIVE_TRAIN_CONFIG else None
        ckpt_dir = None
        if c is not None:
            exp = Path(os.environ.get("VAE_EXPERIMENT_DIR", Path(__file__).resolve().parents[1]))
            ckpt_dir = exp / "checkpoints"
        return run_off_anchor_eval_spatial(
            model,
            val_loader,
            bg=bg,
            off_anchor_mhz=off_anchor_mhz,
            max_batches=max_batches,
            device=device,
            out_csv=out_csv,
            ckpt_dir=ckpt_dir,
            early_stop_state=_SPATIAL_EARLY_STOP,
            early_stop_mhz=float(getattr(c, "early_stop_encode_cross_mhz", 330.0)) if c else 330.0,
            early_stop_patience=int(getattr(c, "early_stop_encode_cross_patience", 3)) if c else 3,
        )

    _eval_cf.run_off_anchor_eval = _spatial_off_anchor_eval
    _eval_cf._exp045_spatial_patched = True


def _print_training_summary(c: _tr.Config) -> None:
    enc_pct = (1.0 - float(c.layout_train_prob)) * 100.0
    print(
        "  exp045 training (unbounded + encode-first + spatial):\n"
        f"    dataset unbounded z-score  clip_recon={c.heatmap_clip_recon}\n"
        f"    huber_delta={c.heatmap_huber_delta}  tail_z>{c.heatmap_tail_z_threshold} "
        f"boost×{c.heatmap_tail_weight_boost}\n"
        f"    spatial: pearson_w={c.heatmap_pearson_weight}  peak_loc_w={c.heatmap_peak_loc_weight}  "
        f"grad_w={c.heatmap_grad_weight}  lap_w={c.heatmap_lap_weight}\n"
        f"    encode_native_teacher_skips={getattr(c, 'encode_native_teacher_skips', True)}  "
        f"occ_tower={getattr(c, 'use_occ_spatial_tower', True)}\n"
        f"    layout_train_prob={c.layout_train_prob} (~{enc_pct:.0f}% encode batches)\n"
        f"    cross_freq_use_encode_z={c.cross_freq_use_encode_z}  "
        f"cross_freq_weight={c.cross_freq_weight} from ep {c.cross_freq_start_epoch}\n"
        f"    early_stop: encode_cross@{c.early_stop_encode_cross_mhz}MHz "
        f"patience={c.early_stop_encode_cross_patience} ckpts",
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
        os.environ.setdefault(
            "VAE_DATA_DIR",
            str(_PROJECT_ROOT / "datasets" / "data_multifreq_unbounded"),
        )
    train_vae()


if __name__ == "__main__":
    main()
