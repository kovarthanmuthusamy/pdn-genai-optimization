"""Train VAE for exp043 — global-max heatmap norm + PI_freq PoE expert.

Same curriculum as exp042; heatmaps are Ω / global_max (not log-z).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp043.codes.gmax_training_patch import apply_gmax_config, patch_trainer
from experiments.exp043.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp043.codes.vae_poe_freq import MultiInputVAEPoeFreq

_orig_prepare = _tr._prepare_batch


@dataclass
class Config(_tr.Config):
    synthetic_blend_prob: float = 0.0
    use_freq_poe_expert: bool = True
    use_global_max_heatmap: bool = False
    global_max_ohm: float = 1.0
    heatmap_fg_threshold: float = 0.01


def build_vae_model(c: Config) -> MultiInputVAEPoeFreq:
    model = MultiInputVAEPoeFreq(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=c.use_heatmap_film,
        use_freq_poe_expert=bool(getattr(c, "use_freq_poe_expert", True)),
    )
    print(
        f"  exp043 model: global-max heatmap + freq PoE expert "
        f"(d={c.heatmap_private_dim}, poe={getattr(c, 'use_freq_poe_expert', True)})",
    )
    return model


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


def _on_stats_loaded(c: Config, raw: dict) -> None:
    apply_gmax_config(c, raw)
    patch_trainer(_tr, c)


def _patch_training() -> None:
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._on_stats_loaded = _on_stats_loaded


def _print_layout_sweep_training(c: _tr.Config) -> None:
    print(
        "  exp043 sweep training path:\n"
        f"    layout_train_prob={c.layout_train_prob} "
        f"(decode z from encode_layout_latent + freq PoE)\n"
        f"    cross_freq_weight={c.cross_freq_weight} "
        f"(cf loss, layout z only={c.cross_freq_layout_z_only}) "
        f"from epoch {c.cross_freq_start_epoch}+\n"
        f"    use_freq_poe_expert={getattr(c, 'use_freq_poe_expert', True)} "
        f"(PI_freq expert in PoE on private dims; occ/imp stay K-only)",
    )


def train_vae() -> None:
    _patch_training()
    c = Config()
    _tr._apply_yaml_config(c)
    _print_layout_sweep_training(c)
    _tr.train_vae()


def main() -> None:
    exp_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(exp_dir))
    cfg_path = exp_dir / "config.yaml"
    if cfg_path.is_file():
        import json
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
            str(_PROJECT_ROOT / "datasets" / "data_multifreq_gmax"),
        )
    train_vae()


if __name__ == "__main__":
    main()
