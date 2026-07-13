"""Train VAE for exp044 — exp042 freq PoE + encode-first latent (latent-opt path).

Same dataset/norm as exp042 (``data_multifreq_norm_z_score`` log1p z-score). Training emphasizes
Path A: ``z = PoE(heatmap_encoder(GT), occ, imp, freq)`` for decode + cross-freq loss.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp044.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp044.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp044.codes.vae_poe_freq import MultiInputVAEPoeFreq

_orig_prepare = _tr._prepare_batch


@dataclass
class Config(_tr.Config):
    synthetic_blend_prob: float = 0.0
    use_freq_poe_expert: bool = True
    # exp044: cross-freq always decodes from encode z (heatmap in latent).
    cross_freq_use_encode_z: bool = True
    # Encode-first decode (latent opt). Layout path kept for occasional sweep aux.
    layout_train_prob: float = 0.15


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
        f"  exp044 model: freq PoE on private d={c.heatmap_private_dim} "
        f"(enabled={getattr(c, 'use_freq_poe_expert', True)})",
    )
    return model


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


def _patch_training() -> None:
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._run_epoch = _run_epoch_encode


def _print_encode_first_training(c: _tr.Config) -> None:
    enc_pct = (1.0 - float(c.layout_train_prob)) * 100.0
    print(
        "  exp044 encode-first training (latent-opt aligned):\n"
        f"    layout_train_prob={c.layout_train_prob} "
        f"(~{enc_pct:.0f}% batches: z includes GT heatmap via encoder)\n"
        f"    cross_freq_use_encode_z={getattr(c, 'cross_freq_use_encode_z', True)} "
        f"(cf loss always uses encode z, not layout z)\n"
        f"    cross_freq_layout_z_only={c.cross_freq_layout_z_only} "
        f"(ignored when cross_freq_use_encode_z=True)\n"
        f"    cross_freq_weight={c.cross_freq_weight} from epoch {c.cross_freq_start_epoch}+\n"
        f"    use_freq_poe_expert={getattr(c, 'use_freq_poe_expert', True)}",
    )


def train_vae() -> None:
    _patch_training()
    c = Config()
    _tr._apply_yaml_config(c)
    _print_encode_first_training(c)
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
            str(_PROJECT_ROOT / "datasets" / "data_multifreq_norm_z_score"),
        )
    train_vae()


if __name__ == "__main__":
    main()
