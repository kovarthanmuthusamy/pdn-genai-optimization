"""Shared paths, model loading, and PI_freq helpers for exp045 evaluation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.exp046.codes.vae_poe_freq import MultiInputVAEPoeFreq  # noqa: E402
from src_vae.others.dataloader import VAEDataset  # noqa: E402
from src_vae.others.multifreq_anchors import load_anchors_mhz  # noqa: E402
from src_vae.others.pi_freq_utils import (  # noqa: E402
    _LOG10_MIN,
    _LOG10_RANGE,
    pi_freq_mhz_to_norm,
)

EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _ROOT
DEFAULT_CKPT = EXP_DIR / "checkpoints/last_model.pt"
ANCHOR_MHZ: tuple[float, ...] = load_anchors_mhz()


def load_exp_config() -> dict[str, Any]:
    path = EXP_DIR / "config.yaml"
    if not path.is_file():
        return {}
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def resolve_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path | str]:
    cfg = cfg or load_exp_config()
    data_dir = Path(cfg.get("data_dir", PROJECT_ROOT / "data_multi_norm"))
    ckpt = Path(cfg.get("checkpoint_path", DEFAULT_CKPT))
    if not ckpt.is_file():
        ep = cfg.get("resume_checkpoint")
        if ep is not None:
            alt = EXP_DIR / "checkpoints" / f"checkpoint_epoch_{int(ep)}.pt"
            if alt.is_file():
                ckpt = alt
    return {
        "exp_dir": EXP_DIR,
        "data_dir": data_dir,
        "checkpoint": ckpt,
        "norm_stats": data_dir / "normalization_stats.json",
        "binary_mask": PROJECT_ROOT / "configs/binary_mask.npy",
        "freq_hz": PROJECT_ROOT / "configs/Frequency_data_hz.npy",
        "target_imp": PROJECT_ROOT / "configs/target_impedance.npy",
        "background_value": float(cfg.get("background_value", -2.9669)),
        "eval_off_anchor_mhz": tuple(cfg.get("eval_off_anchor_mhz", (80.0, 250.0))),
    }


def pi_norm_to_mhz(pi_norm: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(pi_norm, dtype=np.float64)
    log10_hz = arr * _LOG10_RANGE + _LOG10_MIN
    mhz = (10.0 ** log10_hz) / 1e6
    if np.ndim(pi_norm) == 0:
        return float(mhz)
    return mhz.astype(np.float32)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[MultiInputVAEPoeFreq, dict[str, Any], dict | None, dict | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    ld = int(cfg.get("latent_dim", 42))
    cond = int(cfg.get("cond_dim", 8))
    hm_priv = int(cfg.get("heatmap_private_dim", 8))
    model = MultiInputVAEPoeFreq(
        latent_dim=ld,
        cond_dim=cond,
        heatmap_private_dim=hm_priv,
        modality_dropout=float(cfg.get("modality_dropout", 0.0)),
        freq_fourier_features=int(cfg.get("freq_fourier_features", 8)),
        use_heatmap_film=bool(cfg.get("use_heatmap_film", True)),
        use_freq_poe_expert=bool(cfg.get("use_freq_poe_expert", True)),
        use_heatmap_unet_skips=bool(cfg.get("use_heatmap_unet_skips", True)),
        use_occ_spatial_decoder=bool(cfg.get("use_occ_spatial_decoder", True)),
        use_occ_spatial_tower=bool(cfg.get("use_occ_spatial_tower", True)),
        occ_spatial_ch=int(cfg.get("occ_spatial_ch", 8)),
        use_layout_private_head=bool(cfg.get("use_layout_private_head", True)),
    )
    state = ckpt["model_state_dict"]
    ms = model.state_dict()
    compat = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    model.to(device).eval()
    print(f"Loaded {checkpoint_path}  latent_dim={ld}  ({len(compat)}/{len(state)} tensors)")
    return model, cfg, ckpt.get("latent_stats"), ckpt.get("per_K_latent_stats")


def encode_dataset(
    model: MultiInputVAEPoeFreq,
    data_dir: str | Path,
    device: torch.device,
    *,
    max_samples: int = 30_000,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    collect_experts: bool = True,
) -> dict[str, np.ndarray]:
    ds = VAEDataset(data_dir=str(data_dir))
    if not ds.has_pifreq:
        raise ValueError(f"{data_dir} must contain PI_freq/ for multifreq evaluation")
    n = min(len(ds), max_samples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), n, replace=False).tolist()
    sub = Subset(ds, idx)
    dl = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_mu, all_logvar, all_k, all_pi = [], [], [], []
    expert_mu = {"heatmap": [], "occupancy": [], "impedance": []}
    expert_lv = {"heatmap": [], "occupancy": [], "impedance": []}

    with torch.no_grad():
        for batch in dl:
            hm = batch["heatmap_norm"].to(device)
            occ = batch["occupancy"].to(device)
            imp = batch["impedance"].to(device)
            k = batch["K"].to(device)
            pi = batch["PI_freq"].to(device)
            if imp.dim() == 2:
                imp = imp.unsqueeze(1)
            _, mu, logvar, expert_stats = model.encode(hm, occ, imp, k, pi)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_k.append(k.cpu().numpy())
            all_pi.append(pi.cpu().numpy())
            if collect_experts:
                for m in ("heatmap", "occupancy", "impedance"):
                    m_mu, m_lv = expert_stats[m]
                    expert_mu[m].append(m_mu.cpu().numpy())
                    expert_lv[m].append(m_lv.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    k_arr = np.concatenate(all_k, axis=0)
    pi_arr = np.concatenate(all_pi, axis=0)
    mhz_arr = pi_norm_to_mhz(pi_arr)
    sigma = np.exp(0.5 * logvar)

    experts = None
    if collect_experts:
        experts = {}
        for m in ("heatmap", "occupancy", "impedance"):
            m_mu = np.concatenate(expert_mu[m], axis=0)
            m_lv = np.concatenate(expert_lv[m], axis=0)
            experts[m] = {"mu": m_mu, "logvar": m_lv, "sigma": np.exp(0.5 * m_lv)}

    print(f"Encoded {len(mu)} / {len(ds)} samples  K=[{k_arr.min():.0f},{k_arr.max():.0f}]  "
          f"MHz=[{mhz_arr.min():.0f},{mhz_arr.max():.0f}]")
    return {
        "mu": mu,
        "logvar": logvar,
        "sigma": sigma,
        "K": k_arr,
        "pi_norm": pi_arr,
        "mhz": mhz_arr,
        "experts": experts,
    }


def nearest_anchor_index(mhz: float) -> int:
    return int(np.argmin([abs(mhz - a) for a in ANCHOR_MHZ]))


def pi_tensor_mhz(mhz: float, batch: int, device: torch.device) -> torch.Tensor:
    v = pi_freq_mhz_to_norm(mhz)
    return torch.full((batch,), v, dtype=torch.float32, device=device)
