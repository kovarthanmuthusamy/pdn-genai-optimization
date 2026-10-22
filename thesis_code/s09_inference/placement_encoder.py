"""The placement-only encoder: typed placement -> latent vector -> spectrum in ohm, through the VAE.

What: the VAE needs a map and a spectrum to encode a placement; at design time only the placement
    exists. A small network (``_OccToZ``: one-hot placement + K -> 128-dim latent) is fitted to the
    latent the full encoder assigns each training placement (the distiller, occ_to_z_distill.pt),
    and the VAE's own spectrum decoder reads that latent together with the typed placement -- the
    inputs it always had in training. The second training phase then fine-tunes the network and
    the spectrum decoder (the head, vae_route_finetune/<HEAD>.pt); encoder, map and placement
    decoders stay as trained. A head is valid only on the VAE checkpoint it was trained on.
    K outside the trained range 1..10 is refused, not extrapolated.
    A placement passes a target when its predicted worst-case ratio max_f Z(f) / Z_target(f) is at
    most 1 - m; m is the smallest margin with zero false passes on the test placements, measured
    per target (<HEAD>_margin.json, written by s09_inference/margin.py).
Usage (import only):
    enc = load(device=None, head=settings.HEAD, distill=None)    head None = distiller only
    enc.predict_ohm(codes) -> (N, 231) ohm;  enc.net(onehot, k) -> latent;  enc.model -> the VAE
    freeze(enc)                                    no gradients into its weights (the search)
    margin(target_path) -> zero-false-pass margin for that target spectrum
    check_head_matches(head_ck, vae_path)          stop if the head was trained on other weights
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import need
from thesis_code.common.placement import ensure_batch_onehot
from thesis_code.s09_inference.inference import VAEInference

N_SLOTS = 52
FINETUNE_DIR = "vae_route_finetune"


class _OccToZ(nn.Module):
    def __init__(self, n_types: int, zdim: int, hidden: int = 1024):
        super().__init__()
        self.k = nn.Embedding(64, 64)
        self.net = nn.Sequential(
            nn.Linear(N_SLOTS * n_types + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, zdim))

    def forward(self, onehot: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([onehot.flatten(1), self.k(k.long().clamp(0, 63).reshape(-1))], 1))


class VAEImpedance:
    """Placement-only encoder + frozen VAE spectrum decoder, in ohm."""

    def __init__(self, model, net: _OccToZ, n_types: int, log_mean: float, log_std: float,
                 k_range: tuple[int, int], device: torch.device):
        self.model = model.eval()
        self.net = net.eval()
        self.n_types = int(n_types)
        self.log_mean = float(log_mean)
        self.log_std = float(log_std) or 1.0
        self.k_range = (int(k_range[0]), int(k_range[1]))
        self.device = device
        self.head: str | None = None

    def _impedance_norm(self, onehot: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """The decoder's spectrum branch with the typed placement as context."""
        m = self.model
        z = self.net(onehot, k)
        return m.impedance_decoder(torch.cat([m._layout_latent_slice(z), m.k_embedding(k), m.occupancy_encoder(onehot)], 1))

    def predict_ohm(self, codes, k=None, batch: int = 8192) -> np.ndarray:
        """Type codes (52,) or (B, 52) -> spectrum in ohm; K = number of decaps if omitted."""
        c = torch.as_tensor(np.asarray(codes), dtype=torch.long).reshape(-1, N_SLOTS).to(self.device)
        kk = (torch.as_tensor(np.asarray(k), dtype=torch.long).reshape(-1).to(self.device)
              if k is not None else (c != 0).sum(1))
        lo, hi = self.k_range
        bad = (kk < lo) | (kk > hi)
        if bool(bad.any()):
            raise ValueError(f"{int(bad.sum())} placement(s) have K outside the trained range "
                             f"{lo}..{hi} (e.g. K={int(kk[bad][0])})")
        out = []
        with torch.inference_mode():
            for s in range(0, c.shape[0], batch):
                y = self._impedance_norm(ensure_batch_onehot(c[s:s + batch], self.n_types), kk[s:s + batch])
                out.append(np.exp(y.float().cpu().numpy() * self.log_std + self.log_mean))
        a = np.concatenate(out, 0)
        return a[0] if np.asarray(codes).ndim == 1 else a


def freeze(pred: VAEImpedance) -> None:
    """No gradients into the encoder's or the VAE's weights: the search differentiates through them only."""
    for prm in list(pred.model.parameters()) + list(pred.net.parameters()):
        prm.requires_grad_(False)


def check_head_matches(head_ck: dict, vae_path: Path) -> None:
    """Stop when the head was trained on a VAE file of another size. A copied file keeps its size but
    gets a new modification time; that only prints a note."""
    stamp = head_ck.get("vae_stamp")
    if not stamp:
        return
    st = Path(vae_path).stat()
    if int(st.st_size) != int(stamp[0]):
        raise SystemExit(f"the head was trained on another {Path(vae_path).name} ({stamp[0]} bytes, this one "
                         f"{st.st_size}) -- train the head again for these weights")
    if int(st.st_mtime) != int(stamp[1]):
        print(f"note: {vae_path} has a new modification time; size matches, loading")


def _apply_head(pred: VAEImpedance, head: str) -> VAEImpedance:
    """Swap in the fine-tuned spectrum decoder and network (the model object is this encoder's own copy)."""
    path = need(Path(settings.WEIGHTS_DIR) / FINETUNE_DIR / f"{head}.pt", "WEIGHTS_DIR / HEAD")
    ck = torch.load(path, map_location=pred.device, weights_only=False)
    check_head_matches(ck, Path(settings.WEIGHTS_DIR) / ck["vae_ckpt"])
    dec = copy.deepcopy(pred.model.impedance_decoder)
    dec.load_state_dict(ck["impedance_decoder"])
    pred.model.impedance_decoder = dec.eval()
    net = copy.deepcopy(pred.net)
    net.load_state_dict(ck["network"])
    pred.net = net.eval()
    pred.head = head
    return pred


def load(device=None, head: str | None = "default", distill: Path | None = None) -> VAEImpedance:
    """The encoder on the frozen VAE: distiller (+ the fine-tuned head; "default" = settings.HEAD).
    ``distill``: another distiller file than WEIGHTS_DIR/occ_to_z_distill.pt (a new phase-2 run)."""
    head = settings.HEAD if head == "default" else head
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = need(Path(settings.WEIGHTS_DIR), "WEIGHTS_DIR")
    distill = need(Path(distill) if distill is not None else weights / "occ_to_z_distill.pt",
                   "WEIGHTS_DIR (occ_to_z_distill.pt)")
    vae = VAEInference(weights / settings.CHECKPOINT, device=device)
    model = vae.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ck = torch.load(distill, map_location="cpu", weights_only=False)
    if ck.get("vae_ckpt") != settings.CHECKPOINT:
        raise SystemExit(f"{distill} was fitted to {ck.get('vae_ckpt')}, not {settings.CHECKPOINT}")
    check_head_matches(ck, weights / settings.CHECKPOINT)
    net = _OccToZ(int(ck["n_types"]), vae.latent_dim).to(device)
    net.load_state_dict(ck["state_dict"])
    ims = vae.norm_stats.impedance
    pred = VAEImpedance(model, net, int(ck["n_types"]), ims.log_mean, ims.log_std, tuple(ck["k_range"]), device)
    return _apply_head(pred, head) if head else pred


def _saved_target(name: str, path: str) -> Path:
    """The target spectrum a margin was measured for: its recorded file, else the shipped copy
    (target name "0.6" -> common/inputs/target_impedance_0.6.npy)."""
    p = settings.REPO / path
    if p.is_file():
        return p
    return INPUTS / f"target_impedance_{re.search(r'[0-9]+[.][0-9]+', name).group(0)}.npy"


def margin(target_path: Path) -> float:
    """Zero-false-pass margin of the head for the target spectrum in ``target_path`` (matched by content)."""
    f = need(Path(settings.WEIGHTS_DIR) / FINETUNE_DIR / f"{settings.HEAD}_margin.json", "WEIGHTS_DIR / HEAD margin")
    want = np.load(target_path).reshape(-1)
    for name, t in json.loads(f.read_text())["targets"].items():
        if np.allclose(np.load(_saved_target(name, t["path"])).reshape(-1), want):
            if t["margin"] is None:
                raise SystemExit(f"{f}: no margin up to 0.60 gives zero false passes for target {name}")
            return float(t["margin"])
    raise SystemExit(f"no margin measured for target {target_path} in {f}: run thesis_code.s09_inference.margin")
