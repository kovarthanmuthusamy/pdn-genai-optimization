"""Per-K mean and covariance of the latent vectors of the training placements (latent_cov_per_k.json).

What: every training placement is encoded with the full encoder (its 200 MHz map, spectrum and typed
    placement; latent mean). Per decap budget K the mean and the full covariance of these latents are
    stored (at most 6,000 placements per K, drawn with seed 42). The search draws its starting points
    from N(mu_K, Sigma_K) and keeps the latent vector inside a Mahalanobis radius under Sigma_K: the
    latents are far from independent per dimension (at fixed K, 95 % of the variance lies in about
    10 of 128 directions).
Usage: python -m thesis_code.s10_search.latent_stats
    Writes OUTPUT_DIR/s10_search/latent_cov_per_k.json (+ .meta.json); the search reads the copy in
    WEIGHTS_DIR.
Config keys: K_VALUES, MAX_PER_K, SEED.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from thesis_code.common import settings
from thesis_code.common.board import zero_offboard
from thesis_code.common.frequency import pi_freq_mhz_to_norm
from thesis_code.common.paths import need, out_dir
from thesis_code.common.placement import ensure_batch_onehot
from thesis_code.s04_dataset.dataset import VAEDataset
from thesis_code.s04_dataset.loader import _split_indices_by_design, precompute_multifreq_metadata
from thesis_code.s09_inference.inference import VAEInference

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MAX_PER_K = 6000
SEED = 42
# =============================================================================


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck_path = need(Path(settings.WEIGHTS_DIR) / settings.CHECKPOINT, "WEIGHTS_DIR / CHECKPOINT")
    model = VAEInference(ck_path, device=device).model.eval()
    data_dir = need(Path(settings.DATA_DIR) / "normalized", "DATA_DIR")
    ds = VAEDataset(str(data_dir), precompute_k=True)
    meta = precompute_multifreq_metadata(ds)
    at200 = {d: i for i, (d, f) in enumerate(zip(meta["design_keys"], meta["freq_mhz"])) if abs(f - 200.0) < 1e-6}
    if len(at200) != len(set(meta["design_keys"])):
        raise SystemExit(f"{data_dir}: every placement needs a 200 MHz map")
    tr_i, _ = _split_indices_by_design(meta["design_keys"], 0.9, 42)
    train = {meta["design_keys"][i] for i in tr_i}
    keys = sorted(d for d in at200 if d in train)

    t0 = time.time()
    rows = [ds[at200[d]] for d in keys]
    k_all = np.array([int(r["K"].reshape(-1)[0]) for r in rows])
    print(f"loaded {len(keys):,} training placements ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    pi = pi_freq_mhz_to_norm(200.0)
    result, info = {}, {}
    for K in K_VALUES:
        idx = np.flatnonzero(k_all == K)
        if len(idx) > MAX_PER_K:
            idx = rng.choice(idx, MAX_PER_K, replace=False)
        if len(idx) < 2:
            continue
        zs = []
        for s in range(0, len(idx), 1024):
            b = [rows[i] for i in idx[s:s + 1024]]
            occ = ensure_batch_onehot(torch.stack([r["occupancy"] for r in b]).to(device), 3)
            k = torch.stack([r["K"] for r in b]).to(device)
            imp = torch.stack([r["impedance"] for r in b]).to(device)
            hm = torch.stack([r["heatmap_norm"] for r in b]).to(device)
            with torch.inference_mode():
                zs.append(model.encode(zero_offboard(hm), occ, imp, k,
                                       torch.full((len(b),), pi, device=device))[1].double().cpu())
        Z = torch.cat(zs).numpy()
        mu, cov = Z.mean(0), np.cov(Z, rowvar=False)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(min=0)
        d95 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        result[str(K)] = {"mu": mu.tolist(), "cov": cov.tolist(), "n": int(len(idx))}
        info[str(K)] = {"n": int(len(idx)), "dims_95pct_var": d95,
                        "mean_norm": float(np.linalg.norm(Z, axis=1).mean()), "trace": float(np.trace(cov))}
        print(f"K={K:>2}  n={len(idx):>5}  95 % of the variance in {d95} directions")
    out = out_dir("s10_search") / "latent_cov_per_k.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    out.with_name("latent_cov_per_k.meta.json").write_text(json.dumps({
        "checkpoint": settings.CHECKPOINT, "data_dir": str(data_dir), "boards": "training placements (seed 42 split)",
        "encode": "full (200 MHz map + spectrum + typed placement), latent mean", "per_k": info,
        "created": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
