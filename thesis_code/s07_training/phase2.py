"""Second training phase: the placement-only encoder on the frozen VAE.

What: two steps, both on the training placements only (the VAE's own split).
    1. train_distiller: fit ``_OccToZ`` (one-hot placement + K -> latent) to the latent the full
       encoder assigns each placement from its 200 MHz map, placement and spectrum (smooth-L1, AdamW
       3e-4, cosine schedule, 20,000 steps of 256 on 16,000 placements). Checked on 2,000 test
       placements: latent cosine and median spectrum error.
    2. finetune: fine-tune that network and the VAE's spectrum decoder together (8,000 steps of 256,
       AdamW 1e-4, cosine schedule) on
           w_imp    smooth-L1 of the normalized log-spectrum decoded from the network's latent,
           w_hm     the 200 MHz map decoded from that latent vs the real map: MSE + (1 - Pearson)
                    over the board (needs a store with one 200 MHz map per placement, hm_ref_dir),
           w_anchor MSE between the latent and the starting network's latent, so frequencies the
                    loss never sees stay where the decoder was trained.
       Encoder, map and placement decoders, K embedding and placement encoder stay frozen.
    The thesis head (H1) is w_imp 1, w_hm 1, w_anchor 0.01, 8,000 steps.
Usage (import only; the entry point is train_phase2.py):
    train_distiller(steps=20000, n_train=16000, device=None, save=None) -> dict
    finetune(name, steps=8000, ..., save_dir=None, distill=None) -> dict   (writes <name>.pt)
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from thesis_code.common import settings
from thesis_code.common.board import board_mask, zero_offboard
from thesis_code.common.frequency import pi_freq_mhz_to_norm
from thesis_code.common.paths import need, out_dir
from thesis_code.common.placement import ensure_batch_onehot
from thesis_code.s04_dataset.dataset import VAEDataset
from thesis_code.s04_dataset.loader import _split_indices_by_design, precompute_multifreq_metadata
from thesis_code.s09_inference.inference import VAEInference
from thesis_code.s09_inference.placement_encoder import VAEImpedance, _OccToZ, load

PI_REF_MHZ = 200.0
SPLIT_SEED = 42
TRAIN_SPLIT = 0.9
N_TYPES = 3


def _stamp(p: Path) -> list:
    st = Path(p).stat()
    return [int(st.st_size), int(st.st_mtime)]


def train_distiller(*, steps: int = 20000, n_train: int = 16000, device=None, save: Path | None = None) -> dict:
    """Fit the placement -> latent network to the full encoder's latents (200 MHz rows); writes
    occ_to_z_distill.pt to ``save`` or OUTPUT_DIR/s07_training/phase2/."""
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = need(Path(settings.WEIGHTS_DIR) / settings.CHECKPOINT, "WEIGHTS_DIR / CHECKPOINT")
    vae = VAEInference(ckpt_path, device=device)
    model = vae.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ims = vae.norm_stats.impedance
    data_dir = need(Path(settings.DATA_DIR) / "normalized", "DATA_DIR")

    ds = VAEDataset(str(data_dir), precompute_k=True)
    meta = precompute_multifreq_metadata(ds)
    rows_of: dict = {}
    for i, d in enumerate(meta["design_keys"]):
        rows_of.setdefault(d, []).append(i)
    tr_i, va_i = _split_indices_by_design(meta["design_keys"], TRAIN_SPLIT, SPLIT_SEED)
    trs = {meta["design_keys"][i] for i in tr_i}
    vas = {meta["design_keys"][i] for i in va_i}
    ref_row = {meta["design_keys"][i]: i for i, m in enumerate(meta["freq_mhz"]) if abs(float(m) - PI_REF_MHZ) < 1e-6}
    if not all(d in ref_row for d in rows_of):
        raise SystemExit(f"{data_dir}: every placement needs a {PI_REF_MHZ:g} MHz map")
    pool = sorted(rows_of)
    rs = np.random.RandomState(SPLIT_SEED)
    tr_keys = list(rs.permutation([d for d in pool if d in trs])[:n_train])
    va_keys = list(rs.permutation([d for d in pool if d in vas])[:2000])

    def encode_rows(rows):
        """Full-encode latents (posterior means) of dataset rows, each at its own frequency."""
        outs = []
        for s in range(0, len(rows), 1024):
            b = rows[s:s + 1024]
            k = torch.stack([r["K"].reshape(-1)[0] for r in b]).long().to(device)
            pi = torch.stack([r["PI_freq"].reshape(-1)[0] for r in b]).float().to(device)
            occ = torch.stack([r["occupancy"].reshape(-1) for r in b]).long().to(device)
            with torch.inference_mode():
                outs.append(model.encode(zero_offboard(torch.stack([r["heatmap_norm"] for r in b]).float().to(device)),
                                         ensure_batch_onehot(occ, N_TYPES),
                                         torch.stack([r["impedance"] for r in b]).float().to(device), k, pi)[1].float())
        return torch.cat(outs)

    def pack(keys):
        """(codes, K, spectrum, target latent) per placement."""
        codes, ks, imps, zs = [], [], [], []
        for s in range(0, len(keys), 256):
            rows = [ds[ref_row[d]] for d in keys[s:s + 256]]
            z = encode_rows(rows)
            for o, r0 in enumerate(rows):
                codes.append(r0["occupancy"].reshape(-1).long())
                ks.append(r0["K"].reshape(-1)[0].long())
                imps.append(r0["impedance"].float())
                zs.append(z[o:o + 1].mean(0))
        return torch.stack(codes), torch.stack(ks), torch.stack(imps), torch.stack(zs)

    t0 = time.time()
    c_tr, k_tr, i_tr, z_tr = pack(tr_keys)
    c_va, k_va, i_va, z_va = pack(va_keys)
    oh_tr, kd_tr = ensure_batch_onehot(c_tr.to(device), N_TYPES), k_tr.to(device)
    torch.manual_seed(SPLIT_SEED)
    net = _OccToZ(N_TYPES, vae.latent_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-4 * 0.02)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(0, z_tr.shape[0], (256,), device=device)
            opt.zero_grad(set_to_none=True)
            F.smooth_l1_loss(net(oh_tr[idx], kd_tr[idx]), z_tr[idx]).backward()
            opt.step()
            sch.step()
    net.eval()
    k_range = (int(k_tr.min()), int(k_tr.max()))
    pred = VAEImpedance(model, net, N_TYPES, ims.log_mean, ims.log_std, k_range, device)
    with torch.inference_mode():
        cos = float(F.cosine_similarity(net(ensure_batch_onehot(c_va.to(device), N_TYPES), k_va.to(device)),
                                        z_va, dim=1).mean())
    real = np.exp(i_va.reshape(len(va_keys), -1).numpy() * ims.log_std + ims.log_mean)
    med = float(np.median(np.abs(pred.predict_ohm(c_va.numpy(), k_va.numpy()) - real) / real))
    print(f"  distiller: {time.time() - t0:.0f}s, test latent cosine {cos:.4f}, median |err| {med:.1%}")
    path = Path(save) if save is not None else out_dir("s07_training") / "phase2" / "occ_to_z_distill.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
           "n_types": N_TYPES, "data_dir": str(data_dir), "split_seed": SPLIT_SEED,
           "vae_ckpt": settings.CHECKPOINT, "vae_stamp": _stamp(ckpt_path), "k_range": list(k_range),
           "val_cos": cos, "val_med_rel": med, "steps": steps, "n_train": len(tr_keys),
           "target": "full_encode_200MHz"}
    torch.save(out, path)
    return {**{k: v for k, v in out.items() if k != "state_dict"}, "path": str(path)}


def finetune(name: str, *, steps: int = 8000, batch: int = 256, lr: float = 1e-4, w_imp: float = 1.0,
             w_hm: float = 1.0, w_anchor: float = 0.01, hm_ref_dir: str | Path | None = None,
             seed: int | None = None, device=None, save_dir: Path | None = None,
             distill: Path | None = None) -> dict:
    """Fine-tune the placement -> latent network and the spectrum decoder (see the module docstring),
    starting from the distiller ``distill`` (default WEIGHTS_DIR/occ_to_z_distill.pt); writes
    <name>.pt to ``save_dir`` or OUTPUT_DIR/s07_training/phase2/. ``hm_ref_dir`` defaults to the
    dataset's normalized store (one 200 MHz map per placement)."""
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = load(device=device, head=None, distill=distill)
    model = base.model
    store = need(Path(settings.DATA_DIR) / "normalized", "DATA_DIR")
    with (store / "manifest.csv").open() as fh:
        designs = sorted({r["design_id"] for r in csv.DictReader(fh)})
    tr_i, va_i = _split_indices_by_design(designs, TRAIN_SPLIT, SPLIT_SEED)
    tr, va = [designs[i] for i in tr_i], [designs[i] for i in va_i]

    hm_of: dict[str, Path] = {}
    if w_hm > 0:
        ref = Path(hm_ref_dir) if hm_ref_dir is not None else store
        with (ref / "manifest.csv").open() as fh:
            for r in csv.DictReader(fh):
                if abs(float(r["freq_mhz"]) - PI_REF_MHZ) < 1e-6:
                    hm_of[r["design_id"]] = ref / "heatmap" / r["sample_name"]
        tr = [d for d in tr if d in hm_of]
        va = [d for d in va if d in hm_of]

    def read(keys):
        codes = torch.tensor(np.stack([np.load(store / "layouts" / d / "type.npy").astype(np.int64).reshape(-1)
                                       for d in keys]))
        imp = torch.tensor(np.stack([np.load(store / "layouts" / d / "imp.npy").reshape(1, -1) for d in keys]),
                           dtype=torch.float32)
        hm = (torch.tensor(np.stack([np.load(hm_of[d]).reshape(1, 64, 64) for d in keys]), dtype=torch.float32)
              if w_hm > 0 else None)
        return codes, (codes != 0).sum(1), imp, hm

    print(f"  fine-tune '{name}': {len(tr):,} train / {len(va):,} test placements, "
          f"w_imp={w_imp} w_hm={w_hm} w_anchor={w_anchor}, {steps} steps")
    c_tr, k_tr, i_tr, h_tr = read(tr)
    c_va, k_va, i_va, h_va = read(va)
    oh_tr, oh_va = ensure_batch_onehot(c_tr.to(device), N_TYPES), ensure_batch_onehot(c_va.to(device), N_TYPES)
    k_tr, k_va, i_tr, i_va = k_tr.to(device), k_va.to(device), i_tr.to(device), i_va.to(device)
    if h_tr is not None:
        h_tr, h_va = h_tr.to(device), h_va.to(device)
    fg = board_mask(device, torch.bool)[0, 0]
    pi_ref = pi_freq_mhz_to_norm(PI_REF_MHZ)
    seed = SPLIT_SEED if seed is None else int(seed)

    net0 = copy.deepcopy(base.net).eval()
    net = copy.deepcopy(base.net).train()
    dec = copy.deepcopy(model.impedance_decoder).train()
    for p in list(net.parameters()) + list(dec.parameters()):
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(net.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    def imp_head(z, oh, k, decoder):
        return decoder(torch.cat([model._layout_latent_slice(z), model.k_embedding(k), model.occupancy_encoder(oh)], 1))

    def hm_200(z, oh, k):
        pi = torch.full((z.shape[0],), pi_ref, device=device)
        return zero_offboard(model.decode(z, k, pi, occupancy=oh)[0])

    def board_pearson(pred, real):
        m = fg.expand_as(pred)
        p, r = pred[m].view(pred.shape[0], -1), real[m].view(real.shape[0], -1)
        pc, rc = p - p.mean(1, keepdim=True), r - r.mean(1, keepdim=True)
        return p, r, (pc * rc).sum(1) / (pc.norm(dim=1) * rc.norm(dim=1) + 1e-8)

    def hm_loss(pred, real):
        p, r, pear = board_pearson(pred, real)
        return F.mse_loss(p, r) + (1 - pear).mean()

    real_hm_tr = zero_offboard(h_tr) if h_tr is not None else None
    t0 = time.time()
    torch.manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, oh_tr.shape[0], (batch,), device=device)
        oh, k = oh_tr[idx], k_tr[idx]
        z = net(oh, k)
        loss = w_imp * F.smooth_l1_loss(imp_head(z, oh, k, dec).reshape(batch, -1), i_tr[idx].reshape(batch, -1))
        with torch.no_grad():
            z0 = net0(oh, k)
        loss = loss + w_anchor * F.mse_loss(z, z0)
        if w_hm > 0:
            loss = loss + w_hm * hm_loss(hm_200(z, oh, k), real_hm_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if (step + 1) % 2000 == 0:
            print(f"    step {step + 1}/{steps}  loss {float(loss):.4f}  ({time.time() - t0:.0f}s)")
    net.eval(), dec.eval()

    def evaluate(network, decoder):
        out = {}
        with torch.inference_mode():
            y = torch.cat([imp_head(network(oh_va[s:s + 2048], k_va[s:s + 2048]), oh_va[s:s + 2048],
                                    k_va[s:s + 2048], decoder).reshape(len(oh_va[s:s + 2048]), -1)
                           for s in range(0, oh_va.shape[0], 2048)])
            real = torch.exp(i_va.reshape(y.shape[0], -1) * base.log_std + base.log_mean)
            pred = torch.exp(y * base.log_std + base.log_mean)
            out["spec_med_rel"] = float(((pred - real).abs() / real).median(1).values.median())
            if h_va is not None:
                rs = [board_pearson(hm_200(network(oh_va[s:s + 512], k_va[s:s + 512]), oh_va[s:s + 512], k_va[s:s + 512]),
                                    zero_offboard(h_va[s:s + 512]))[2] for s in range(0, oh_va.shape[0], 512)]
                out["hm200_pearson_mean_normspace"] = float(torch.cat(rs).mean())
        return out

    before = evaluate(net0, model.impedance_decoder)
    after = evaluate(net, dec)
    save = (Path(save_dir) if save_dir is not None else out_dir("s07_training") / "phase2") / f"{name}.pt"
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"impedance_decoder": {k: v.cpu() for k, v in dec.state_dict().items()},
                "network": {k: v.cpu() for k, v in net.state_dict().items()},
                "vae_ckpt": settings.CHECKPOINT, "vae_stamp": _stamp(Path(settings.WEIGHTS_DIR) / settings.CHECKPOINT),
                "losses": {"w_imp": w_imp, "w_hm": w_hm, "w_anchor": w_anchor}, "steps": steps,
                "hm_ref_dir": str(hm_ref_dir) if hm_ref_dir is not None else str(store),
                "heldout_before": before, "heldout_after": after, "n_train": len(tr), "n_heldout": len(va),
                "seed": seed}, save)
    print(f"  test placements before {before}  after {after}  ({time.time() - t0:.0f}s) -> {save}")
    return {"before": before, "after": after, "path": str(save)}
