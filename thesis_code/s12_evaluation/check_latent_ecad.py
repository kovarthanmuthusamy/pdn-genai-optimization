"""Inside the latent space: what the latent vector is built from, and a walk between two placements.

What:
  1. What the latent vector is built from. For the test placements encoded with all views (map at
     200 MHz, placement, spectrum), each encoder's share of the fused precision,
     1/sigma_e^2 / (sum_e 1/sigma_e^2 + 1), averaged over placements and per latent number: which
     view the latent vector comes from when all are given.
  2. Map of the latent space. The latent vectors of the test placements (a) with all views and (b)
     from the placement alone, on the two principal directions of (a); rank correlation of the first
     direction with the budget and with the simulated worst-case ratio; where the latent vectors
     returned by the search (label s0 of ch4_search_runs.json) lie, in units of the spread of the
     data per latent number.
  3. A walk through the latent space, checked with eCADSTAR. The search relies on the latent space
     being continuous: a small step must change the decoded placement and its spectrum only a little,
     and the decoded spectrum must stay the spectrum of the decoded placement. For one pair of test
     placements per budget in WALK_K (one that passes the 0.8 ohm target, one that fails), the latent
     vector is moved in N_STEPS equal steps from the first to the second. At every step the placement
     is read out as the search does and its spectrum is decoded from the latent vector; the read-out
     placement is simulated. Measures: decaps changed from one step to the next; error of the decoded
     spectrum against the simulation of the read-out placement (typical point error, error of the
     worst-case ratio), on the way and at the two ends; and the same for the prediction from the
     read-out placement alone. Null: the error on the way is no larger than at the ends, where the
     latent vector belongs to a real placement.
Usage: python -m thesis_code.s12_evaluation.check_latent_ecad     (CONFIG-only; GPU recommended; about
       1 min of eCADSTAR when the store is new; resumable)
Outputs: store latent_walk_ecad/ (when new); ch4_latent.json (+ .npz)
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
WALK_K = (4, 5, 6, 8, 10)        # one walk per budget
N_STEPS = 10                     # 11 latent vectors per walk, the two ends included
WALK_TARGET = "0.8"
SEED = 20261011
SEARCH_LABEL = "s0"
STORE = "latent_walk_ecad"
# =============================================================================


def main() -> None:
    import torch
    from scipy.stats import spearmanr
    from thesis_code.common.board import zero_offboard
    from thesis_code.common.frequency import pi_freq_mhz_to_norm
    from thesis_code.common.placement import ensure_batch_onehot
    rng = np.random.default_rng(SEED)
    tg = c.targets()
    sur = c.Surrogate()
    dev, model = sur.dev, sur.vae.model.eval()
    test = c.split_ids("test")
    maps_of = c.maps_index()
    codes, spec = c.codes_of(test), c.spectra_of(test)
    K = (codes > 0).sum(1)
    res: dict = {}
    arrays: dict = {"K": K}

    # ---- 1. and 2.: encode with all views ---------------------------------------------------------
    mus, shares = [], []
    for s in range(0, len(test), 256):
        ds = test[s:s + 256]
        hm = np.stack([np.load(c.data() / "normalized" / "heatmap" / next(n for fq, n in maps_of[d] if abs(fq - c.F0_MHZ) < 1e-6)).reshape(1, 64, 64) for d in ds])
        imp = np.stack([np.load(c.data() / "normalized" / "layouts" / d / "imp.npy").reshape(1, -1) for d in ds])
        cd = torch.as_tensor(codes[s:s + 256], device=dev)
        occ, kt = ensure_batch_onehot(cd, 3), (cd > 0).sum(1).long()
        pi0 = torch.full((len(ds),), pi_freq_mhz_to_norm(c.F0_MHZ), device=dev)
        with torch.inference_mode():
            _, mu, _, ex = model.encode(zero_offboard(torch.as_tensor(hm, device=dev, dtype=torch.float32)), occ,
                                        torch.as_tensor(imp, device=dev, dtype=torch.float32), kt, pi0)
            prec = torch.stack([torch.exp(-ex[e][1]) for e in ("heatmap", "occupancy", "impedance")])
            tot = prec.sum(0) + 1.0
            shares.append(torch.cat([prec / tot, (1.0 / tot).unsqueeze(0)]).cpu().numpy())
            mus.append(mu.cpu().numpy())
    mu = np.concatenate(mus)
    sh = np.concatenate(shares, axis=1)
    names = ("map", "placement", "spectrum", "prior")
    dim_share = sh.mean(1)
    res["encoder_share"] = {"mean": {n: float(sh[i].mean()) for i, n in enumerate(names)},
                            "numbers_led_by": {n: int((dim_share.argmax(0) == i).sum()) for i, n in enumerate(names)}}
    zpl = np.concatenate([sur.latent(codes[s:s + 1024])[0].cpu().numpy() for s in range(0, len(codes), 1024)])
    mc = mu - mu.mean(0)
    _, sv, vt = np.linalg.svd(mc, full_matrices=False)
    pcs, pcs_pl = mc @ vt[:2].T, (zpl - mu.mean(0)) @ vt[:2].T
    rho = c.worst_ratio(spec, tg[WALK_TARGET])
    cos = (mu * zpl).sum(1) / (np.linalg.norm(mu, axis=1) * np.linalg.norm(zpl, axis=1))
    res["latent_map"] = {"variation_in_first_two_pct": float((sv[:2] ** 2).sum() / (sv ** 2).sum() * 100),
                         "rank_correlation_first_with_log_ratio": float(spearmanr(pcs[:, 0], np.log(rho))[0]),
                         "rank_correlation_first_with_K": float(spearmanr(pcs[:, 0], K)[0]),
                         "placement_alone_cosine_to_all_views": {"typical": c.typical(cos), "1of10_below": float(np.quantile(cos, 0.1))}}
    spread = lambda z, k: float(np.sqrt((((z - mu[K == k].mean(0)) / mu[K == k].std(0)) ** 2).mean()))
    runs_f = c.ev_in("ch4_search_runs.json")
    best = []
    if runs_f.is_file():
        for key, info in json.loads(runs_f.read_text()).items():
            if key.split("/")[0] == SEARCH_LABEL:
                for kd in sorted(c.search_run_dir(info["run_dir"]).glob("K*")):
                    if (kd / "best_latent.npy").is_file():
                        best.append((int(kd.name[1:]), key.split("/")[1], np.load(kd / "best_latent.npy").reshape(-1)))
    if best:
        d_search = [spread(z, k) for k, _, z in best]
        d_data = [spread(mu[i], K[i]) for i in range(len(mu))]
        res["search_latents"] = {"n": len(best), "distance_typical": c.typical(d_search), "distance_max": float(max(d_search)),
                                 "data_distance_typical": c.typical(d_data), "data_distance_9of10": c.nine_of_ten(d_data)}
        arrays.update(search_pcs=(np.stack([b[2] for b in best]) - mu.mean(0)) @ vt[:2].T, search_K=np.array([b[0] for b in best]),
                      search_target=np.array([b[1] for b in best]))
    arrays.update(latent_pcs=pcs, latent_pcs_placement=pcs_pl, ratio=rho, encoder_share_per_number=dim_share)

    # ---- 3. walks ----------------------------------------------------------------------------------
    zt = tg[WALK_TARGET]
    steps = []
    for k in WALK_K:
        ok = (K == k) & (codes[:, 0] == 0)
        a = int(rng.choice(np.flatnonzero(ok & (rho <= 0.9))))
        b = int(rng.choice(np.flatnonzero(ok & (rho > 1.1) & (c.distances(codes[[a]], codes)[0] >= k - 1))))
        za, zb = zpl[a], zpl[b]
        kt = torch.full((N_STEPS + 1,), k, dtype=torch.long, device=dev)
        z = torch.as_tensor(np.stack([za + (zb - za) * s / N_STEPS for s in range(N_STEPS + 1)]), device=dev, dtype=torch.float32)
        with torch.inference_mode():
            prob = torch.sigmoid(model.occupancy_decoder(torch.cat([z[:, :model.shared_latent_dim], model.k_embedding(kt)], 1)))
            score = prob.max(-1).values                                   # read-out of the search (s10_search.search._topk_occ)
            keep = torch.zeros_like(score).scatter_(-1, score.topk(k, dim=-1).indices, 1.0)
            hard = torch.nn.functional.one_hot(prob.argmax(-1), prob.shape[-1]).to(prob.dtype) * keep.unsqueeze(-1)
            cd = ((hard.argmax(-1) + 1) * (hard.sum(-1) > 0)).cpu().numpy().astype(np.int64)
            y = model.impedance_decoder(torch.cat([model._layout_latent_slice(z), model.k_embedding(kt), model.occupancy_encoder(hard)], 1))
        dec = np.exp(y.float().cpu().numpy().reshape(N_STEPS + 1, -1) * sur.head.log_std + sur.head.log_mean)
        for s in range(N_STEPS + 1):
            steps.append({"K": int(k), "step": s, "codes": cd[s].tolist(), "decoded": dec[s], "from": test[a], "to": test[b],
                          "end_codes": (codes[a] if s == 0 else codes[b] if s == N_STEPS else None)})
    wc = np.array([s["codes"] for s in steps], dtype=np.int64)
    Z = ecad.spectra_cached(ecad.store(STORE), wc)
    P = sur.spectra(wc)
    D = np.stack([s["decoded"] for s in steps])
    kk, st = np.array([s["K"] for s in steps]), np.array([s["step"] for s in steps])
    changed = np.array([0 if s["step"] == 0 else int(c.distances(wc[[i - 1]], wc[[i]])[0, 0]) for i, s in enumerate(steps)])
    end, mid = (st == 0) | (st == N_STEPS), (st > 0) & (st < N_STEPS)
    ends_ok = [bool(np.array_equal(s["codes"], s["end_codes"])) for s in steps if s["end_codes"] is not None]
    perr = lambda A, sel: np.abs(A[sel] / Z[sel] - 1) * 100
    rerr = lambda A, sel: np.abs(c.worst_ratio(A[sel], zt) / c.worst_ratio(Z[sel], zt) - 1) * 100
    res["walk"] = {
        "n_walks": len(WALK_K), "n_steps": N_STEPS, "budgets": list(WALK_K), "target": WALK_TARGET,
        "ends_read_out_as_the_placement": [int(sum(ends_ok)), len(ends_ok)],
        "decaps_changed_per_step": {"typical": c.typical(changed[st > 0]), "max": int(changed[st > 0].max()),
                                    "steps_without_change_pct": c.share(changed[st > 0] == 0)},
        "distinct_placements": int(len({x.tobytes() for x in wc})),
        "decoded_from_latent": {"on_the_way": {"point_error_typical_pct": c.typical(perr(D, mid)), "ratio_error_typical_pct": c.typical(rerr(D, mid)),
                                               "ratio_error_9of10_pct": c.nine_of_ten(rerr(D, mid))},
                                "ends": {"point_error_typical_pct": c.typical(perr(D, end)), "ratio_error_typical_pct": c.typical(rerr(D, end))}},
        "from_placement_alone": {"on_the_way": {"point_error_typical_pct": c.typical(perr(P, mid)), "ratio_error_typical_pct": c.typical(rerr(P, mid)),
                                                "ratio_error_9of10_pct": c.nine_of_ten(rerr(P, mid))},
                                 "ends": {"point_error_typical_pct": c.typical(perr(P, end)), "ratio_error_typical_pct": c.typical(rerr(P, end))}},
        "verdict_agreement_pct": {"decoded": c.share((c.worst_ratio(D, zt) <= 1) == (c.worst_ratio(Z, zt) <= 1)),
                                  "placement_alone": c.share((c.worst_ratio(P, zt) <= 1) == (c.worst_ratio(Z, zt) <= 1))}}
    arrays.update(walk_codes=wc, walk_K=kk, walk_step=st, walk_sim=Z, walk_decoded=D, walk_placement=P, walk_changed=changed)
    c.save_json(c.ev_out("ch4_latent.json"), res)
    np.savez_compressed(c.ev_out("ch4_latent.npz"), **arrays)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
