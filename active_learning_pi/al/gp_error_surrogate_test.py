"""Falsification test for the GP error-surrogate AL idea.

Question this answers (cheaply, before building the full AL loop):
    "Is the VAE's prediction error SMOOTH in its latent z?"
    i.e. can a Gaussian Process trained on (z -> error) for some layouts
    predict the error on *held-out* layouts?

If yes  -> the latent-space error-GP acquisition is viable.
If no   -> the latent can't support it (switch to physics features, or the
           occ-only bottleneck makes error unlearnable there).

Method
------
For a set of real, ground-truth-labelled layouts (the val split):
  x = VAE latent z            (full-encode  OR  occ-only-encode path)
  y = VAE prediction error    (peak-location px / structural / MAE),
      all computed in NORMALISED heatmap space so no denorm ambiguity.
      Peak targets include both hard argmax (diagnostic) and smooth soft-argmax
      (training-aligned surrogate — use this for GP acquisition).
Then: random split -> fit sklearn GP on a subset -> predict on held-out ->
report Spearman/Pearson(pred_error, true_error). A kNN-in-z baseline is
reported as an assumption-light cross-check, plus GP sigma calibration.

Run:
    python active_learning_pi/al/gp_error_surrogate_test.py \
        --exp experiments/exp059_capacity_freq \
        --path occ --n-samples 3000
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from active_learning_pi.al.gp_error_surrogate import (
    SklearnGP as _SklearnGP,
    SVGPRegressor,
    errors_from_norm as _errors_from_norm,
)


def _load_engine_and_loader(exp: str, checkpoint: str | None, device: torch.device):
    groot = Path.cwd()
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))
    pkg = exp.replace("/", ".").rstrip(".")
    inf = importlib.import_module(pkg + ".codes.inference_vae")
    dl = importlib.import_module(pkg + ".codes.dataloader_multifreq")

    cfg = inf.load_experiment_config(Path(exp) / "config.yaml")
    data_dir = cfg["data_dir"]
    ckpt = checkpoint or str(Path(exp) / "checkpoints" / "last_model.pt")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    engine = inf.VAEInference(checkpoint_path=ckpt, device=device)
    engine.model.eval()

    _, val_loader = dl.create_multifreq_data_loaders(
        data_dir,
        batch_size=int(cfg.get("val_batch_size", 128) or 128),
        num_workers=0,
        pin_memory=False,
        balance_k=False,
        balance_freq=False,
        stratify_by_k=False,
        cross_freq_pairs=False,
        drop_last_train=False,
    )
    # The val split is stored in design/K order; iterating it sequentially gives a
    # biased sample (e.g. only low K). Rebuild a SHUFFLED loader over the same
    # dataset so the collected sample spans K and MHz representatively.
    from torch.utils.data import DataLoader as _DL
    val_loader = _DL(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=val_loader.collate_fn,
    )
    return engine, val_loader, data_dir, ckpt, cfg


@torch.no_grad()
def collect(engine, val_loader, *, path: str, n_samples: int, device: torch.device, cfg: dict):
    from src_vae.others.norm_stats import pi_norm_to_mhz

    model = engine.model
    Z, Y_peak, Y_soft, Y_sharp, Y_struct, Y_mae, MHZ, KK = [], [], [], [], [], [], [], []
    seen = 0
    for batch in val_loader:
        hm = batch["heatmap_norm"].to(device).float()
        occ = batch["occupancy"].to(device).float()
        imp = batch["impedance"].to(device).float()
        K = batch["K"].to(device)
        pi = batch["PI_freq"].to(device).float()

        if path == "full":
            # x = encode(real heatmap); prediction = reconstruction. Best-case
            # (leaky) reference: needs the ECAD heatmap you won't have at AL time.
            _, mu, _, _ = model.encode(hm, occ, imp, K, pi)
            z = mu
            hm_used, _, _ = model.decode(z, K, pi, occupancy=occ)
        elif path == "occ":
            # x = occupancy-only latent (deployed AL path, cheap).
            z = model.encode_layout_latent(occ, imp, K, pi)
            hm_used, _, _ = model.decode(z, K, pi, occupancy=occ)
        elif path == "pred":
            # Option 1: self-prediction bootstrap. Predict a heatmap from the
            # occ path (cheap, no ECAD), then RE-ENCODE that prediction to form a
            # pseudo-full-z as the GP input. Error target is the SAME occ-path
            # prediction, so pred vs occ isolates the feature quality gain.
            z_occ = model.encode_layout_latent(occ, imp, K, pi)
            hm_used, _, _ = model.decode(z_occ, K, pi, occupancy=occ)
            _, mu, _, _ = model.encode(hm_used, occ, imp, K, pi)
            z = mu
        else:
            raise ValueError(f"unknown path {path!r}")

        z_np = z.detach().cpu().numpy()
        pred_np = hm_used[:, 0].detach().cpu().numpy()
        gt_np = hm[:, 0].detach().cpu().numpy()
        pi_np = pi.detach().cpu().numpy().reshape(-1)
        k_np = K.detach().cpu().numpy().reshape(-1)

        bg = float(cfg.get("background_value", -3.1792)) + 0.5
        peak_t = float(cfg.get("heatmap_peak_loc_temperature", 0.035))
        for i in range(z_np.shape[0]):
            e = _errors_from_norm(pred_np[i], gt_np[i], bg=bg, peak_temperature=peak_t)
            Z.append(z_np[i])
            Y_peak.append(e["peak_px"])
            Y_soft.append(e["peak_soft_px"])
            Y_sharp.append(e["peak_sharp_px"])
            Y_struct.append(e["struct"])
            Y_mae.append(e["mae"])
            MHZ.append(float(pi_norm_to_mhz(float(pi_np[i]))))
            KK.append(int(k_np[i]))
            seen += 1
        if seen >= n_samples:
            break

    return (
        np.asarray(Z, dtype=np.float64),
        {
            "peak_px": np.asarray(Y_peak),
            "peak_soft_px": np.asarray(Y_soft),
            "peak_sharp_px": np.asarray(Y_sharp),
            "struct": np.asarray(Y_struct),
            "mae": np.asarray(Y_mae),
        },
        np.asarray(MHZ),
        np.asarray(KK),
    )


@torch.no_grad()
def collect_multi(engine, val_loader, paths, *, n_samples, device, cfg):
    """Collect latents + errors for several paths on the SAME layouts (apples-to-
    apples). 'occ' and 'pred' share the occ prediction as error target and differ
    only in the GP input (occ-z vs re-encoded prediction); 'full' is the leaky
    reference (encodes the real heatmap)."""
    from src_vae.others.norm_stats import pi_norm_to_mhz

    model = engine.model
    tkeys = ("peak_px", "peak_soft_px", "peak_sharp_px", "struct", "mae")
    acc = {p: {"Z": [], "Y": {t: [] for t in tkeys}} for p in paths}
    MHZ, KK = [], []
    bg = float(cfg.get("background_value", -3.1792)) + 0.5
    peak_t = float(cfg.get("heatmap_peak_loc_temperature", 0.035))
    seen = 0
    for batch in val_loader:
        hm = batch["heatmap_norm"].to(device).float()
        occ = batch["occupancy"].to(device).float()
        imp = batch["impedance"].to(device).float()
        K = batch["K"].to(device)
        pi = batch["PI_freq"].to(device).float()
        gt_np = hm[:, 0].detach().cpu().numpy()
        pi_np = pi.detach().cpu().numpy().reshape(-1)
        k_np = K.detach().cpu().numpy().reshape(-1)

        z_occ = hm_occ = None
        if any(p in ("occ", "pred") for p in paths):
            z_occ = model.encode_layout_latent(occ, imp, K, pi)
            hm_occ, _, _ = model.decode(z_occ, K, pi, occupancy=occ)
        if "full" in paths:
            _, mu_full, _, _ = model.encode(hm, occ, imp, K, pi)
            hm_full, _, _ = model.decode(mu_full, K, pi, occupancy=occ)
        if "pred" in paths:
            _, mu_pred, _, _ = model.encode(hm_occ, occ, imp, K, pi)

        for p in paths:
            if p == "full":
                z, pred = mu_full, hm_full
            elif p == "occ":
                z, pred = z_occ, hm_occ
            elif p == "pred":
                z, pred = mu_pred, hm_occ  # target = occ prediction error
            else:
                raise ValueError(f"unknown path {p!r}")
            z_np = z.detach().cpu().numpy()
            pred_np = pred[:, 0].detach().cpu().numpy()
            for i in range(z_np.shape[0]):
                e = _errors_from_norm(pred_np[i], gt_np[i], bg=bg, peak_temperature=peak_t)
                acc[p]["Z"].append(z_np[i])
                for t in tkeys:
                    acc[p]["Y"][t].append(e[t])

        for i in range(hm.shape[0]):
            MHZ.append(float(pi_norm_to_mhz(float(pi_np[i]))))
            KK.append(int(k_np[i]))
        seen += hm.shape[0]
        if seen >= n_samples:
            break

    per_path = {
        p: (np.asarray(acc[p]["Z"], dtype=np.float64),
            {t: np.asarray(acc[p]["Y"][t]) for t in tkeys})
        for p in paths
    }
    return per_path, np.asarray(MHZ), np.asarray(KK)


def _structured_split(holdout, MHZ, KK, values, test_frac, seed):
    """Return (idx_fit, idx_test, description).

    holdout='random' -> interpolation baseline (over-optimistic).
    holdout='k'/'mhz' -> remove whole K / MHz bands from training and test ONLY
    on the held-out band => simulates a genuine hole the GP never saw.
    holdout='type'    -> decap-type extrapolation (needs multi-type data).
    """
    n = len(KK)
    if holdout == "random":
        perm = np.random.RandomState(seed).permutation(n)
        nt = int(n * test_frac)
        return perm[nt:], perm[:nt], "random (interpolation baseline)"

    if holdout == "type":
        raise SystemExit(
            "holdout=type: this dataset has a single decap type — no type to hold "
            "out yet. Re-run once multi-type data exists (this is the true "
            "extrapolation test for the decap-type extension)."
        )

    if holdout == "k":
        uk = sorted({int(k) for k in KK})
        if values:
            hold_vals = [int(v) for v in values]
        else:  # middle ~30% of unique K values (an interior gap)
            lo, hi = int(len(uk) * 0.35), int(len(uk) * 0.65)
            hold_vals = uk[lo:hi] if hi > lo else [uk[len(uk) // 2]]
        hold = set(hold_vals)
        test_mask = np.array([int(k) in hold for k in KK])
        desc = f"K held out = {sorted(hold)}"
    elif holdout == "mhz":
        um = sorted({round(float(m), 1) for m in MHZ})
        if values:
            hold_vals = [float(v) for v in values]
        else:  # middle ~30% of unique MHz bins
            lo, hi = int(len(um) * 0.35), int(len(um) * 0.65)
            hold_vals = um[lo:hi] if hi > lo else [um[len(um) // 2]]
        test_mask = np.array(
            [any(abs(float(m) - h) <= 2.0 for h in hold_vals) for m in MHZ]
        )
        desc = f"MHz held out ≈ {sorted(hold_vals)}"
    else:
        raise ValueError(f"unknown holdout {holdout!r}")

    idx_test = np.where(test_mask)[0]
    idx_fit = np.where(~test_mask)[0]
    if len(idx_test) < 30 or len(idx_fit) < 100:
        raise SystemExit(
            f"holdout '{holdout}' gave n_fit={len(idx_fit)} n_test={len(idx_test)} "
            f"— too few. Increase --n-samples or choose different --holdout-values."
        )
    return idx_fit, idx_test, desc


def _knn_baseline(z_fit, y_fit, z_test, k: int = 10) -> np.ndarray:
    """Assumption-light smoothness check: predict error = mean of k nearest
    training neighbours in z-space."""
    preds = np.empty(z_test.shape[0])
    for i in range(z_test.shape[0]):
        d = np.sum((z_fit - z_test[i]) ** 2, axis=1)
        nn = np.argpartition(d, min(k, len(d) - 1))[:k]
        preds[i] = float(np.mean(y_fit[nn]))
    return preds


def _make_regressor(kind, device):
    if kind == "svgp":
        return SVGPRegressor(
            device=device,
            n_inducing=_GP_OPTS["n_inducing"],
            epochs=_GP_OPTS["epochs"],
            verbose=_GP_OPTS["verbose"],
        )
    return _SklearnGP()


_GP_OPTS = {"kind": "sklearn", "n_inducing": 256, "epochs": 80, "verbose": False}


def _report_target(name, y, z, idx_fit, idx_test, n_fit_cap, idx_ref=None,
                   gp_kind="sklearn", device=None):
    z_fit_all, y_fit_all = z[idx_fit], y[idx_fit]
    # cap fit size for the exact sklearn GP (O(N^3)); SVGP scales, use all points.
    if gp_kind == "sklearn" and len(idx_fit) > n_fit_cap:
        sub = np.random.RandomState(0).choice(len(idx_fit), n_fit_cap, replace=False)
        z_fit, y_fit = z_fit_all[sub], y_fit_all[sub]
    else:
        z_fit, y_fit = z_fit_all, y_fit_all
    z_test, y_test = z[idx_test], y[idx_test]

    reg = _make_regressor(gp_kind, device).fit(z_fit, y_fit)
    mean, std = reg.predict(z_test)
    knn = _knn_baseline(z_fit, y_fit, z_test, k=10)

    sp_gp = spearmanr(mean, y_test).statistic
    pe_gp = pearsonr(mean, y_test)[0]
    sp_knn = spearmanr(knn, y_test).statistic
    # sigma calibration: does predicted std track the GP's own abs residual?
    resid = np.abs(y_test - mean)
    sp_cal = spearmanr(std, resid).statistic

    # sigma OOD-elevation: is GP uncertainty higher on the held-out hole than on
    # an in-distribution reference? >1 => sigma detects the dataset hole.
    sigma_elev = None
    if idx_ref is not None and len(idx_ref) > 5:
        _, std_ref = reg.predict(z[idx_ref])
        med_test, med_ref = float(np.median(std)), float(np.median(std_ref))
        sigma_elev = med_test / med_ref if med_ref > 1e-9 else float("nan")

    print(f"\n── target: {name}   (y: mean={y.mean():.3f} std={y.std():.3f} "
          f"min={y.min():.3f} max={y.max():.3f})")
    print(f"   n_fit={len(z_fit)}  n_test={len(z_test)}")
    print(f"   GP   Spearman(pred, true) = {sp_gp:+.3f}   Pearson = {pe_gp:+.3f}")
    print(f"   kNN  Spearman(pred, true) = {sp_knn:+.3f}   (assumption-light check)")
    print(f"   sigma calibration Spearman(std, |resid|) = {sp_cal:+.3f}")
    if sigma_elev is not None:
        print(f"   sigma OOD-elevation (median std_hole / std_indist) = {sigma_elev:.2f}x")
    verdict = (
        "STRONG — error is smooth in z, GP viable"
        if sp_gp is not None and sp_gp > 0.4
        else "WEAK — latent poorly predicts error (see notes)"
        if sp_gp is not None and sp_gp > 0.15
        else "FAIL — z does not predict error here"
    )
    print(f"   => {verdict}")
    return {"target": name, "spearman_gp": sp_gp, "pearson_gp": pe_gp,
            "spearman_knn": sp_knn, "sigma_cal": sp_cal, "sigma_elev": sigma_elev}


def _evaluate(Z, Y, MHZ, KK, args, *, targets, device=None):
    """Split (structured OOD or random) + scale-on-fit + per-target GP report.

    Returns (results, desc). For k/mhz holdout, carves an in-distribution
    reference subset (excluded from GP fit) so sigma OOD-elevation can be measured.
    """
    hv = [v for v in args.holdout_values.split(",")] if args.holdout_values else None
    idx_fit, idx_test, desc = _structured_split(
        args.holdout, MHZ, KK, hv, args.test_frac, args.seed)

    idx_ref = None
    if args.holdout in ("k", "mhz"):
        rng = np.random.RandomState(args.seed + 1)
        nref = min(300, len(idx_fit) // 5)
        sel = rng.choice(len(idx_fit), nref, replace=False)
        mask = np.ones(len(idx_fit), dtype=bool)
        mask[sel] = False
        idx_ref = idx_fit[sel]
        idx_fit = idx_fit[mask]

    print(f"\nholdout = {desc}\n  n_fit={len(idx_fit)}  n_test={len(idx_test)}"
          + (f"  n_ref(in-dist)={len(idx_ref)}" if idx_ref is not None else ""))
    if args.holdout in ("k", "mhz"):
        print(f"  test K range=[{KK[idx_test].min()},{KK[idx_test].max()}]  "
              f"test MHz range=[{MHZ[idx_test].min():.0f},{MHZ[idx_test].max():.0f}]")

    scaler = StandardScaler().fit(Z[idx_fit])  # fit-split only (no leakage)
    Zs = scaler.transform(Z)
    results = [_report_target(name, Y[name], Zs, idx_fit, idx_test, args.n_fit_cap,
                              idx_ref, gp_kind=args.gp, device=device)
               for name in targets]
    return results, desc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="experiments/exp059_capacity_freq")
    ap.add_argument("--checkpoint", default=None, help="default: <exp>/checkpoints/last_model.pt")
    ap.add_argument("--path", choices=["full", "occ", "pred"], default="occ",
                    help="latent path: 'occ' = deployed AL path (occupancy-only), "
                         "'full' = uses ground-truth heatmap (reference/best-case), "
                         "'pred' = self-prediction bootstrap (Option 1: encode the "
                         "model's own predicted heatmap; cheap, no ECAD)")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--n-fit-cap", type=int, default=1500, help="max GP training points")
    ap.add_argument("--test-frac", type=float, default=0.4, help="only for --holdout random")
    ap.add_argument("--holdout", choices=["random", "k", "mhz", "type"], default="random",
                    help="'random' = interpolation baseline; 'k'/'mhz' = structured OOD "
                         "(hold out whole bands = a real hole); 'type' = decap-type extrapolation")
    ap.add_argument("--holdout-values", default=None,
                    help="comma-separated K or MHz values to hold out (default: auto mid-band)")
    ap.add_argument("--compare-paths", default=None,
                    help="comma-separated paths to compare side-by-side on the SAME "
                         "layouts, e.g. 'full,pred,occ'. Reports mu (ranking) and "
                         "sigma (uncertainty) axes per path.")
    ap.add_argument("--gp", choices=["sklearn", "svgp"], default="sklearn",
                    help="GP backend: 'sklearn' exact GP (capped fit), 'svgp' sparse "
                         "variational GP (gpytorch, inducing points, scales, better sigma)")
    ap.add_argument("--n-inducing", type=int, default=256, help="SVGP inducing points")
    ap.add_argument("--svgp-epochs", type=int, default=80)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _GP_OPTS.update(kind=args.gp, n_inducing=args.n_inducing,
                    epochs=args.svgp_epochs, verbose=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"exp={args.exp}  device={device}  gp={args.gp}"
          + (f" (inducing={args.n_inducing}, epochs={args.svgp_epochs})" if args.gp == "svgp" else ""))
    engine, val_loader, data_dir, ckpt, cfg = _load_engine_and_loader(
        args.exp, args.checkpoint, device)
    print(f"data_dir={data_dir}\ncheckpoint={ckpt}")

    targets = ("peak_sharp_px", "peak_soft_px", "peak_px", "struct", "mae")

    if args.compare_paths:
        paths = [p.strip() for p in args.compare_paths.split(",") if p.strip()]
        print(f"\nCOMPARE PATHS = {paths}  (same layouts, same holdout)")
        per_path, MHZ, KK = collect_multi(
            engine, val_loader, paths, n_samples=args.n_samples, device=device, cfg=cfg)
        n0 = per_path[paths[0]][0].shape[0]
        print(f"collected N={n0} samples, MHz=[{MHZ.min():.0f},{MHZ.max():.0f}], "
              f"K=[{KK.min()},{KK.max()}]")
        summary = {}
        for p in paths:
            Zp, Yp = per_path[p]
            print("\n" + "#" * 64 + f"\n#### PATH: {p}")
            res, _ = _evaluate(Zp, Yp, MHZ, KK, args, targets=targets, device=device)
            summary[p] = {r["target"]: r for r in res}

        prim = ["peak_sharp_px", "struct", "mae"]
        print("\n" + "=" * 72)
        print(f"PATH COMPARISON  | holdout={args.holdout}")
        print("  mu   = Spearman(pred_error, true_error)   [RANKING quality]")
        print("  cal  = Spearman(sigma, |residual|)        [uncertainty calibration]")
        print("  elev = median sigma_hole / sigma_indist   [does sigma flag the hole? >1 = yes]")
        for t in prim:
            print(f"\n  target: {t}")
            print(f"    {'path':<6}{'mu':>8}{'cal':>8}{'elev':>8}")
            for p in paths:
                r = summary[p][t]
                el = r.get("sigma_elev")
                el_s = f"{el:.2f}x" if el is not None else "—"
                print(f"    {p:<6}{r['spearman_gp']:>+8.3f}{r['sigma_cal']:>+8.3f}{el_s:>8}")
        print("\nRead: pred near full on 'mu' => cheap self-prediction recovers full-path")
        print("ranking quality. Compare 'cal'/'elev' to see whether pred's UNCERTAINTY")
        print("(sigma) behaves like full's. elev>1 => sigma is elevated on the held-out hole.")
        return

    print(f"path={args.path}")
    Z, Y, MHZ, KK = collect(
        engine, val_loader, path=args.path, n_samples=args.n_samples, device=device, cfg=cfg)
    print(f"\ncollected N={Z.shape[0]} samples, latent_dim={Z.shape[1]}, "
          f"MHz range=[{MHZ.min():.0f},{MHZ.max():.0f}], K range=[{KK.min()},{KK.max()}]")

    results, _ = _evaluate(Z, Y, MHZ, KK, args, targets=targets, device=device)

    print("\n" + "=" * 64)
    print(f"SUMMARY (GP Spearman vs true error | holdout={args.holdout} | path={args.path}):")
    for r in results:
        el = r.get("sigma_elev")
        el_s = f"  sigma_elev={el:.2f}x" if el is not None else ""
        print(f"  {r['target']:<14}  GP={r['spearman_gp']:+.3f}  kNN={r['spearman_knn']:+.3f}{el_s}")
    print("Interpretation:")
    print("  peak_sharp_px = training-aligned soft-argmax (PRIMARY for GP acquisition)")
    print("  peak_soft_px  = mass-centroid peak (eval metric)")
    print("  peak_px       = hard argmax (diagnostic; often unlearnable ~0.1)")
    print("  struct/mae    = smooth structural proxies")
    print("If peak_sharp_px is strong but peak_px is weak, the measurement problem is")
    print("confirmed — use peak_sharp_px for AL. If ALL peak targets are weak on")
    print("--path occ but stronger on --path full, the occ-only bottleneck is the limiter.")


if __name__ == "__main__":
    main()
