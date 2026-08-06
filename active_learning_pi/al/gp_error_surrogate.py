"""Latent-space error-GP surrogate for active-learning acquisition.

Fits a GP (SVGP or sklearn) that maps a cheap VAE latent feature ``x`` to a
scalar residual ``y`` (VAE error vs ECAD on labeled layouts), then scores
unlabeled candidates with UCB:

    a(x) = mu_e(x) + kappa * sigma_e(x)

Production feature path (``pred``): self-prediction bootstrap without ECAD —

    occ → encode_occupancy_latent → decode Ĥ → re-encode(Ĥ) → z_pred

Falsification CLI: ``gp_error_surrogate_test.py`` (imports regressors / errors here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

ErrorTarget = Literal[
    "mse",
    "mae",
    "struct",
    "joint",
    "peak_px",
    "peak_soft_px",
    "peak_sharp_px",
    "p99_ae",
]
FeaturePath = Literal["full", "occ", "pred"]


# ── Peak / residual helpers ───────────────────────────────────────────────────


def peak_xy(hm: np.ndarray) -> tuple[int, int]:
    r, c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return int(r), int(c)


def fg_mask(hm: np.ndarray, *, bg: float, margin: float = 0.5) -> np.ndarray:
    return hm > (bg + margin)


def soft_peak_xy(hm: np.ndarray, *, bg: float, margin: float = 0.5) -> tuple[float, float]:
    fg = fg_mask(hm, bg=bg, margin=margin)
    h, w = hm.shape
    if not fg.any():
        return (h - 1) * 0.5, (w - 1) * 0.5
    wmap = np.maximum(hm - (bg + margin), 0.0) * fg
    s = float(wmap.sum())
    if s < 1e-12:
        return (h - 1) * 0.5, (w - 1) * 0.5
    ys, xs = np.indices(hm.shape)
    return float((wmap * ys).sum() / s), float((wmap * xs).sum() / s)


def sharp_peak_xy(
    hm: np.ndarray, *, bg: float, margin: float = 0.5, temperature: float = 0.035
) -> tuple[float, float]:
    fg = fg_mask(hm, bg=bg, margin=margin)
    h, w = hm.shape
    if not fg.any():
        return (h - 1) * 0.5, (w - 1) * 0.5
    logits = np.maximum(hm - (bg + margin), 0.0) * fg
    flat = logits.reshape(-1).astype(np.float64) / max(float(temperature), 1e-3)
    flat = flat - flat.max()
    wmap = np.exp(flat)
    s = float(wmap.sum())
    if s < 1e-12:
        return (h - 1) * 0.5, (w - 1) * 0.5
    ys = np.repeat(np.arange(h), w)
    xs = np.tile(np.arange(w), h)
    return float((wmap * ys).sum() / s), float((wmap * xs).sum() / s)


def peak_dist_px(ay: float, ax: float, by: float, bx: float) -> float:
    return float(np.hypot(ay - by, ax - bx))


def errors_from_norm(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    bg: float = -2.6792,
    peak_temperature: float = 0.035,
) -> dict[str, float]:
    """pred, gt: (H, W) normalised heatmaps (channel 0)."""
    pr, pc = peak_xy(pred)
    gr, gc = peak_xy(gt)
    peak_px = peak_dist_px(pr, pc, gr, gc)

    sry, srx = soft_peak_xy(pred, bg=bg)
    sty, stx = soft_peak_xy(gt, bg=bg)
    peak_soft_px = peak_dist_px(sry, srx, sty, stx)

    hry, hrx = sharp_peak_xy(pred, bg=bg, temperature=peak_temperature)
    hty, htx = sharp_peak_xy(gt, bg=bg, temperature=peak_temperature)
    peak_sharp_px = peak_dist_px(hry, hrx, hty, htx)

    a = pred.reshape(-1).astype(np.float64)
    b = gt.reshape(-1).astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        struct = 1.0
    else:
        struct = float(1.0 - np.corrcoef(a, b)[0, 1])
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    return {
        "peak_px": peak_px,
        "peak_soft_px": peak_soft_px,
        "peak_sharp_px": peak_sharp_px,
        "struct": struct,
        "mae": mae,
        "mse": mse,
    }


def impedance_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean squared error on impedance spectrum vectors (any matching shape)."""
    a = np.asarray(pred, dtype=np.float64).reshape(-1)
    b = np.asarray(gt, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean((a[:n] - b[:n]) ** 2))


def physical_p99_abs_err(pred_phys: np.ndarray, gt_phys: np.ndarray) -> float:
    """Absolute error of foreground p99 between physical heatmaps (Ω)."""
    from active_learning_pi.al.robust_stats import robust_peak_stats

    pp = robust_peak_stats(pred_phys)
    gg = robust_peak_stats(gt_phys)
    return abs(float(pp["p99"]) - float(gg["p99"]))


def build_joint_target(
    y_hm: np.ndarray,
    y_imp: np.ndarray,
    *,
    alpha_hm: float = 1.0,
    alpha_imp: float = 1.0,
) -> np.ndarray:
    """Scale-normalize heatmap and impedance residuals then weighted-sum."""
    y_hm = np.asarray(y_hm, dtype=np.float64)
    y_imp = np.asarray(y_imp, dtype=np.float64)
    hm_s = float(y_hm.std() or 1.0)
    imp_s = float(y_imp.std() or 1.0)
    return (
        float(alpha_hm) * (y_hm - y_hm.mean()) / hm_s
        + float(alpha_imp) * (y_imp - y_imp.mean()) / imp_s
    )


def knn_novelty(
    z_query: np.ndarray,
    z_bank: np.ndarray,
    *,
    k: int = 10,
) -> np.ndarray:
    """Distance to k-th nearest neighbor in ``z_bank`` (higher = more novel / sparse)."""
    z_query = np.asarray(z_query, dtype=np.float64)
    z_bank = np.asarray(z_bank, dtype=np.float64)
    if z_bank.ndim != 2 or z_bank.shape[0] == 0:
        return np.zeros(z_query.shape[0], dtype=np.float64)
    k_eff = max(1, min(int(k), z_bank.shape[0]))
    out = np.empty(z_query.shape[0], dtype=np.float64)
    # Chunked to avoid huge (Nq x Nb) matrices for large pools
    chunk = 512
    for i0 in range(0, z_query.shape[0], chunk):
        q = z_query[i0 : i0 + chunk]
        # (cq, nb)
        d2 = np.sum((q[:, None, :] - z_bank[None, :, :]) ** 2, axis=-1)
        part = np.partition(d2, k_eff - 1, axis=1)[:, k_eff - 1]
        out[i0 : i0 + chunk] = np.sqrt(np.maximum(part, 0.0))
    return out


# ── Regressors ────────────────────────────────────────────────────────────────


class SklearnGP:
    """Thin wrapper giving sklearn GP a uniform fit/predict interface."""

    def fit(self, z, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0
        )
        self.gp.fit(z, y)
        return self

    def predict(self, z):
        m, s = self.gp.predict(z, return_std=True)
        return np.asarray(m), np.asarray(s)


class SVGPRegressor:
    """Sparse Variational GP (gpytorch): inducing points + ARD-RBF + variational ELBO."""

    def __init__(
        self,
        *,
        n_inducing: int = 256,
        epochs: int = 80,
        lr: float = 0.01,
        batch: int = 1024,
        device=None,
        verbose: bool = False,
    ):
        self.n_inducing = n_inducing
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def fit(self, z, y):
        import gpytorch
        from gpytorch.models import ApproximateGP
        from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
        from torch.utils.data import DataLoader, TensorDataset

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        yn = (y - self.y_mean_) / self.y_std_

        zt = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(yn, dtype=torch.float32, device=self.device)
        n, _d = zt.shape
        m = min(self.n_inducing, n)
        ind = zt[torch.randperm(n, device=self.device)[:m]].clone()

        class _Model(ApproximateGP):
            def __init__(self, inducing):
                vd = CholeskyVariationalDistribution(inducing.size(0))
                vs = VariationalStrategy(self, inducing, vd, learn_inducing_locations=True)
                super().__init__(vs)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=inducing.size(1))
                )
                self.covar_module.base_kernel.lengthscale = float(np.sqrt(inducing.size(1)))

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        self._gp = _Model(ind).to(self.device)
        self._lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self._gp.train()
        self._lik.train()
        opt = torch.optim.Adam(
            [{"params": self._gp.parameters()}, {"params": self._lik.parameters()}],
            lr=self.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(self._lik, self._gp, num_data=n)
        dl = DataLoader(TensorDataset(zt, yt), batch_size=self.batch, shuffle=True)
        for ep in range(self.epochs):
            tot = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                loss = -mll(self._gp(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss)
            if self.verbose and (ep % 20 == 0 or ep == self.epochs - 1):
                print(f"      svgp epoch {ep:3d}  elbo_loss={tot / len(dl):.4f}")
        return self

    @torch.no_grad()
    def predict(self, z):
        import gpytorch

        self._gp.eval()
        self._lik.eval()
        zt = torch.as_tensor(np.asarray(z, dtype=np.float64), dtype=torch.float32, device=self.device)
        means, stds = [], []
        with gpytorch.settings.fast_pred_var():
            for i in range(0, zt.shape[0], 4096):
                pred = self._lik(self._gp(zt[i : i + 4096]))
                means.append(pred.mean.cpu().numpy())
                stds.append(pred.variance.clamp_min(1e-12).sqrt().cpu().numpy())
        mean = np.concatenate(means) * self.y_std_ + self.y_mean_
        std = np.concatenate(stds) * self.y_std_
        return mean, std


def make_regressor(
    kind: str,
    device,
    *,
    n_inducing: int = 256,
    epochs: int = 80,
    verbose: bool = False,
):
    if kind == "svgp":
        return SVGPRegressor(
            device=device, n_inducing=n_inducing, epochs=epochs, verbose=verbose
        )
    return SklearnGP()


# ── Artifact ──────────────────────────────────────────────────────────────────


@dataclass
class ErrorGPArtifact:
    """Fitted error-GP ready for UCB (+ optional novelty) scoring."""

    regressor: Any
    scaler: StandardScaler
    path: str = "pred"
    target: str = "mse"
    kappa: float = 0.5
    gp_kind: str = "svgp"
    n_fit: int = 0
    novelty_weight: float = 0.0
    novelty_k: int = 10
    score_mode: str = "mu"  # mu | ucb | ucb_novelty
    z_bank_scaled: np.ndarray | None = None
    novelty_mean: float = 0.0
    novelty_std: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def score_z(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (mu, sigma, ucb, acquisition) for raw latent rows ``z`` (N, D).

        ``score_mode``:
          - ``mu``: expected residual only (preferred when σ/novelty are poorly calibrated)
          - ``ucb``: mu + kappa * sigma
          - ``ucb_novelty``: ucb + novelty_weight * zscore(knn_novelty)
        """
        zs = self.scaler.transform(np.asarray(z, dtype=np.float64))
        mu, sigma = self.regressor.predict(zs)
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        ucb = mu + float(self.kappa) * sigma
        novelty = np.zeros_like(ucb)
        mode = str(self.score_mode).lower()
        if mode == "mu":
            acq = mu.copy()
        elif mode == "ucb":
            acq = ucb.copy()
        else:
            acq = ucb.copy()
            if self.z_bank_scaled is not None and float(self.novelty_weight) > 0.0:
                novelty = knn_novelty(zs, self.z_bank_scaled, k=self.novelty_k)
                nov_z = (novelty - float(self.novelty_mean)) / float(self.novelty_std or 1.0)
                novelty = float(self.novelty_weight) * nov_z
                acq = ucb + novelty
        return mu, sigma, ucb, acq


# ── Feature extraction ────────────────────────────────────────────────────────


@torch.no_grad()
def pred_bootstrap_features(
    model,
    *,
    occupancy: torch.Tensor,
    K: torch.Tensor,
    pi: torch.Tensor,
    impedance: torch.Tensor | None = None,
    path: FeaturePath = "pred",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Compute GP input latent ``z`` and predicted heatmap.

    Production ``pred`` path (no ECAD):
      encode_occupancy_latent → decode → re-encode(pred heatmap) → z_pred

    ``occ``: occupancy latent only (cheap; weak ranking).
    """
    if path == "occ":
        z = model.encode_occupancy_latent(occupancy, K, pi, sample=False)
        hm, _, imp_out = model.decode(z, K, pi, occupancy=occupancy)
        return z, hm, imp_out

    if path != "pred":
        raise ValueError(f"pred_bootstrap_features only supports occ|pred, got {path!r}")

    z_occ = model.encode_occupancy_latent(occupancy, K, pi, sample=False)
    hm, _, imp_hat = model.decode(z_occ, K, pi, occupancy=occupancy)
    imp_use = impedance if impedance is not None else imp_hat
    _, mu, _, _ = model.encode(hm, occupancy, imp_use, K, pi)
    return mu, hm, imp_hat


@torch.no_grad()
def collect_residuals(
    engine,
    loader,
    *,
    path: FeaturePath,
    n_samples: int,
    device: torch.device,
    cfg: dict,
    return_occupancy: bool = False,
):
    """Collect (Z, Y_dict, MHZ, KK[, OCC]) on labeled batches for GP fitting / falsification.

    ``Y`` includes heatmap residuals plus ``imp_mse``. ``joint`` is filled by the
    caller via ``build_joint_target`` when needed.
    """
    from src_vae.others.norm_stats import pi_norm_to_mhz

    model = engine.model
    tkeys = ("peak_px", "peak_soft_px", "peak_sharp_px", "struct", "mae", "mse", "imp_mse", "p99_ae")
    Z: list[np.ndarray] = []
    Y: dict[str, list[float]] = {t: [] for t in tkeys}
    MHZ: list[float] = []
    KK: list[int] = []
    OCC: list[np.ndarray] = []
    bg = float(cfg.get("background_value", -3.1792)) + 0.5
    peak_t = float(cfg.get("heatmap_peak_loc_temperature", 0.035))
    seen = 0
    denorm = getattr(engine, "denorm_heatmap_physical", None)

    for batch in loader:
        hm = batch["heatmap_norm"].to(device).float()
        occ = batch["occupancy"].to(device).float()
        imp = batch["impedance"].to(device).float()
        K = batch["K"].to(device)
        pi = batch["PI_freq"].to(device).float()
        gt_np = hm[:, 0].detach().cpu().numpy()
        imp_gt = imp.detach().cpu().numpy()
        occ_np = occ.detach().cpu().numpy()
        pi_np = pi.detach().cpu().numpy().reshape(-1)
        k_np = K.detach().cpu().numpy().reshape(-1)

        if path == "full":
            _, mu, _, _ = model.encode(hm, occ, imp, K, pi)
            hm_used, _, imp_used = model.decode(mu, K, pi, occupancy=occ)
            z = mu
        elif path in ("occ", "pred"):
            z, hm_used, imp_used = pred_bootstrap_features(
                model, occupancy=occ, K=K, pi=pi, impedance=imp, path=path
            )
        else:
            raise ValueError(f"unknown path {path!r}")

        z_np = z.detach().cpu().numpy()
        pred_np = hm_used[:, 0].detach().cpu().numpy()
        imp_pred = (
            imp_used.detach().cpu().numpy()
            if imp_used is not None
            else np.zeros_like(imp_gt)
        )

        # Physical denorm for p99_ae (aligned with ECAD rank-quality metric).
        pred_phys = gt_phys = None
        if callable(denorm):
            pred_t = denorm(hm_used, pi_norm=pi)
            gt_t = denorm(hm, pi_norm=pi)
            if isinstance(pred_t, torch.Tensor):
                pred_phys = pred_t.detach().cpu().numpy()
            else:
                pred_phys = np.asarray(pred_t)
            if isinstance(gt_t, torch.Tensor):
                gt_phys = gt_t.detach().cpu().numpy()
            else:
                gt_phys = np.asarray(gt_t)
            # (B,C,H,W) → use channel 0
            if pred_phys.ndim == 4:
                pred_phys = pred_phys[:, 0]
            if gt_phys.ndim == 4:
                gt_phys = gt_phys[:, 0]

        for i in range(z_np.shape[0]):
            e = errors_from_norm(pred_np[i], gt_np[i], bg=bg, peak_temperature=peak_t)
            e["imp_mse"] = impedance_mse(imp_pred[i], imp_gt[i])
            if pred_phys is not None and gt_phys is not None:
                e["p99_ae"] = physical_p99_abs_err(pred_phys[i], gt_phys[i])
            else:
                e["p99_ae"] = float("nan")
            Z.append(z_np[i])
            for t in tkeys:
                Y[t].append(e[t])
            MHZ.append(float(pi_norm_to_mhz(float(pi_np[i]))))
            KK.append(int(k_np[i]))
            if return_occupancy:
                OCC.append(occ_np[i].astype(np.float32).reshape(-1))
            seen += 1
        if seen >= n_samples:
            break

    Y_arr = {t: np.asarray(Y[t]) for t in tkeys}
    Y_arr["joint"] = np.zeros_like(Y_arr["mse"])
    if not np.isfinite(Y_arr["p99_ae"]).any():
        # Engine lacked denorm — keep key but mark unusable
        Y_arr["p99_ae"] = np.zeros_like(Y_arr["mse"])
    base = (
        np.asarray(Z, dtype=np.float64),
        Y_arr,
        np.asarray(MHZ),
        np.asarray(KK),
    )
    if return_occupancy:
        return (*base, np.asarray(OCC, dtype=np.float32))
    return base


def fit_error_gp(
    engine,
    loader,
    *,
    path: FeaturePath = "pred",
    target: ErrorTarget = "mse",
    gp: str = "svgp",
    n_fit: int = 4000,
    n_inducing: int = 256,
    svgp_epochs: int = 80,
    kappa: float = 0.5,
    novelty_weight: float = 0.0,
    novelty_k: int = 10,
    score_mode: str = "mu",
    joint_alpha_hm: float = 1.0,
    joint_alpha_imp: float = 1.0,
    device: torch.device | None = None,
    cfg: dict | None = None,
    verbose: bool = True,
    seed: int = 0,
) -> ErrorGPArtifact:
    """Fit error-GP on labeled loader residuals; return scoring artifact."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or {}
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print(
            f"  Fitting error-GP: path={path} target={target} gp={gp} "
            f"n_fit≤{n_fit} inducing={n_inducing} epochs={svgp_epochs} "
            f"novelty_w={novelty_weight}",
            flush=True,
        )

    Z, Y, MHZ, KK = collect_residuals(
        engine, loader, path=path, n_samples=n_fit, device=device, cfg=cfg
    )
    if Z.shape[0] < 50:
        raise RuntimeError(f"error-GP fit collected only {Z.shape[0]} samples (need ≥50)")

    if target == "joint":
        Y["joint"] = build_joint_target(
            Y["mse"], Y["imp_mse"], alpha_hm=joint_alpha_hm, alpha_imp=joint_alpha_imp
        )
    if target not in Y:
        raise KeyError(f"unknown target {target!r}; have {sorted(Y)}")

    y = Y[target]
    finite = np.isfinite(y)
    if target == "p99_ae" and int(finite.sum()) < 50:
        raise RuntimeError(
            f"error-GP target=p99_ae needs physical denorm; "
            f"only {int(finite.sum())} finite labels (need ≥50). "
            "Ensure engine.denorm_heatmap_physical is available."
        )
    if not finite.all():
        Z, y, MHZ, KK = Z[finite], y[finite], MHZ[finite], KK[finite]
        for k in list(Y.keys()):
            Y[k] = Y[k][finite]

    idx = np.arange(Z.shape[0])
    if gp == "sklearn" and len(idx) > min(n_fit, 1500):
        idx = np.random.RandomState(seed).choice(len(idx), min(n_fit, 1500), replace=False)
    elif len(idx) > n_fit:
        idx = np.random.RandomState(seed).choice(len(idx), n_fit, replace=False)

    z_fit, y_fit = Z[idx], y[idx]
    scaler = StandardScaler().fit(z_fit)
    zs = scaler.transform(z_fit)

    reg = make_regressor(
        gp, device, n_inducing=n_inducing, epochs=svgp_epochs, verbose=verbose
    ).fit(zs, y_fit)

    # Novelty bank = fit latents (scaled); calibrate novelty mean/std on fit set
    nov_fit = knn_novelty(zs, zs, k=novelty_k)
    nov_mean = float(nov_fit.mean())
    nov_std = float(nov_fit.std() or 1.0)

    if verbose:
        print(
            f"  error-GP fit done: N={len(z_fit)} latent_dim={Z.shape[1]} "
            f"y_mean={float(y_fit.mean()):.4f} y_std={float(y_fit.std()):.4f} "
            f"MHz=[{MHZ.min():.0f},{MHZ.max():.0f}] K=[{KK.min()},{KK.max()}] "
            f"novelty_mean={nov_mean:.4f}",
            flush=True,
        )

    return ErrorGPArtifact(
        regressor=reg,
        scaler=scaler,
        path=path,
        target=target,
        kappa=float(kappa),
        gp_kind=gp,
        n_fit=len(z_fit),
        novelty_weight=float(novelty_weight),
        novelty_k=int(novelty_k),
        score_mode=str(score_mode),
        z_bank_scaled=zs.copy(),
        novelty_mean=nov_mean,
        novelty_std=nov_std,
        meta={
            "mhz_min": float(MHZ.min()),
            "mhz_max": float(MHZ.max()),
            "k_min": int(KK.min()),
            "k_max": int(KK.max()),
            "y_mean": float(y_fit.mean()),
            "y_std": float(y_fit.std()),
            "joint_alpha_hm": float(joint_alpha_hm),
            "joint_alpha_imp": float(joint_alpha_imp),
            "novelty_weight": float(novelty_weight),
            "novelty_k": int(novelty_k),
            "score_mode": str(score_mode),
        },
    )


def build_fit_loader(experiment_dir: str, data_dir: str | None, groot: Path, cfg_yaml: dict | None = None):
    """Shuffled loader over the experiment's multifreq dataset (for residual collection)."""
    import importlib
    import sys

    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))
    pkg = experiment_dir.replace("\\", "/").replace("/", ".").rstrip(".")
    dl = importlib.import_module(pkg + ".codes.dataloader_multifreq")
    inf = importlib.import_module(pkg + ".codes.inference_vae")

    exp_cfg = cfg_yaml or inf.load_experiment_config(Path(experiment_dir) / "config.yaml")
    root = data_dir or exp_cfg["data_dir"]
    _, val_loader = dl.create_multifreq_data_loaders(
        root,
        batch_size=int(exp_cfg.get("val_batch_size", 128) or 128),
        num_workers=0,
        pin_memory=False,
        balance_k=False,
        balance_freq=False,
        stratify_by_k=False,
        cross_freq_pairs=False,
        drop_last_train=False,
    )
    from torch.utils.data import DataLoader as _DL

    return _DL(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=val_loader.collate_fn,
    )


def resolve_gp_error_cfg(cfg: dict) -> dict[str, Any]:
    """Defaults for ``cfg['gp_error']`` block."""
    raw = dict(cfg.get("gp_error") or {})
    return {
        "path": str(raw.get("path", "pred")),
        "target": str(raw.get("target", "p99_ae")),
        "gp": str(raw.get("gp", "svgp")),
        "kappa": float(raw.get("kappa", 0.5)),
        "n_fit": int(raw.get("n_fit", 4000)),
        "n_inducing": int(raw.get("n_inducing", 256)),
        "svgp_epochs": int(raw.get("svgp_epochs", 80)),
        "seed": int(raw.get("seed", 0)),
        "verbose": bool(raw.get("verbose", True)),
        "novelty_weight": float(raw.get("novelty_weight", 0.0)),
        "novelty_k": int(raw.get("novelty_k", 10)),
        "score_mode": str(raw.get("score_mode", "mu")),
        "joint_alpha_hm": float(raw.get("joint_alpha_hm", 1.0)),
        "joint_alpha_imp": float(raw.get("joint_alpha_imp", 1.0)),
    }


def save_artifact_meta(artifact: ErrorGPArtifact, path: Path) -> None:
    """Save lightweight JSON metadata (not the full GP weights)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": artifact.path,
        "target": artifact.target,
        "kappa": artifact.kappa,
        "gp_kind": artifact.gp_kind,
        "n_fit": artifact.n_fit,
        "novelty_weight": artifact.novelty_weight,
        "novelty_k": artifact.novelty_k,
        "score_mode": artifact.score_mode,
        "meta": artifact.meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
