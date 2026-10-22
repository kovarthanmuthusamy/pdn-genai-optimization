"""Shared data access, the surrogate and the measures of the Chapter 4 checks.

What: every check_*.py script of this folder takes its model, data and measures from here.
    Spectrum measures: the peaks that matter (simulated peaks that reach at least half the tightest
      target spectrum) found; error of the peak height in percent; shape of the curve right
      (correlation of the logarithmic spectra above 0.99); error of a frequency point in percent.
      Dips are not scored.
    Map measures: distance between the predicted and the simulated hot spot in mm; error of the
      hot-spot impedance in percent; pattern right (correlation over the board above 0.9).
    Errors are reported as the typical (median) value and the value that nine of ten cases stay within.
    Names: test placements = the dataset's placements the model was not trained on (split_heldout.txt);
    independent placements = the 480 placements generated and simulated separately (INDEPENDENT_DIR).
    Results of other checks are read with ev_in() (new output first, else the saved results);
    every result is written with ev_out().
Usage: import only (``from thesis_code.s12_evaluation import common as c``).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.frequency import freqs_mhz  # noqa: F401  (part of this module's interface)
from thesis_code.common.paths import out_dir, result_path

F0_MHZ = 200.0
N_SLOTS = 52
K_RANGE = (1, 10)
TARGET_PATHS = {t: INPUTS / f"target_impedance_{t}.npy" for t in ("0.8", "0.6", "0.5")}
REGIONS = {"1-5": (1.0, 5.0), "5-40": (5.0, 40.0), "40-180": (40.0, 180.0), "180-600": (180.0, 601.0)}
CAP_NF = {1: 100.0, 2: 47.0, 3: 22.0}        # T1, T2, T3

# ---- measures -------------------------------------------------------------------------------------
PEAK_PROMINENCE = 0.15       # ln ohm: a local maximum counts as a peak when it stands out by 16 %
PEAK_TARGET = "0.5"          # the peaks that matter are taken against the tightest target spectrum
PEAK_MATTERS = 0.5           # a simulated peak matters when it reaches PEAK_MATTERS x the target
PEAK_FREQ_TOL = 1.10         # found: a predicted peak within +-10 % in frequency
SHAPE_R = 0.99               # curve shape right: Pearson r of ln|Z| above this
PATTERN_R = 0.9              # map pattern right: Pearson r over the board pixels above this
NEAR_MM = 10.2               # hot spot "in the same place": within two pixels (2 x 5.07 mm)
AS_HOT = 0.9                 # predicted hot spot "as hot as the true one": >= 90 % of the simulated maximum
# The 64 x 64 map grid in board coordinates (mm): every eCADSTAR .map file of this board has the same
# mesh, x from -1.761 to 301.765 and y from 0.097 to 319.598, spread evenly over the 64 grid points.
PIX_X, PIX_Y = (301.765 + 1.761) / 63.0, (319.598 - 0.097) / 63.0      # 4.818 mm, 5.071 mm


def ev_in(name: str) -> Path:
    """A result of another check: this run's output if it exists, else the saved result."""
    return result_path(name)


def ev_out(name: str) -> Path:
    """Where a result of this folder is written."""
    return out_dir("s12_evaluation") / name


def data() -> Path:
    return Path(settings.DATA_DIR)


def targets() -> dict[str, np.ndarray]:
    return {t: np.load(p).reshape(-1).astype(np.float64) for t, p in TARGET_PATHS.items()}


def board_mask() -> np.ndarray:
    return np.load(INPUTS / "binary_mask.npy").astype(bool)


def slot_xy_mm() -> tuple[np.ndarray, np.ndarray]:
    """(52, 2) slot positions and (2,) position of the IC port in mm."""
    j = json.loads((INPUTS / "slot_xy_mm.json").read_text())
    return np.array(j["slots"], dtype=np.float64), np.array(j["ic_port"], dtype=np.float64)


def region_of(mhz: np.ndarray) -> np.ndarray:
    """Name of the frequency region of each frequency."""
    mhz = np.asarray(mhz, dtype=np.float64)
    out = np.empty(mhz.shape, dtype=object)
    for r, (lo, hi) in REGIONS.items():
        out[(mhz >= lo) & (mhz < hi)] = r
    return out


# ---- dataset --------------------------------------------------------------------------------------
def split_ids(which: str) -> list[str]:
    """Design ids of the 'train' or the 'test' placements (file order)."""
    fn = {"train": "split_train.txt", "test": "split_heldout.txt"}[which]
    return [l.strip() for l in (data() / fn).read_text().split() if l.strip()]


def codes_of(ids: list[str], root: Path | None = None) -> np.ndarray:
    root = root or data()
    return np.stack([np.load(root / "raw" / "layouts" / d / "type.npy").astype(np.int64).reshape(-1) for d in ids])


def spectra_of(ids: list[str], root: Path | None = None) -> np.ndarray:
    root = root or data()
    return np.stack([np.load(root / "raw" / "layouts" / d / "imp.npy").reshape(-1).astype(np.float64) for d in ids])


def maps_index(root: Path | None = None) -> dict[str, list[tuple[float, str]]]:
    """design id -> [(frequency in MHz, sample file name)] of its simulated maps."""
    root = root or data()
    out = defaultdict(list)
    with (root / "normalized" / "manifest.csv").open() as fh:
        for r in csv.DictReader(fh):
            out[r["design_id"]].append((float(r["freq_mhz"]), r["sample_name"]))
    return out


def type_mix(codes: np.ndarray) -> np.ndarray:
    """Number of different decap types on each placement (1, 2 or 3)."""
    return np.stack([(codes == t).any(1) for t in (1, 2, 3)], 1).sum(1)


def total_capacitance_nf(codes: np.ndarray) -> np.ndarray:
    return sum((codes == t).sum(-1) * c for t, c in CAP_NF.items())


def distances(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Decaps of each query placement that a database placement does not hold in the same slot with the same type."""
    kq = (q > 0).sum(1)
    out = np.empty((len(q), len(db)), dtype=np.int16)
    for s in range(0, len(q), 256):
        qq = q[s:s + 256]
        out[s:s + 256] = kq[s:s + 256, None] - ((qq[:, None, :] == db[None, :, :]) & (qq[:, None, :] > 0)).sum(2)
    return out


def closest_training(q: np.ndarray, db: np.ndarray, seed: int = 20261011) -> tuple[np.ndarray, np.ndarray]:
    """Closest-placement lookup: for each query the index of the closest database placement with the same
    number of decaps (ties broken at random with a fixed seed) and its distance in decaps."""
    rng = np.random.default_rng(seed)
    kq, kd = (q > 0).sum(1), (db > 0).sum(1)
    idx, dist = np.zeros(len(q), dtype=np.int64), np.zeros(len(q), dtype=np.int64)
    for k in np.unique(kq):
        qi, di = np.flatnonzero(kq == k), np.flatnonzero(kd == k)
        D = distances(q[qi], db[di])
        for a, row in zip(qi, D):
            best = di[row == row.min()]
            idx[a], dist[a] = int(rng.choice(best)), int(row.min())
    return idx, dist


# ---- the surrogate, used from the placement alone -------------------------------------------------
class Surrogate:
    """Placement -> latent vector (placement-only encoder) -> spectrum and maps in ohms.

    ``vae`` is the trained VAE as saved (its map statistics turn maps into ohms); ``head`` is the
    placement-only encoder with its own copy of the VAE, whose spectrum decoder is the fine-tuned one.
    """

    def __init__(self, device=None):
        import torch
        from thesis_code.s09_inference import placement_encoder
        from thesis_code.s09_inference.inference import VAEInference
        self.torch = torch
        self.dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vae = VAEInference(device=self.dev)
        self.head = placement_encoder.load(device=self.dev)

    def latent(self, codes: np.ndarray, k: np.ndarray | None = None):
        from thesis_code.common.placement import ensure_batch_onehot
        cd = self.torch.as_tensor(np.asarray(codes), dtype=self.torch.long, device=self.dev).reshape(-1, N_SLOTS)
        oh = ensure_batch_onehot(cd, 3)
        kk = (cd > 0).sum(1).long() if k is None else self.torch.as_tensor(np.asarray(k), dtype=self.torch.long, device=self.dev)
        with self.torch.inference_mode():
            return self.head.net(oh, kk), oh, kk

    def spectra(self, codes: np.ndarray, k: np.ndarray | None = None) -> np.ndarray:
        """(N, 231) ohms. ``k`` overrides the budget fed to the model (default: the number of decaps)."""
        if k is None:
            return self.head.predict_ohm(np.asarray(codes).reshape(-1, N_SLOTS))
        out = []
        for s in range(0, len(codes), 4096):
            z, oh, kk = self.latent(codes[s:s + 4096], k[s:s + 4096])
            with self.torch.inference_mode():
                m = self.head.model
                y = m.impedance_decoder(self.torch.cat([m._layout_latent_slice(z), m.k_embedding(kk), m.occupancy_encoder(oh)], 1))
            out.append(np.exp(y.float().cpu().numpy().reshape(len(z), -1) * self.head.log_std + self.head.log_mean))
        return np.concatenate(out)

    def maps(self, codes: np.ndarray, mhz: np.ndarray, k: np.ndarray | None = None, batch: int = 512) -> np.ndarray:
        """(N, 64, 64) ohms: the map of placement n at frequency mhz[n]."""
        from thesis_code.common.board import zero_offboard
        from thesis_code.common.frequency import pi_freq_mhz_to_norm
        t = self.torch
        out = []
        for s in range(0, len(codes), batch):
            z, oh, kk = self.latent(codes[s:s + batch], None if k is None else k[s:s + batch])
            f = t.as_tensor(np.asarray(mhz[s:s + batch]), dtype=t.float32, device=self.dev)
            pif = t.as_tensor([pi_freq_mhz_to_norm(float(x)) for x in mhz[s:s + batch]], dtype=t.float32, device=self.dev)
            with t.inference_mode():
                g = zero_offboard(self.head.model.decode(z, kk, pif, occupancy=oh)[0])
                out.append(self.vae.denorm_heatmap_physical(g, mhz=f).reshape(len(z), 64, 64).cpu().numpy())
        return np.concatenate(out)


def search_run_dir(run_dir: str) -> Path:
    """A run folder of a runs file: absolute as written by this folder, or relative to the repository
    in the saved runs file."""
    p = Path(run_dir)
    return p if p.is_absolute() else settings.REPO / p


# ---- spectrum measures ----------------------------------------------------------------------------
def peaks(z: np.ndarray) -> np.ndarray:
    """Indices of the impedance peaks of one spectrum (local maxima of ln|Z| with PEAK_PROMINENCE)."""
    from scipy.signal import find_peaks
    return find_peaks(np.log(z), prominence=PEAK_PROMINENCE)[0]


def peak_table(pred: np.ndarray, sim: np.ndarray, f: np.ndarray) -> np.ndarray:
    """One row per SIMULATED peak of every placement:
    [placement, index of the frequency point, simulated height (ohm), found (0/1), predicted height
    (ohm, nan if not found), predicted frequency (MHz, nan)].
    A peak is found when the prediction has a peak within +-10 % in frequency; the closest one is taken."""
    rows = []
    for n in range(len(sim)):
        ps, pp = peaks(sim[n]), peaks(pred[n])
        for i in ps:
            near = pp[(f[pp] >= f[i] / PEAK_FREQ_TOL) & (f[pp] <= f[i] * PEAK_FREQ_TOL)] if pp.size else pp
            if near.size:
                j = near[np.argmin(np.abs(np.log(f[near] / f[i])))]
                rows.append((n, i, sim[n, i], 1.0, pred[n, j], f[j]))
            else:
                rows.append((n, i, sim[n, i], 0.0, np.nan, np.nan))
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def matters(pt: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rows of a peak table whose simulated peak reaches at least PEAK_MATTERS x the target spectrum."""
    return pt[:, 2] / target[pt[:, 1].astype(int)] >= PEAK_MATTERS


def peak_stats(pt: np.ndarray, f: np.ndarray) -> dict:
    """Found share and height error of the peaks in the rows ``pt`` of a peak table."""
    h = (pt[:, 4] / pt[:, 2] - 1) * 100.0                          # signed error of the height, found peaks
    return {"n": int(len(pt)), "found_pct": share(pt[:, 3]), "height_error_typical_pct": typical(np.abs(h)),
            "height_error_9of10_pct": nine_of_ten(np.abs(h)), "height_signed_typical_pct": typical(h),
            "too_low_pct": share(h[np.isfinite(h)] < 0),
            "frequency_error_typical_pct": typical(np.abs(pt[:, 5] / f[pt[:, 1].astype(int)] - 1) * 100.0)}


def shape_r(pred: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Pearson r of ln|Z| over the frequency points, per placement."""
    a, b = np.log(pred), np.log(sim)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-30)


def typical(x) -> float:
    """Median of the finite values, in the unit of x."""
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def nine_of_ten(x) -> float:
    """The value that nine of ten cases stay within (90th percentile of the finite values)."""
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    return float(np.quantile(x, 0.9)) if x.size else float("nan")


def share(x) -> float:
    """Share of true values in percent."""
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    return float(x.mean() * 100.0) if x.size else float("nan")


def spectrum_summary(pred: np.ndarray, sim: np.ndarray, f: np.ndarray, target: np.ndarray, *, pt: np.ndarray | None = None,
                     placements: np.ndarray | None = None, freq_sel: np.ndarray | None = None) -> dict:
    """The spectrum measures of the thesis. ``placements`` (bool per placement) and ``freq_sel`` (bool per
    frequency point) restrict the point error and the peaks; the shape of the curve is always taken over
    the whole spectrum. ``pt`` = peak_table(pred, sim, f) if it has been computed already."""
    pt = peak_table(pred, sim, f) if pt is None else pt
    pl = np.ones(len(sim), bool) if placements is None else placements
    fs = np.ones(len(f), bool) if freq_sel is None else freq_sel
    pt = pt[pl[pt[:, 0].astype(int)] & fs[pt[:, 1].astype(int)]] if len(pt) else pt
    err = np.abs(pred[pl][:, fs] / sim[pl][:, fs] - 1) * 100.0
    return {"n_placements": int(pl.sum()), "point_error_typical_pct": typical(err), "point_error_9of10_pct": nine_of_ten(err),
            "point_signed_typical_pct": typical((pred[pl][:, fs] / sim[pl][:, fs] - 1) * 100.0),
            "shape_right_pct": share(shape_r(pred[pl], sim[pl]) > SHAPE_R),
            "peaks_that_matter": peak_stats(pt[matters(pt, target)], f), "all_peaks": peak_stats(pt, f)}


def worst_ratio(z: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Worst-case ratio of each spectrum to the target (a placement passes when it is at most 1)."""
    return (z / target).max(-1)


# ---- map measures ---------------------------------------------------------------------------------
def map_measures(p: np.ndarray, r: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """(distance between the predicted and the simulated hot spot in mm, predicted / simulated hot-spot
    impedance, Pearson r over the board pixels) of one predicted map ``p`` against the simulated ``r``.
    The hot spot is the board pixel with the highest impedance."""
    pm, rm = np.where(mask, p, -np.inf), np.where(mask, r, -np.inf)
    (pi, pj), (ri, rj) = np.unravel_index(pm.argmax(), pm.shape), np.unravel_index(rm.argmax(), rm.shape)
    dist = float(np.hypot((pj - rj) * PIX_X, (pi - ri) * PIX_Y))
    pv, rv = p[mask].astype(np.float64), r[mask].astype(np.float64)
    pc, rc = pv - pv.mean(), rv - rv.mean()
    cor = float((pc * rc).sum() / (np.sqrt((pc ** 2).sum() * (rc ** 2).sum()) + 1e-30))
    return dist, float(pv.max() / rv.max()), cor


def map_summary(rows: np.ndarray) -> dict:
    """rows (N, 3) of map_measures -> the map measures of the thesis (rows without a map are left out)."""
    rows = rows[np.isfinite(rows[:, 0])]
    amp = np.abs(rows[:, 1] - 1) * 100.0
    return {"n": int(len(rows)), "hot_spot_distance_typical_mm": typical(rows[:, 0]),
            "hot_spot_distance_9of10_mm": nine_of_ten(rows[:, 0]),
            "hot_spot_same_pixel_pct": share(rows[:, 0] < 1e-9), "hot_spot_within_10mm_pct": share(rows[:, 0] <= NEAR_MM),
            "hot_spot_error_typical_pct": typical(amp), "hot_spot_error_9of10_pct": nine_of_ten(amp),
            "hot_spot_signed_typical_pct": typical((rows[:, 1] - 1) * 100.0),
            "pattern_right_pct": share(rows[:, 2] > PATTERN_R)}


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")
