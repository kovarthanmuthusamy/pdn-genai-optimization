"""eCADSTAR check of the page's placement: simulate its spectrum (and the map at the picked frequency) and
compare with the model.

What: ``EcadRunner.start`` runs one eCADSTAR job at a time, in a background thread, through thesis_code's
    ``s12_evaluation.ecad`` (``spectra_cached`` / ``maps_cached``: batches start with a C1 group, runs
    resume, results are kept in STORE so a placement is simulated once). Before a job it asks
    ``ecad.busy()`` (engineer.exe, dataset or queue jobs) and refuses while eCADSTAR is in use; it never
    queues behind another job. ``compare`` reads STORE only and never simulates: the eCADSTAR side of the
    page's table for one target (worst-case ratio, its frequency, pass with no margin, map + hot spot).
Why a thread: one spectrum batch takes about 16 s (2026-10 runs), so the request returns at once and the
    page polls ``status``.
Usage: import only -- ``EcadRunner(engine)``; ``start``, ``status``, ``compare``.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from assistant_web.board import MAP_X0, MAP_Y0, PIX_X, PIX_Y
from thesis_code.common import settings
from thesis_code.s12_evaluation import ecad

STORE = Path(__file__).resolve().parent / "cache" / "ecad"   # spectra.npz / maps.npz + one folder per run


class EcadBusy(Exception):
    """eCADSTAR is in use (another job of this page, or engineer.exe / a dataset job)."""


@contextmanager
def _thesis_run_settings():
    """thesis_code writes only below settings.OUTPUT_DIR and starts eCADSTAR only with ECADSTAR_RUN True;
    both are set for the job and restored after it (settings.py itself is not changed)."""
    old = settings.OUTPUT_DIR, settings.ECADSTAR_RUN
    settings.OUTPUT_DIR, settings.ECADSTAR_RUN = STORE.parent, True
    try:
        yield
    finally:
        settings.OUTPUT_DIR, settings.ECADSTAR_RUN = old


def _saved(kind: str, codes: np.ndarray, mhz: float | None = None) -> np.ndarray | None:
    """The saved simulation of ``codes`` (and ``mhz`` for a map, matched to 0.1 MHz as ecad stores it)."""
    f = STORE / f"{kind}.npz"
    if not f.is_file():
        return None
    d = np.load(f)
    hit = np.flatnonzero((d["codes"] == codes[None]).all(1))
    if mhz is not None:
        hit = [i for i in hit if f"{float(d['mhz'][i]):.1f}" == f"{float(mhz):.1f}"]
    return d["values"][hit[0]] if len(hit) else None


class EcadRunner:
    def __init__(self, engine, log=print):
        self.eng, self.log = engine, log
        self._lock = threading.Lock()
        self._state: dict = {"state": "idle"}

    # ---- job ---------------------------------------------------------------------------------
    def status(self) -> dict:
        s = dict(self._state)
        if "t0" in s:
            s["seconds"] = round((s.pop("t1", None) or time.time()) - s.pop("t0"), 1)
        return s

    def start(self, codes, index: int | None) -> dict:
        """Simulate what is missing for ``codes``: the spectrum, and the map at frequency point ``index``."""
        c = self.eng.codes(codes)
        k = int((c != 0).sum())
        if not self.eng.k_range[0] <= k <= self.eng.k_range[1]:
            raise ValueError(f"place {self.eng.k_range[0]}..{self.eng.k_range[1]} decaps first")
        mhz = None if index is None else float(self.eng.f[int(np.clip(index, 0, len(self.eng.f) - 1))])
        with self._lock:
            if self._state["state"] == "running":
                raise EcadBusy("an eCADSTAR check of this page is still running")
            need_spec = _saved("spectra", c) is None
            need_map = mhz is not None and _saved("maps", c, mhz) is None
            if not (need_spec or need_map):
                return {"state": "done", "codes": c.tolist(), "mhz": mhz, "saved": True}
            why = ecad.busy()
            if why is not None:
                raise EcadBusy(f"eCADSTAR is busy: {why}")
            self._state = {"state": "running", "codes": c.tolist(), "mhz": mhz,
                           "phase": "spectrum" if need_spec else "map", "t0": time.time()}
            threading.Thread(target=self._job, args=(c, mhz, need_spec, need_map), daemon=True).start()
        return self.status()

    def _job(self, c: np.ndarray, mhz: float | None, need_spec: bool, need_map: bool) -> None:
        try:
            with _thesis_run_settings():
                if need_spec:
                    self.log(f"eCADSTAR: spectrum of K = {int((c != 0).sum())} ...")
                    ecad.spectra_cached(STORE, c[None])
                if need_map:
                    self._state["phase"] = "map"
                    self.log(f"eCADSTAR: map at {mhz:g} MHz ...")
                    ecad.maps_cached(STORE, c[None], np.array([mhz]))
            self._state.update(state="done", t1=time.time())
            self.log(f"eCADSTAR: done in {self.status()['seconds']} s")
        except BaseException as e:                       # thesis_code stops with SystemExit on a failed run
            self._state.update(state="error", message=str(e) or type(e).__name__, t1=time.time())
            self.log(f"eCADSTAR: failed: {e}")

    # ---- comparison ---------------------------------------------------------------------------
    def compare(self, codes, target: str, index: int | None) -> dict:
        """The saved eCADSTAR results of ``codes`` for one target (the model side is what the page already
        shows): spectrum, worst-case ratio with its frequency and pass (no margin), and the map at
        frequency point ``index`` with its hot spot. A part not simulated yet is None."""
        eng = self.eng
        c = eng.codes(codes)
        tgt, _ = eng._target(target)
        out = {"codes": c.tolist(), "spectrum": None, "worst": None, "map": None}
        z_sim = _saved("spectra", c)
        if z_sim is not None:
            out["spectrum"] = np.asarray(z_sim, dtype=np.float64).tolist()
            out["worst"] = self._worst(np.asarray(z_sim, dtype=np.float64), tgt)
        if index is not None:
            index = int(np.clip(index, 0, len(eng.f) - 1))
            m_sim = _saved("maps", c, float(eng.f[index]))
            if m_sim is not None:
                out["map"] = {"index": index, "mhz": float(eng.f[index]), **self._map(m_sim)}
        return out

    def _worst(self, z: np.ndarray, tgt: np.ndarray) -> dict:
        r = z[self.eng.band] / np.maximum(tgt[self.eng.band], 1e-12)
        i = int(np.argmax(r))
        return {"ratio": float(r[i]), "mhz": float(self.eng.f[self.eng.band[i]]), "passes": bool(r[i] <= 1.0)}

    def _map(self, m: np.ndarray) -> dict:
        mask = self.eng.mask
        m = np.asarray(m, dtype=np.float64)
        on = np.where(mask, m, -np.inf)
        r, q = np.unravel_index(int(np.argmax(on)), on.shape)
        vals = np.where(mask, m, np.nan)
        return {"values": [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in vals],
                "hot": {"x": float(MAP_X0 + q * PIX_X), "y": float(MAP_Y0 + r * PIX_Y), "ohm": float(m[r, q])}}
