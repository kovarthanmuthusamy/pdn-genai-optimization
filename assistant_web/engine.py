"""Model layer of the web assistant: the thesis surrogate, loaded once, and the questions the page asks.

What: loads the thesis model through ``thesis_code`` (``settings.WEIGHTS_DIR / CHECKPOINT`` = exp096
    ``last_model.pt``, placement-only encoder ``settings.HEAD`` = H1_spec_hm200): the full surrogate on
    the first GPU (spectrum + maps) and, for every further GPU, a worker process with one more copy of
    the placement-only encoder. It answers four questions:
      check      verdict of a placement for one target, safe (measured margin) and direct (no margin),
                 with the predicted spectrum and the worst frequency (``suggest.verdict``);
      additions  every empty slot with every part (K + 1), ranked by the predicted worst-case ratio;
      improve    same-K placements up to 2 changes away with a lower worst-case ratio, lowest first;
      map        the predicted map at one of the 231 frequencies, with the hot spot;
      fixes      the nearest passing placements (``suggest.find_fixes``, the thesis assistant, with
                 REMOVE on: fewer changes first, then fewer decaps; for a passing placement, the nearest
                 ones with fewer decaps), judged by ``SplitJudge``: the thesis judge with its candidates
                 spread over the GPUs. A newer fix search cancels an older one.
    Nothing in ``thesis_code`` is changed; verdicts and proposals are those of the thesis assistant.
    ``check``, ``additions`` and ``fixes`` take an optional ``margin`` (0..MARGIN_MAX) for the page's margin
    slider; None (the default) is the thesis margin of the target, so the default answers stay the thesis's.
Why processes, not threads: two threads on two GPUs gave 1.2x (measured 2026-10-20: 4.22 s -> 3.47 s for
    131,072 placements); two processes run fully in parallel (65,536 each in 2.42 / 2.48 s, alone 2.43 s).
Usage: ``from assistant_web.engine import Engine; eng = Engine(gpus=(0, 1))``. Import-only.
Cache: ``assistant_web/cache/map_scale.json`` (colour scale of the maps per frequency).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import torch

from assistant_web.board import MAP_X0, MAP_Y0, PIX_X, PIX_Y, board_mask
from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.frequency import freqs_mhz
from thesis_code.s09_inference import placement_encoder
from thesis_code.s11_assistant import suggest as S
from thesis_code.s12_evaluation.common import Surrogate

N_SLOTS = 52
N_TYPES = 3
TARGETS = ("0.8", "0.6", "0.5")
MARGIN_MAX = 0.6            # largest margin of the page's slider (thesis margins: 0.25 / 0.37 / 0.27)
PARTS = {1: {"name": "T1", "value": "100 nF", "esl": "222 pH", "esr": "8.9 mΩ"},
         2: {"name": "T2", "value": "47 nF", "esl": "154 pH", "esr": "21.4 mΩ"},
         3: {"name": "T3", "value": "22 nF", "esl": "142 pH", "esr": "25.2 mΩ"}}
CACHE_DIR = Path(__file__).resolve().parent / "cache"
PREDICT_BATCH = 8192        # the batch of VAEImpedance.predict_ohm; splits fall on its multiples, so every
                            # GPU sees the same batches as the one-GPU thesis judge (identical numbers)
SPLIT_MIN = 4 * PREDICT_BATCH   # smaller blocks stay on the first GPU
SPLIT_CHUNK = 8 * PREDICT_BATCH  # rows per split block; a cancelled search stops between blocks
IMPROVE_RADIUS = 2          # improve search: every placement up to this many changes (author, 2026-10-22)
MAP_REF_K = (3, 8)          # colour scale of the maps: reference placements with these budgets ...
MAP_REF_PER_K = 25          # ... this many per budget ...
MAP_REF_SEED = 0
MAP_PCT = (1.0, 99.0)       # ... and these percentiles of their board values


class Cancelled(Exception):
    """A newer fix search started; this one stops."""


def worst_ratios(z: np.ndarray, tgt: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Worst-case ratio per placement, computed as in ``suggest.Judge.assess``."""
    return (z / np.maximum(tgt, 1e-12))[:, band].max(1)


# ---- worker process: one placement-only encoder on one more GPU ------------------------------
def _worker_main(conn, gpu: int) -> None:
    head = placement_encoder.load(device=torch.device(f"cuda:{gpu}"))
    conn.send(("ready", gpu))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):                       # the main process went away
            break
        if msg[0] == "stop":
            break
        _, codes, tgt, band = msg
        try:
            z = head.predict_ohm(codes.astype(np.int64), batch=PREDICT_BATCH)
            conn.send(("ok", worst_ratios(z, tgt, band)))
        except Exception as e:  # noqa: BLE001 -- reported to the main process
            conn.send(("error", repr(e)))


class GpuWorker:
    def __init__(self, gpu: int):
        ctx = mp.get_context("spawn")                     # CUDA needs a fresh process
        self.gpu = gpu
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child, gpu), daemon=True)
        self.proc.start()
        self.lock = threading.Lock()                      # one job at a time per worker

    def wait_ready(self, timeout: float = 300.0) -> None:
        if not self.conn.poll(timeout):
            raise RuntimeError(f"worker on GPU {self.gpu} did not start")
        msg = self.conn.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"worker on GPU {self.gpu}: {msg}")

    def stop(self) -> None:
        try:
            self.conn.send(("stop",))
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=5)


class SplitJudge(S.Judge):
    """The thesis judge (``suggest.Judge``: PASS when Z <= (1 - margin) x target everywhere) with the
    candidates spread over the GPUs. ``assess`` returns what ``Judge.assess`` returns for one member;
    only where the spectra are computed changes, so ``find_fixes`` ranks exactly as in the thesis."""

    def __init__(self, engine: "Engine", margin: float, tgt: np.ndarray, band: np.ndarray, cancelled=None):
        super().__init__([(f"vae:{settings.HEAD}", engine.heads[0], float(margin))], tgt, band)
        self.engine = engine
        self.cancelled = cancelled or (lambda: False)
        self.n_scored = 0

    def assess(self, codes):
        codes = np.asarray(codes, np.int64).reshape(-1, N_SLOTS)
        name, _, mg = self.members[0]
        r = self.engine.split_ratios(codes, self.tgt, self.band, self.cancelled)
        self.n_scored += len(codes)
        passes = np.ones(len(r), dtype=bool) & (r <= 1.0 - mg)
        score = np.maximum(np.zeros(len(r)), r / (1.0 - mg))
        return passes, score, {name: r}


def _edits(codes: np.ndarray, n_types: int, remove: bool) -> list[np.ndarray]:
    """``suggest._edits`` with ADD off and REMOVE as an argument (the thesis reads module flags):
    RETYPE, MOVE, then REMOVE when K > 1, in the thesis order."""
    out: list[np.ndarray] = []
    occ, empty = np.flatnonzero(codes), np.flatnonzero(codes == 0)
    for s in occ:                                    # RETYPE
        for t in range(1, n_types + 1):
            if t != codes[s]:
                c = codes.copy(); c[s] = t; out.append(c)
    for s in occ:                                    # MOVE
        for d in empty:
            c = codes.copy(); c[d] = c[s]; c[s] = 0; out.append(c)
    if remove and len(occ) > 1:                      # REMOVE
        for s in occ:
            c = codes.copy(); c[s] = 0; out.append(c)
    return out


def neighbours_fast(codes: np.ndarray, radius: int, n_types: int, rng, remove: bool = False) -> np.ndarray:
    """``suggest.neighbours_at`` with the same result and the same generator state, faster. At its sampled
    level (radius 3) the thesis builds every single-edit neighbour of each frontier placement (245 arrays
    for K = 5, 3.2 million in all), keeps 64 drawn by ``rng.choice(len(cand), 64)`` and de-duplicates
    them in order with tuple keys until RADIUS3_BUDGET. With ADD off, a placement has K (T - 1) retypes,
    K (52 - K) moves and (with ``remove``, as the thesis's ALLOW_REMOVE) K removals in a fixed order, so
    the same draw can be made first and only the 64 drawn neighbours built; the de-duplication uses byte
    keys. Same draws, same order, same stop (tested against ``suggest.neighbours_at``)."""
    if S.ALLOW_ADD:
        return S.neighbours_at(codes, radius, n_types, rng)
    remove = remove or S.ALLOW_REMOVE
    seen = {tuple(codes.tolist())}
    frontier = [codes]
    for r in range(radius):
        budget = S.RADIUS3_BUDGET if r >= 2 else None
        if budget is not None and r == radius - 1:
            return _sampled_level(frontier, seen, n_types, rng, budget, remove)
        nxt: dict[tuple, np.ndarray] = {}
        for base in frontier:
            cand = _edits(base, n_types, remove)
            if budget is not None and len(cand) > 0:
                idx = rng.choice(len(cand), size=min(len(cand), 64), replace=False)
                cand = [cand[i] for i in idx]
            for c in cand:
                k = tuple(c.tolist())
                if k not in seen:
                    nxt[k] = c
            if budget is not None and len(nxt) >= budget:
                break
        seen.update(nxt.keys())
        frontier = list(nxt.values())
        if not frontier:
            break
    return np.array(frontier, dtype=np.int64) if frontier else np.zeros((0, N_SLOTS), np.int64)


def _sampled_level(frontier: list, seen: set, n_types: int, rng, budget: int, remove: bool = False) -> np.ndarray:
    """The last, sampled level of ``neighbours_at``: per frontier placement the same draw, then only the
    drawn neighbours are built."""
    seen_b = {np.asarray(t, dtype=np.int8).tobytes() for t in seen}
    kept: list[bytes] = []
    for base in frontier:
        occ, empty = np.flatnonzero(base), np.flatnonzero(base == 0)
        n_re = len(occ) * (n_types - 1)
        n_mv = n_re + len(occ) * len(empty)
        n = n_mv + (len(occ) if remove and len(occ) > 1 else 0)
        if n == 0:
            continue
        idx = rng.choice(n, size=min(n, 64), replace=False)
        cand = np.repeat(base[None].astype(np.int8), len(idx), 0)
        row = np.arange(len(idx))
        re = idx < n_re                                         # RETYPE: slot-major, types ascending
        sl = occ[idx[re] // (n_types - 1)]
        jj = idx[re] % (n_types - 1)
        cand[row[re], sl] = jj + 1 + (jj + 1 >= base[sl])       # the jj-th type other than the current one
        mv = ~re & (idx < n_mv)
        q = idx[mv] - n_re                                      # MOVE: slot-major, targets ascending
        if q.size:
            sl, d = occ[q // len(empty)], empty[q % len(empty)]
            cand[row[mv], d] = base[sl]
            cand[row[mv], sl] = 0
        rm = idx >= n_mv                                        # REMOVE: slots ascending
        cand[row[rm], occ[idx[rm] - n_mv]] = 0
        buf = cand.tobytes()
        for j in range(len(idx)):
            key = buf[j * N_SLOTS:(j + 1) * N_SLOTS]
            if key not in seen_b:
                seen_b.add(key)
                kept.append(key)
        if len(kept) >= budget:
            break
    if not kept:
        return np.zeros((0, N_SLOTS), np.int64)
    return np.frombuffer(b"".join(kept), dtype=np.int8).reshape(-1, N_SLOTS).astype(np.int64)


def find_fixes_fast(codes: np.ndarray, judge, n_types: int, *, max_radius: int, min_suggestions: int,
                    seed: int, log=print, remove: bool = False, k_max: int | None = None) -> list[dict]:
    """``suggest.find_fixes`` line for line, with ``neighbours_fast`` in place of ``neighbours_at``.
    ``remove`` adds REMOVE to the edits (the thesis's ALLOW_REMOVE); ``k_max`` keeps only candidates with
    at most that many decaps (fewer decaps for a placement that already passes). Ranked by the number of
    changed decaps, then fewer decaps (author, 2026-10-22), then type changes before moves, then headroom;
    with K fixed this is the thesis order."""
    rng = np.random.default_rng(seed)
    k_lo, k_hi = judge.k_range
    if k_max is not None:
        k_hi = min(k_hi, k_max)
    found: list[dict] = []
    for r in range(1, max_radius + 1):
        cand = neighbours_fast(codes, r, n_types, rng, remove)
        dropped = 0
        if cand.shape[0]:
            kc = (cand != 0).sum(1)
            inside = (kc >= k_lo) & (kc <= k_hi)
            dropped, cand = int((~inside).sum()), cand[inside]
        if cand.shape[0] == 0:
            continue
        passes, score, ratios = judge.assess(cand)
        ok = np.flatnonzero(passes)
        if log:
            log(f"  radius {r}: {cand.shape[0]:,} candidate placements, {len(ok):,} pass"
                + (f"  ({dropped:,} skipped: K outside trained {k_lo}..{k_hi})" if dropped else ""))
        for i in ok:
            found.append({"codes": cand[i], "score": float(score[i]), "edits": S.n_edits(codes, cand[i]),
                          "moves": S.n_moves(codes, cand[i]), "K": int((cand[i] != 0).sum()),
                          "radius": r, "ratios": {k: float(v[i]) for k, v in ratios.items()}})
        found = [x for x in found if x["edits"] <= r]
        if len(found) >= min_suggestions:
            break
    found.sort(key=lambda x: (x["edits"], x["K"], x["moves"], x["score"]))
    return found


def change_list(base: np.ndarray, cand: np.ndarray) -> list[dict]:
    """How ``cand`` differs from ``base``, paired as in ``suggest.edit_summary`` (retypes, then moves
    in slot order, then removals / additions)."""
    b_occ, c_occ = set(np.flatnonzero(base).tolist()), set(np.flatnonzero(cand).tolist())
    items = [{"kind": "retype", "slot": s + 1, "from": int(base[s]), "to": int(cand[s])}
             for s in sorted(b_occ & c_occ) if base[s] != cand[s]]
    removed, added = sorted(b_occ - c_occ), sorted(c_occ - b_occ)
    while removed and added:
        s, d = removed.pop(0), added.pop(0)
        items.append({"kind": "move", "from_slot": s + 1, "to_slot": d + 1, "from": int(base[s]), "to": int(cand[d])})
    items += [{"kind": "remove", "slot": s + 1, "from": int(base[s])} for s in removed]
    items += [{"kind": "add", "slot": d + 1, "to": int(cand[d])} for d in added]
    return items


def change_text(items: list[dict]) -> str:
    """Thesis wording: 'C19: T3 -> T1', 'move C30 -> C47', 'move C38 (T3) -> C31 as T1'."""
    t = lambda c: PARTS[c]["name"]  # noqa: E731
    words = []
    for it in items:
        if it["kind"] == "retype":
            words.append(f"C{it['slot']}: {t(it['from'])} → {t(it['to'])}")
        elif it["kind"] == "move":
            words.append(f"move C{it['from_slot']} → C{it['to_slot']}" if it["from"] == it["to"]
                         else f"move C{it['from_slot']} ({t(it['from'])}) → C{it['to_slot']} as {t(it['to'])}")
        elif it["kind"] == "remove":
            words.append(f"remove C{it['slot']} ({t(it['from'])})")
        else:
            words.append(f"add {t(it['to'])} in C{it['slot']}")
    return "; ".join(words) or "(no change)"


class Engine:
    def __init__(self, gpus=(0, 1), log=print):
        self.log = log
        n_gpu = torch.cuda.device_count()
        gpus = [int(g) for g in gpus if int(g) < n_gpu]
        dev = torch.device(f"cuda:{gpus[0]}") if gpus else torch.device("cpu")
        t0 = time.time()
        self.workers = [GpuWorker(g) for g in gpus[1:]]     # load in parallel with the main copy
        self.sur = Surrogate(device=dev)                      # spectra + maps (thesis_code s12 common)
        self.heads = [self.sur.head]
        for w in self.workers:
            w.wait_ready()
        self.devices = [str(dev)] + [f"cuda:{w.gpu} (worker)" for w in self.workers]
        self.f = freqs_mhz()
        self.band = np.flatnonzero((self.f >= S.CHECK_BAND_MHZ[0]) & (self.f <= S.CHECK_BAND_MHZ[1]))
        self.target_paths = {t: INPUTS / f"target_impedance_{t}.npy" for t in TARGETS}
        self.targets = {t: np.load(p).reshape(-1) for t, p in self.target_paths.items()}   # as suggest.main
        self.margins = {t: float(placement_encoder.margin(p)) for t, p in self.target_paths.items()}
        self.bare = np.load(INPUTS / "bare_impedance.npy").reshape(-1).astype(np.float64)
        self.k_range = tuple(int(k) for k in self.heads[0].k_range)
        self.mask = board_mask()
        self._fix_gen = 0
        self._imp_gen = 0                 # improve searches cancel each other, not the fix search
        self._gen_lock = threading.Lock()
        self.log(f"model loaded on {', '.join(self.devices)} in {time.time() - t0:.1f} s "
                 f"(margins {', '.join(f'{t}: {m:.2f}' for t, m in self.margins.items())})")
        self.map_scale = self._map_scale()

    def close(self) -> None:
        for w in self.workers:
            w.stop()

    # ---- inputs ---------------------------------------------------------------------------
    def codes(self, codes) -> np.ndarray:
        c = np.asarray(codes, dtype=np.int64).reshape(-1)
        if c.size != N_SLOTS:
            raise ValueError(f"a placement has {N_SLOTS} slots, got {c.size}")
        if c.min() < 0 or c.max() > N_TYPES:
            raise ValueError(f"part codes are 0 (empty) or 1..{N_TYPES}")
        return c

    def _target(self, target: str, margin: float | None = None) -> tuple[np.ndarray, float]:
        """Target curve and safe margin: the thesis margin of the target, or ``margin`` when given."""
        if target not in self.targets:
            raise ValueError(f"target must be one of {', '.join(TARGETS)}")
        if margin is None:
            return self.targets[target], self.margins[target]
        margin = float(margin)
        if not 0.0 <= margin <= MARGIN_MAX:
            raise ValueError(f"margin must be 0..{MARGIN_MAX}")
        return self.targets[target], margin

    def _k_ok(self, k: int) -> bool:
        return self.k_range[0] <= k <= self.k_range[1]

    # ---- spectra, split over the GPUs ------------------------------------------------------
    def split_ratios(self, codes: np.ndarray, tgt: np.ndarray, band: np.ndarray, cancelled=None) -> np.ndarray:
        """Worst-case ratio of every placement. Blocks of SPLIT_CHUNK rows; each block is cut at
        multiples of PREDICT_BATCH, the workers take their parts while this process predicts the first."""
        codes = np.asarray(codes, np.int64).reshape(-1, N_SLOTS)
        out = []
        for s in range(0, len(codes), SPLIT_CHUNK):
            if cancelled is not None and cancelled():
                raise Cancelled()
            block = codes[s:s + SPLIT_CHUNK]
            n_dev = 1 + len(self.workers)
            if len(block) < SPLIT_MIN or n_dev == 1:
                out.append(worst_ratios(self.heads[0].predict_ohm(block, batch=PREDICT_BATCH), tgt, band))
                continue
            per = -(-len(block) // n_dev)                               # ceil
            per = -(-per // PREDICT_BATCH) * PREDICT_BATCH               # up to a multiple of the batch
            cuts = [min(i * per, len(block)) for i in range(n_dev + 1)]
            sent = []
            for w, (a, b) in zip(self.workers, zip(cuts[1:-1], cuts[2:])):
                if b > a:
                    w.lock.acquire()
                    w.conn.send(("ratios", block[a:b].astype(np.int8), tgt, band))
                    sent.append(w)
            replies = []
            try:
                local = worst_ratios(self.heads[0].predict_ohm(block[:cuts[1]], batch=PREDICT_BATCH), tgt, band)
            finally:                                                    # always collect what was sent
                for w in sent:
                    try:
                        replies.append((w, w.conn.recv()))
                    finally:
                        w.lock.release()
            parts = [local]
            for w, (kind, val) in replies:
                if kind != "ok":
                    raise RuntimeError(f"worker on GPU {w.gpu}: {val}")
                parts.append(val)
            out.append(np.concatenate(parts))
        return np.concatenate(out) if out else np.zeros(0)

    # ---- check ----------------------------------------------------------------------------
    def check(self, codes, target: str, n_additions: int = 5, margin: float | None = None) -> dict:
        c = self.codes(codes)
        tgt, mg = self._target(target, margin)
        k = int((c != 0).sum())
        if k > self.k_range[1]:
            raise ValueError(f"the model was trained for up to {self.k_range[1]} decaps")
        if k == 0:
            z, kind = self.bare, "bare"
        else:
            z, kind = self.heads[0].predict_ohm(c[None])[0].astype(np.float64), "model"
        ratio = z[self.band] / np.maximum(tgt[self.band], 1e-12)
        return {"kind": kind, "K": k, "target": target, "margin": mg, "thesis_margin": self.margins[target],
                "spectrum": z.tolist(),
                "safe": S.verdict(z, tgt, self.band, self.f, mg),
                "direct": S.verdict(z, tgt, self.band, self.f, 0.0),
                "worst_index": int(self.band[int(np.argmax(ratio))]),
                "additions": self.additions(c, target, n_additions, mg) if k + 1 <= self.k_range[1] else []}

    def additions(self, codes, target: str, n: int = 5, margin: float | None = None) -> list[dict]:
        """Every empty slot with every part, ranked by the predicted worst-case ratio (lowest first)."""
        c = self.codes(codes)
        tgt, mg = self._target(target, margin)
        if not self._k_ok(int((c != 0).sum()) + 1):
            return []
        cand, meta = [], []
        for s in np.flatnonzero(c == 0):
            for t in range(1, N_TYPES + 1):
                x = c.copy(); x[s] = t
                cand.append(x); meta.append((int(s), t))
        z = self.heads[0].predict_ohm(np.array(cand))
        ratio = worst_ratios(z, tgt, self.band)
        rows = []
        for i in np.argsort(ratio, kind="stable")[:n]:
            s, t = meta[i]
            rows.append({"codes": cand[i].tolist(), "slot": s + 1, "type": t,
                         "text": f"add {PARTS[t]['name']} in C{s + 1}",
                         "changes": [{"kind": "add", "slot": s + 1, "to": t}],
                         "ratio": float(ratio[i]), "safe_pass": bool(ratio[i] <= 1.0 - mg),
                         "direct_pass": bool(ratio[i] <= 1.0), "spectrum": z[i].tolist()})
        return rows

    # ---- map ------------------------------------------------------------------------------
    def map(self, codes, index: int) -> dict:
        c = self.codes(codes)
        k = int((c != 0).sum())
        if not self._k_ok(k):
            raise ValueError(f"maps are predicted for K = {self.k_range[0]}..{self.k_range[1]} (this placement: K = {k})")
        index = int(np.clip(index, 0, len(self.f) - 1))
        mhz = float(self.f[index])
        m = self.sur.maps(c[None], np.array([mhz]))[0].astype(np.float64)
        on = np.where(self.mask, m, -np.inf)
        r, q = np.unravel_index(int(np.argmax(on)), on.shape)
        vals = np.where(self.mask, m, np.nan)
        lo, hi = self.map_scale[index]
        return {"index": index, "mhz": mhz, "lo": lo, "hi": hi,
                "values": [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in vals],
                "hot": {"x": float(MAP_X0 + q * PIX_X), "y": float(MAP_Y0 + r * PIX_Y), "ohm": float(m[r, q])},
                "board_min": float(np.nanmin(vals)), "board_max": float(np.nanmax(vals))}

    def _map_scale(self) -> list[tuple[float, float]]:
        """Colour scale per frequency: MAP_PCT percentiles of the board values of the predicted maps of
        MAP_REF_PER_K random placements per budget in MAP_REF_K (typical design budgets; smaller budgets
        then show hotter than the scale, larger ones cooler). Fixed per frequency, so the colour of a
        pixel only changes when its predicted value changes. Cached; recomputed when the weights change."""
        ck = Path(settings.WEIGHTS_DIR) / settings.CHECKPOINT
        head = Path(settings.WEIGHTS_DIR) / "vae_route_finetune" / f"{settings.HEAD}.pt"
        key = {"checkpoint": str(ck), "size": ck.stat().st_size, "mtime": int(ck.stat().st_mtime),
               "head": str(head), "head_mtime": int(head.stat().st_mtime) if head.is_file() else None,
               "ref_k": list(MAP_REF_K), "per_k": MAP_REF_PER_K, "seed": MAP_REF_SEED, "pct": list(MAP_PCT),
               "n_freq": int(len(self.f))}
        path = CACHE_DIR / "map_scale.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("key") == key:
                return [tuple(x) for x in saved["scale"]]
        t0 = time.time()
        rng = np.random.default_rng(MAP_REF_SEED)
        ref = []
        for k in range(MAP_REF_K[0], MAP_REF_K[1] + 1):
            for _ in range(MAP_REF_PER_K):
                c = np.zeros(N_SLOTS, np.int64)
                c[rng.choice(N_SLOTS, size=k, replace=False)] = rng.integers(1, N_TYPES + 1, size=k)
                ref.append(c)
        ref = np.array(ref)
        self.log(f"computing the colour scale of the maps ({len(self.f)} frequencies x {len(ref)} reference "
                 f"placements, once) ...")
        scale = []
        for mhz in self.f:
            m = self.sur.maps(ref, np.full(len(ref), float(mhz)))
            v = m[:, self.mask]
            lo, hi = np.percentile(v[v > 0], MAP_PCT)
            scale.append((float(lo), float(hi)))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "scale": scale}))
        self.log(f"colour scale done in {time.time() - t0:.0f} s -> {path}")
        return scale

    # ---- nearest passing placement --------------------------------------------------------
    def fixes(self, codes, target: str, mode: str, max_radius: int = S.MAX_RADIUS,
              margin: float | None = None, remove: bool = True) -> dict:
        """Nearest passing placements: move, type change and (``remove``) removal, up to ``max_radius``
        changes. A placement that already passes gets the nearest placements with fewer decaps that
        still pass (``base_passes``); ``remove=False`` is the thesis assistant (K fixed)."""
        c = self.codes(codes)
        tgt, mg_safe = self._target(target, margin)
        if mode not in ("safe", "direct"):
            raise ValueError("mode must be 'safe' or 'direct'")
        mg = mg_safe if mode == "safe" else 0.0
        with self._gen_lock:
            self._fix_gen += 1
            gen = self._fix_gen
        k = int((c != 0).sum())
        base = {"mode": mode, "margin": mg, "max_radius": int(max_radius), "target": target, "K": k,
                "base_passes": False}
        if not self._k_ok(k):
            return {**base, "status": "no_model", "suggestions": []}
        judge = SplitJudge(self, mg, tgt, self.band, cancelled=lambda: self._fix_gen != gen)
        if bool(judge.assess(c[None])[0][0]):
            base["base_passes"] = True
            if not remove or k - 1 < self.k_range[0]:            # nothing smaller to look for
                return {**base, "max_radius": S.MAX_RADIUS, "status": "none", "suggestions": []}
        lines: list[str] = []
        t0 = time.time()
        try:
            found = find_fixes_fast(c, judge, N_TYPES, max_radius=int(max_radius),
                                    min_suggestions=S.MIN_SUGGESTIONS, seed=S.SEED, log=lines.append,
                                    remove=remove, k_max=k - 1 if base["base_passes"] else None)
        except Cancelled:
            return {**base, "status": "cancelled", "suggestions": []}
        top = found[:S.MAX_SUGGESTIONS]
        z = self.heads[0].predict_ohm(np.array([x["codes"] for x in top])) if top else np.zeros((0, len(self.f)))
        name = judge.members[0][0]
        rows = []
        for x, zz in zip(top, z):
            ch = change_list(c, x["codes"])
            rows.append({"codes": x["codes"].tolist(), "edits": int(x["edits"]), "moves": int(x["moves"]), "K": x["K"],
                         "radius": int(x["radius"]), "ratio": float(x["ratios"][name]), "score": float(x["score"]),
                         "changes": ch, "text": change_text(ch), "spectrum": zz.tolist()})
        return {**base, "status": "found" if rows else "none", "suggestions": rows, "n_found": len(found),
                "n_scored": judge.n_scored, "seconds": round(time.time() - t0, 2), "log": [s.strip() for s in lines]}

    # ---- lower spectrum within the budget --------------------------------------------------
    def improve(self, codes, target: str, max_radius: int = IMPROVE_RADIUS, n: int = S.MAX_SUGGESTIONS) -> dict:
        """Same-K placements up to ``max_radius`` changes away (type change or move, all of them) whose
        predicted worst-case ratio is lower than the placement's own; the ``n`` lowest first (author,
        2026-10-22: biggest drop first; ties: fewer changes, type changes before moves). A lower ratio
        than a passing placement's also passes. A newer improve search cancels an older one."""
        c = self.codes(codes)
        tgt, _ = self._target(target)
        with self._gen_lock:
            self._imp_gen += 1
            gen = self._imp_gen
        k = int((c != 0).sum())
        base = {"target": target, "K": k, "max_radius": int(max_radius)}
        if not self._k_ok(k):
            return {**base, "status": "no_model", "suggestions": []}
        t0 = time.time()
        rng = np.random.default_rng(S.SEED)           # not drawn from: levels 1 and 2 are enumerated
        cand = np.concatenate([neighbours_fast(c, r, N_TYPES, rng) for r in range(1, int(max_radius) + 1)])
        r0 = float(worst_ratios(self.heads[0].predict_ohm(c[None]), tgt, self.band)[0])
        try:                                          # candidates alone: the same batches as a one-GPU call
            ratio = self.split_ratios(cand, tgt, self.band, cancelled=lambda: self._imp_gen != gen)
        except Cancelled:
            return {**base, "status": "cancelled", "suggestions": []}
        better = np.flatnonzero(ratio < r0)
        edits = np.array([S.n_edits(c, cand[i]) for i in better], dtype=np.int64)
        moves = np.array([S.n_moves(c, cand[i]) for i in better], dtype=np.int64)
        order = better[np.lexsort((moves, edits, ratio[better]))][:n] if len(better) else better
        z = self.heads[0].predict_ohm(cand[order]) if len(order) else np.zeros((0, len(self.f)))
        rows = []
        for i, zz in zip(order, z):
            ch = change_list(c, cand[i])
            rows.append({"codes": cand[i].tolist(), "edits": S.n_edits(c, cand[i]), "moves": S.n_moves(c, cand[i]),
                         "K": k, "ratio": float(ratio[i]), "changes": ch, "text": change_text(ch),
                         "spectrum": zz.tolist()})
        return {**base, "status": "found" if rows else "none", "base_ratio": r0, "suggestions": rows,
                "n_found": int(len(better)), "n_scored": int(len(cand)), "seconds": round(time.time() - t0, 2)}

    # ---- page setup -----------------------------------------------------------------------
    def info(self) -> dict:
        return {"checkpoint": f"{Path(settings.WEIGHTS_DIR).parent.name}/{settings.CHECKPOINT}",
                "head": settings.HEAD, "devices": self.devices, "k_range": list(self.k_range),
                "max_radius": S.MAX_RADIUS, "max_suggestions": S.MAX_SUGGESTIONS, "margin_max": MARGIN_MAX}

    def targets_payload(self) -> list[dict]:
        return [{"name": t, "label": f"{t} Ω", "margin": self.margins[t], "curve": self.targets[t].tolist()}
                for t in TARGETS]
