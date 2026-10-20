"""The web assistant gives the thesis assistant's answers (assistant_web/ against thesis_code/).

Needs the exp096 weights and, for the split tests, two GPUs (one is enough; the split then has one part).
Run: pytest tests/assistant_web -q     (about 2 minutes: the thesis fix search runs on one GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from assistant_web.board import board_mask, geometry  # noqa: E402
from assistant_web.engine import Engine, neighbours_fast, worst_ratios  # noqa: E402
from thesis_code.common.board import INPUTS  # noqa: E402
from thesis_code.s11_assistant import suggest as S  # noqa: E402

EXAMPLES = json.loads((REPO / "assistant_web/examples.json").read_text())
TARGETS = ("0.8", "0.6", "0.5")


def _random_codes(n, seed, k_lo=1, k_hi=10):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 52), np.int64)
    for i in range(n):
        k = int(rng.integers(k_lo, k_hi + 1))
        out[i, rng.choice(52, k, replace=False)] = rng.integers(1, 4, k)
    return out


@pytest.fixture(scope="module")
def eng():
    e = Engine(gpus=(0, 1), log=lambda *_: None)
    yield e
    e.close()


@pytest.fixture(scope="module")
def thesis_judges(eng):
    """The one-GPU thesis judges (suggest.load_judge), safe and direct, per target."""
    from thesis_code.common import settings
    from thesis_code.s09_inference.placement_encoder import load, margin
    head = load()                              # what load_judge loads, once for all six judges
    out = {}
    for t in TARGETS:
        p = INPUTS / f"target_impedance_{t}.npy"
        tgt = np.load(p).reshape(-1)
        out[t] = {mode: S.Judge([(f"vae:{settings.HEAD}", head, mg)], tgt, eng.band)
                  for mode, mg in (("safe", float(margin(p))), ("direct", 0.0))}
    return out


# ---- no model needed ---------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 10])
def test_neighbours_fast_identical(k):
    codes = _random_codes(1, seed=100 + k, k_lo=k, k_hi=k)[0]
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for r in (1, 2, 3):                        # as find_fixes calls it: one generator, radius 1, 2, 3
        A, B = S.neighbours_at(codes, r, 3, a), neighbours_fast(codes, r, 3, b)
        assert A.shape == B.shape and np.array_equal(A, B), f"K={k} radius {r}"
    assert a.integers(1 << 62) == b.integers(1 << 62), "generator ends in another state"


def test_board_geometry():
    g = geometry()
    mask = board_mask()
    w, h = g["board_mm"]
    assert len(g["outline"]) == 12 and all(0 <= x <= w and 0 <= y <= h for x, y in g["outline"])
    x0, x1, y0, y1 = g["map_extent"]
    for x, y in g["slots"]:                    # every slot lies on a board pixel of the mask
        q = int(round((x - x0) / (x1 - x0) * 64 - 0.5))
        r = int(round((y - y0) / (y1 - y0) * 64 - 0.5))
        assert mask[r, q], (x, y)


# ---- with the model ----------------------------------------------------------------------
def test_split_equals_one_gpu(eng):
    codes = _random_codes(70_000, seed=1)
    tgt = eng.targets["0.6"]
    split = eng.split_ratios(codes, tgt, eng.band)
    one = worst_ratios(eng.heads[0].predict_ohm(codes), tgt, eng.band)
    assert np.array_equal(split, one)


@pytest.mark.parametrize("ex", [e["id"] for e in EXAMPLES])
def test_check_matches_thesis_verdict(eng, thesis_judges, ex):
    codes = np.array(next(e for e in EXAMPLES if e["id"] == ex)["codes"])
    for t in TARGETS:
        r = eng.check(codes, t)
        for mode in ("safe", "direct"):
            j = thesis_judges[t][mode]
            z = j.spectra(codes)[j.members[0][0]][0]
            v = S.verdict(z, j.tgt, j.band, eng.f, j.members[0][2])
            assert r[mode] == v, (ex, t, mode)
            assert r[mode]["passes"] == bool(j.assess(codes)[0][0])


@pytest.mark.parametrize("ex,target,mode", [("thesis", "0.8", "safe"), ("thesis", "0.8", "direct"),
                                            ("sweep", "0.6", "safe"), ("sweep", "0.6", "direct"),
                                            ("thesis", "0.5", "direct")])
def test_fixes_match_thesis_find_fixes(eng, thesis_judges, ex, target, mode):
    codes = np.array(next(e for e in EXAMPLES if e["id"] == ex)["codes"])
    r = eng.fixes(codes, target, mode, S.MAX_RADIUS)
    ref = S.find_fixes(codes, thesis_judges[target][mode], 3, max_radius=S.MAX_RADIUS,
                       min_suggestions=S.MIN_SUGGESTIONS, seed=S.SEED, log=None)
    assert r["status"] == ("found" if ref else "none")
    assert r.get("n_found", 0) == len(ref)
    top = ref[:S.MAX_SUGGESTIONS]
    assert [s["codes"] for s in r["suggestions"]] == [x["codes"].tolist() for x in top]
    assert [s["score"] for s in r["suggestions"]] == [x["score"] for x in top]
    for s, x in zip(r["suggestions"], top):
        assert S.edit_summary(codes, np.array(s["codes"])) == S.edit_summary(codes, x["codes"])


def test_additions_rank_every_addition(eng):
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "rule")["codes"])
    rows = eng.additions(codes, "0.5", n=10_000)
    assert len(rows) == (52 - 3) * 3
    ratios = [x["ratio"] for x in rows]
    assert ratios == sorted(ratios)
    tgt = eng.targets["0.5"]
    for x in rows[:5]:
        z = eng.heads[0].predict_ohm(np.array(x["codes"]))
        assert x["ratio"] == pytest.approx(float((z / tgt)[eng.band].max()), rel=1e-6)


def test_map_matches_surrogate(eng):
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "sweep")["codes"])
    m = eng.map(codes, 150)
    ref = eng.sur.maps(codes[None], np.array([eng.f[150]]))[0]
    mask = board_mask()
    vals = np.array([[np.nan if v is None else v for v in row] for row in m["values"]])
    assert np.array_equal(np.isnan(vals), ~mask)
    assert np.allclose(vals[mask], ref[mask], rtol=0, atol=1e-6)
    assert m["hot"]["ohm"] == pytest.approx(float(ref[mask].max()))
    assert m["lo"] < m["hi"]


def test_budget_limits(eng):
    empty = np.zeros(52, np.int64)
    r = eng.check(empty, "0.8")
    assert r["kind"] == "bare" and r["K"] == 0 and len(r["additions"]) == 5
    assert eng.fixes(empty, "0.8", "safe")["status"] == "no_model"
    with pytest.raises(ValueError):
        eng.map(empty, 0)
    full = _random_codes(1, seed=3, k_lo=10, k_hi=10)[0]
    assert eng.check(full, "0.8")["additions"] == []
    over = full.copy()
    over[np.flatnonzero(over == 0)[0]] = 1
    with pytest.raises(ValueError):
        eng.check(over, "0.8")
    with pytest.raises(ValueError):
        eng.check(empty, "0.7")


def test_server_routes(eng):
    from fastapi import HTTPException
    from assistant_web import server
    server.STATE["engine"] = eng
    codes = next(e for e in EXAMPLES if e["id"] == "rule")["codes"]
    r = server.check(server.CheckReq(codes=codes, target="0.8"))
    assert r["safe"] == eng.check(codes, "0.8")["safe"]
    with pytest.raises(HTTPException) as e:
        server.check(server.CheckReq(codes=codes[:10], target="0.8"))
    assert e.value.status_code == 400
    assert server.fixes(server.FixReq(codes=codes, target="0.8", mode="safe", max_radius=9))["max_radius"] == 3
