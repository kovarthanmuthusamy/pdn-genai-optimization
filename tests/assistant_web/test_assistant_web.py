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
def thesis_head():
    """The placement-only encoder as suggest.load_judge loads it, once for every judge."""
    from thesis_code.s09_inference.placement_encoder import load
    return load()


def _thesis_judge(head, target, mg, band):
    from thesis_code.common import settings
    tgt = np.load(INPUTS / f"target_impedance_{target}.npy").reshape(-1)
    return S.Judge([(f"vae:{settings.HEAD}", head, float(mg))], tgt, band)


@pytest.fixture(scope="module")
def thesis_judges(eng, thesis_head):
    """The one-GPU thesis judges (suggest.load_judge), safe and direct, per target."""
    from thesis_code.s09_inference.placement_encoder import margin
    return {t: {mode: _thesis_judge(thesis_head, t, mg, eng.band)
                for mode, mg in (("safe", margin(INPUTS / f"target_impedance_{t}.npy")), ("direct", 0.0))}
            for t in TARGETS}


# ---- no model needed ---------------------------------------------------------------------
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 10])
def test_neighbours_fast_identical(k, remove, monkeypatch):
    """remove=True against the thesis builder with its ALLOW_REMOVE switched on."""
    codes = _random_codes(1, seed=100 + k, k_lo=k, k_hi=k)[0]
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for r in (1, 2, 3):                        # as find_fixes calls it: one generator, radius 1, 2, 3
        monkeypatch.setattr(S, "ALLOW_REMOVE", remove)
        A = S.neighbours_at(codes, r, 3, a)
        monkeypatch.setattr(S, "ALLOW_REMOVE", False)
        B = neighbours_fast(codes, r, 3, b, remove=remove)
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


def _budget_order(found):
    """The page's order: fewer changes, then fewer decaps, then the thesis order (stable re-sort)."""
    return sorted(found, key=lambda x: (x["edits"], int((x["codes"] != 0).sum()), x["moves"], x["score"]))


def _same_list(r, ref, codes):
    assert r["status"] == ("found" if ref else "none")
    assert r.get("n_found", 0) == len(ref)
    top = ref[:S.MAX_SUGGESTIONS]
    assert [s["codes"] for s in r["suggestions"]] == [x["codes"].tolist() for x in top]
    assert [s["score"] for s in r["suggestions"]] == [x["score"] for x in top]
    for s, x in zip(r["suggestions"], top):
        assert S.edit_summary(codes, np.array(s["codes"])) == S.edit_summary(codes, x["codes"])
        assert s["K"] == int((x["codes"] != 0).sum())


@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("ex,target,mode", [("thesis", "0.8", "safe"), ("thesis", "0.8", "direct"),
                                            ("sweep", "0.6", "safe"), ("sweep", "0.6", "direct"),
                                            ("thesis", "0.5", "direct")])
def test_fixes_match_thesis_find_fixes(eng, thesis_judges, ex, target, mode, remove, monkeypatch):
    """remove=False is the thesis assistant; remove=True (the page) is the thesis search with its
    ALLOW_REMOVE on, ranked fewer changes first, then fewer decaps."""
    codes = np.array(next(e for e in EXAMPLES if e["id"] == ex)["codes"])
    r = eng.fixes(codes, target, mode, S.MAX_RADIUS, remove=remove)
    assert not r["base_passes"]
    monkeypatch.setattr(S, "ALLOW_REMOVE", remove)
    ref = S.find_fixes(codes, thesis_judges[target][mode], 3, max_radius=S.MAX_RADIUS,
                       min_suggestions=S.MIN_SUGGESTIONS, seed=S.SEED, log=None)
    _same_list(r, _budget_order(ref), codes)


def test_passing_placement_gets_fewer_decaps(eng, thesis_head, monkeypatch):
    """A passing placement: the nearest placements with fewer decaps that still pass = the thesis search
    with ALLOW_REMOVE on and the judge's K range capped at K - 1."""
    from thesis_code.s09_inference.placement_encoder import margin
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "search")["codes"])
    k = int((codes != 0).sum())
    r = eng.fixes(codes, "0.8", "safe", S.MAX_RADIUS)        # K = 4 passes 0.8 ohm; K = 3 can too
    assert r["base_passes"] and r["K"] == k
    j = _thesis_judge(thesis_head, "0.8", margin(INPUTS / "target_impedance_0.8.npy"), eng.band)
    j.k_range = (j.k_range[0], k - 1)
    monkeypatch.setattr(S, "ALLOW_REMOVE", True)
    ref = S.find_fixes(codes, j, 3, max_radius=S.MAX_RADIUS, min_suggestions=S.MIN_SUGGESTIONS,
                       seed=S.SEED, log=None)
    _same_list(r, _budget_order(ref), codes)
    assert ref and all(s["K"] < k for s in r["suggestions"])
    assert eng.fixes(codes, "0.8", "safe", remove=False)["status"] == "none"   # thesis: already passes


@pytest.mark.parametrize("mg", [0.0, 0.1, 0.2, 0.5, 0.6])
def test_check_custom_margin(eng, thesis_head, mg):
    """The margin slider: the safe verdict is suggest.verdict with that margin; None is the thesis margin."""
    for ex in ("sweep", "search"):
        codes = np.array(next(e for e in EXAMPLES if e["id"] == ex)["codes"])
        for t in TARGETS:
            r = eng.check(codes, t, margin=mg)
            j = _thesis_judge(thesis_head, t, mg, eng.band)
            z = j.spectra(codes)[j.members[0][0]][0]
            assert r["safe"] == S.verdict(z, j.tgt, j.band, eng.f, mg), (ex, t, mg)
            assert r["margin"] == mg and r["thesis_margin"] == eng.margins[t]
            assert all(a["safe_pass"] == (a["ratio"] <= 1.0 - mg) for a in r["additions"])
            if mg == 0.0:
                assert r["safe"] == r["direct"]
    codes = next(e for e in EXAMPLES if e["id"] == "sweep")["codes"]
    assert eng.check(codes, "0.6") == eng.check(codes, "0.6", margin=None)
    assert eng.check(codes, "0.6")["margin"] == eng.margins["0.6"]


def test_fixes_custom_margin(eng, thesis_head, monkeypatch):
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "sweep")["codes"])
    monkeypatch.setattr(S, "ALLOW_REMOVE", True)
    for mg in (0.2, 0.45):
        r = eng.fixes(codes, "0.6", "safe", S.MAX_RADIUS, margin=mg)
        ref = S.find_fixes(codes, _thesis_judge(thesis_head, "0.6", mg, eng.band), 3, max_radius=S.MAX_RADIUS,
                           min_suggestions=S.MIN_SUGGESTIONS, seed=S.SEED, log=None)
        assert r["margin"] == mg
        _same_list(r, _budget_order(ref), codes)
    assert eng.fixes(codes, "0.6", "direct", 2, margin=0.3)["margin"] == 0.0   # direct ignores the slider


def test_margin_out_of_range(eng):
    codes = next(e for e in EXAMPLES if e["id"] == "rule")["codes"]
    for bad in (-0.01, 0.61):
        with pytest.raises(ValueError):
            eng.check(codes, "0.8", margin=bad)
        with pytest.raises(ValueError):
            eng.fixes(codes, "0.8", "safe", margin=bad)


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
    assert server.check(server.CheckReq(codes=codes, target="0.8", margin=0.1))["margin"] == 0.1
    with pytest.raises(HTTPException) as e:
        server.check(server.CheckReq(codes=codes, target="0.8", margin=0.9))
    assert e.value.status_code == 400


def test_predict_batch(eng):
    """/api/predict: the placement-only encoder's spectra, ratios and verdicts for a batch."""
    codes = _random_codes(300, seed=7)
    r = eng.predict(codes.tolist(), "0.6")
    z = eng.heads[0].predict_ohm(codes)
    assert r["n"] == 300 and len(r["freqs_mhz"]) == 231
    assert np.allclose(np.array(r["spectra"]), z, rtol=1e-5, atol=1e-6)
    ratio = worst_ratios(z, eng.targets["0.6"], eng.band)
    assert np.allclose(r["worst_ratio"], ratio, rtol=1e-6)
    assert r["safe_pass"] == (ratio <= 1 - eng.margins["0.6"]).tolist() and r["direct_pass"] == (ratio <= 1).tolist()
    assert "worst_ratio" not in eng.predict(codes[:2].tolist())
    for bad in ([[0] * 52], [[1] * 52], [[0] * 51], [[5] + [0] * 51]):
        with pytest.raises(ValueError):
            eng.predict(bad)


def test_improve_lists_lowest_ratio(eng):
    """improve = every same-K placement 1 or 2 changes away (thesis neighbours_at), judged on one GPU, those
    below the placement's own ratio, lowest first (ties: fewer changes, type changes before moves)."""
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "search")["codes"])
    r = eng.improve(codes, "0.8")
    rng = np.random.default_rng(0)
    cand = np.concatenate([S.neighbours_at(codes, k, 3, rng) for k in (1, 2)])
    tgt = eng.targets["0.8"]
    r0 = worst_ratios(eng.heads[0].predict_ohm(codes[None]), tgt, eng.band)[0]
    ratio = worst_ratios(eng.heads[0].predict_ohm(cand), tgt, eng.band)
    better = [i for i in range(len(cand)) if ratio[i] < r0]
    better.sort(key=lambda i: (ratio[i], S.n_edits(codes, cand[i]), S.n_moves(codes, cand[i])))
    assert r["status"] == "found" and r["n_found"] == len(better) and r["n_scored"] == len(cand)
    assert r["base_ratio"] == pytest.approx(float(r0), rel=1e-6)
    assert [x["codes"] for x in r["suggestions"]] == [cand[i].tolist() for i in better[:S.MAX_SUGGESTIONS]]
    assert all(x["K"] == int((codes != 0).sum()) and 1 <= x["edits"] <= 2 for x in r["suggestions"])
    assert eng.improve(np.zeros(52, np.int64), "0.8")["status"] == "no_model"


# ---- eCADSTAR check (no eCADSTAR needed: the simulation calls are replaced) ---------------
@pytest.fixture
def ecad_runner(eng, tmp_path, monkeypatch):
    from assistant_web import ecad_check
    monkeypatch.setattr(ecad_check, "STORE", tmp_path / "ecad")
    return ecad_check.EcadRunner(eng, log=lambda *_: None)


def _fake_store(store, codes, spectrum=None, maps=None):
    store.mkdir(parents=True, exist_ok=True)
    if spectrum is not None:
        np.savez_compressed(store / "spectra.npz", codes=codes[None], values=spectrum[None])
    if maps is not None:
        mhz, m = maps
        np.savez_compressed(store / "maps.npz", codes=codes[None], values=m[None].astype(np.float32), mhz=np.array([mhz]))


def test_ecad_compare_reads_store(eng, ecad_runner):
    from assistant_web import ecad_check
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "sweep")["codes"])
    empty = ecad_runner.compare(codes, "0.6", 150)
    assert empty["spectrum"] is None and empty["worst"] is None and empty["map"] is None
    z = eng.heads[0].predict_ohm(codes[None])[0].astype(np.float64)
    ref_map = eng.sur.maps(codes[None], np.array([eng.f[150]]))[0] * board_mask()
    _fake_store(ecad_check.STORE, codes, 1.1 * z, (float(eng.f[150]), ref_map))
    r = ecad_runner.compare(codes, "0.6", 150)
    check = eng.check(codes, "0.6")["direct"]
    assert r["worst"]["ratio"] == pytest.approx(1.1 * check["worst_ratio"], rel=1e-6)
    assert r["worst"]["mhz"] == pytest.approx(check["worst_mhz"])
    assert r["worst"]["passes"] == (1.1 * check["worst_ratio"] <= 1.0)
    m = eng.map(codes, 150)
    assert r["map"]["hot"]["ohm"] == pytest.approx(m["hot"]["ohm"], rel=1e-5)
    assert (r["map"]["hot"]["x"], r["map"]["hot"]["y"]) == (m["hot"]["x"], m["hot"]["y"])
    assert ecad_runner.compare(codes, "0.6", 151)["map"] is None          # another frequency: not simulated
    assert ecad_runner.compare(codes, "0.6", None)["map"] is None


def test_ecad_start_refuses_when_busy(ecad_runner, monkeypatch):
    from assistant_web import ecad_check
    monkeypatch.setattr(ecad_check.ecad, "busy", lambda: "engineer.exe running")
    called = []
    monkeypatch.setattr(ecad_check.ecad, "spectra_cached", lambda *a: called.append(a))
    codes = next(e for e in EXAMPLES if e["id"] == "rule")["codes"]
    with pytest.raises(ecad_check.EcadBusy, match="engineer.exe running"):
        ecad_runner.start(codes, None)
    assert ecad_runner.status()["state"] == "idle" and not called
    with pytest.raises(ValueError):
        ecad_runner.start([0] * 52, None)                                   # K = 0


def test_ecad_job_runs_and_saves(eng, ecad_runner, monkeypatch):
    """The job calls thesis_code's spectra_cached / maps_cached with ECADSTAR_RUN on and OUTPUT_DIR set to the
    page's cache, restores both, and a saved placement is not simulated again."""
    import time as _time
    from assistant_web import ecad_check
    from thesis_code.common import settings
    old = settings.OUTPUT_DIR, settings.ECADSTAR_RUN
    monkeypatch.setattr(ecad_check.ecad, "busy", lambda: None)
    seen = []

    def fake_spectra(store, codes):
        seen.append(("spectra", settings.ECADSTAR_RUN, Path(settings.OUTPUT_DIR) == ecad_check.STORE.parent))
        _fake_store(store, codes[0], spectrum=eng.heads[0].predict_ohm(codes)[0].astype(np.float64))

    def fake_maps(store, codes, mhz):
        seen.append(("maps", settings.ECADSTAR_RUN, float(mhz[0])))
        m = eng.sur.maps(codes, mhz)[0] * board_mask()
        np.savez_compressed(store / "maps.npz", codes=codes, values=m[None].astype(np.float32), mhz=mhz)

    monkeypatch.setattr(ecad_check.ecad, "spectra_cached", fake_spectra)
    monkeypatch.setattr(ecad_check.ecad, "maps_cached", fake_maps)
    codes = np.array(next(e for e in EXAMPLES if e["id"] == "rule")["codes"])
    st = ecad_runner.start(codes, 120)
    assert st["state"] == "running" and st["phase"] == "spectrum"
    for _ in range(100):
        if ecad_runner.status()["state"] != "running":
            break
        _time.sleep(0.1)
    st = ecad_runner.status()
    assert st["state"] == "done", st
    assert seen == [("spectra", True, True), ("maps", True, float(eng.f[120]))]
    assert (settings.OUTPUT_DIR, settings.ECADSTAR_RUN) == old
    r = ecad_runner.compare(codes, "0.8", 120)
    assert r["worst"]["ratio"] == pytest.approx(eng.check(codes, "0.8")["direct"]["worst_ratio"], rel=1e-6)
    assert r["map"]["index"] == 120
    assert ecad_runner.start(codes, 120)["saved"] is True and len(seen) == 2   # nothing simulated again


def test_ecad_routes(eng, ecad_runner, monkeypatch):
    from fastapi import HTTPException
    from assistant_web import ecad_check, server
    server.STATE["engine"], server.STATE["ecad"] = eng, ecad_runner
    monkeypatch.setattr(ecad_check.ecad, "busy", lambda: "pipeline running: queue_x.sh")
    codes = next(e for e in EXAMPLES if e["id"] == "rule")["codes"]
    with pytest.raises(HTTPException) as e:
        server.ecad_run(server.EcadReq(codes=codes, index=None))
    assert e.value.status_code == 409 and "busy" in e.value.detail
    assert server.ecad_status()["state"] == "idle"
    assert server.ecad_result(server.EcadReq(codes=codes, target="0.8", index=3))["spectrum"] is None
