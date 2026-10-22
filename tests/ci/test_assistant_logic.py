"""Design-assistant logic that needs no model: board geometry, candidate builder, fix search, wording.

The fix search is checked against the thesis assistant (thesis_code s11) with a stand-in judge whose
worst-case ratio is a fixed random function of the placement, so the ranking and the search order can be
compared exactly on a CPU.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from assistant_web.board import board_mask, geometry
from assistant_web.engine import change_list, change_text, find_fixes_fast, neighbours_fast
from thesis_code.s11_assistant import suggest as S

REPO = Path(__file__).resolve().parents[2]


class StandInJudge:
    """ratio = 1.1 - 0.05 * sum of a random weight per (slot, part type); pass when ratio <= 1 - margin."""

    k_range = (1, 10)

    def __init__(self, seed: int = 0, margin: float = 0.0):
        w = np.random.default_rng(seed).normal(size=(52, 4))
        w[:, 0] = 0.0
        self.w, self.members, self.n_scored = w, [("stand-in", None, margin)], 0

    def ratio(self, codes):
        codes = np.asarray(codes, dtype=np.int64).reshape(-1, 52)
        return 1.1 - 0.05 * self.w[np.arange(52)[None, :], codes].sum(1)

    def assess(self, codes):
        r = self.ratio(codes)
        self.n_scored += len(r)
        mg = self.members[0][2]
        return r <= 1.0 - mg, np.maximum(0.0, r / (1.0 - mg)), {"stand-in": r}


def _failing_placement(judge, k, seed):
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        c = np.zeros(52, np.int64)
        c[rng.choice(52, k, replace=False)] = rng.integers(1, 4, k)
        if 1.0 < judge.ratio(c)[0] < 1.12:
            return c
    raise AssertionError("no failing placement found")


def _random_codes(k, seed):
    rng = np.random.default_rng(seed)
    c = np.zeros(52, np.int64)
    c[rng.choice(52, k, replace=False)] = rng.integers(1, 4, k)
    return c


def test_board_geometry():
    g = geometry()
    mask = board_mask()
    w, h = g["board_mm"]
    assert len(g["slots"]) == 52 and len(g["outline"]) == 12
    assert all(0 <= x <= w and 0 <= y <= h for x, y in g["outline"])
    x0, x1, y0, y1 = g["map_extent"]
    for x, y in g["slots"]:                    # every slot lies on a board pixel
        q = int(round((x - x0) / (x1 - x0) * 64 - 0.5))
        r = int(round((y - y0) / (y1 - y0) * 64 - 0.5))
        assert mask[r, q], (x, y)


@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 6, 10])
def test_candidate_builder_equals_thesis(k, remove, monkeypatch):
    codes = _random_codes(k, 100 + k)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for r in (1, 2, 3) if k <= 3 else (1, 2):
        monkeypatch.setattr(S, "ALLOW_REMOVE", remove)
        A = S.neighbours_at(codes, r, 3, a)
        monkeypatch.setattr(S, "ALLOW_REMOVE", False)
        B = neighbours_fast(codes, r, 3, b, remove=remove)
        assert np.array_equal(A, B), (k, r)
    assert a.integers(1 << 62) == b.integers(1 << 62)


@pytest.mark.parametrize("k,seed", [(3, 1), (5, 2), (8, 3)])
def test_fix_search_equals_thesis(k, seed):
    judge = StandInJudge(seed)
    codes = _failing_placement(judge, k, seed)
    ref = S.find_fixes(codes, judge, 3, max_radius=2, min_suggestions=1, seed=S.SEED, log=None)
    got = find_fixes_fast(codes, judge, 3, max_radius=2, min_suggestions=1, seed=S.SEED, log=None)
    assert len(got) == len(ref) > 0
    for x, y in zip(got, ref):
        assert np.array_equal(x["codes"], y["codes"]) and x["score"] == y["score"]
        assert (x["edits"], x["moves"]) == (y["edits"], y["moves"])


def test_fix_search_with_removals_ranks_fewer_decaps_second(monkeypatch):
    judge = StandInJudge(7)
    codes = _failing_placement(judge, 6, 7)
    got = find_fixes_fast(codes, judge, 3, max_radius=2, min_suggestions=1, seed=S.SEED, log=None, remove=True)
    monkeypatch.setattr(S, "ALLOW_REMOVE", True)
    ref = S.find_fixes(codes, judge, 3, max_radius=2, min_suggestions=1, seed=S.SEED, log=None)
    ref.sort(key=lambda x: (x["edits"], int((x["codes"] != 0).sum()), x["moves"], x["score"]))
    assert [x["codes"].tolist() for x in got] == [x["codes"].tolist() for x in ref]
    keys = [(x["edits"], x["K"]) for x in got]
    assert keys == sorted(keys)


def test_passing_placement_only_gets_fewer_decaps():
    judge = StandInJudge(11)
    rng = np.random.default_rng(11)
    while True:
        codes = np.zeros(52, np.int64)
        codes[rng.choice(52, 5, replace=False)] = rng.integers(1, 4, 5)
        if judge.ratio(codes)[0] < 0.97:
            break
    got = find_fixes_fast(codes, judge, 3, max_radius=2, min_suggestions=1, seed=S.SEED, log=None,
                          remove=True, k_max=4)
    assert got and all(x["K"] <= 4 for x in got) and all(x["ratios"]["stand-in"] <= 1.0 for x in got)


def test_change_wording():
    base = np.zeros(52, np.int64)
    base[[18, 29, 37]] = [3, 3, 3]
    cand = base.copy()
    cand[18] = 1                               # a type change
    cand[29], cand[46] = 0, 3                  # two moves; removed and added slots are paired in slot
    cand[37], cand[30] = 0, 1                  # order, as suggest.edit_summary pairs them
    items = change_list(base, cand)
    assert [i["kind"] for i in items] == ["retype", "move", "move"]
    assert change_text(items) == "C19: T3 → T1; move C30 (T3) → C31 as T1; move C38 → C47"
    assert change_text(change_list(base, np.where(np.arange(52) == 18, 0, base))) == "remove C19 (T3)"
    assert change_text(change_list(base, np.where(np.arange(52) == 0, 2, base))) == "add T2 in C1"


def test_examples_file():
    examples = json.loads((REPO / "assistant_web/examples.json").read_text())
    assert len(examples) >= 4
    for e in examples:
        codes = np.array(e["codes"])
        assert codes.shape == (52,) and 1 <= (codes != 0).sum() <= 10 and set(np.unique(codes)) <= {0, 1, 2, 3}
        assert e["target"] in ("0.8", "0.6", "0.5") and e["name"]
