import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP_NAME = "exp096_k1k10_balanced30k_ep700"
EXP = REPO / "experiments" / EXP_NAME
DATA = REPO / "datasets/pdn_k1-10_balanced_v2"


def assert_close(a, b, what, rtol=1e-6, atol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if bad.any():
            i = tuple(np.argwhere(bad)[0])
            raise AssertionError(f"{what}: {int(bad.sum())} values differ, first at {i}: {a[i]!r} vs {b[i]!r}")
    else:
        assert np.array_equal(a, b), f"{what}: arrays differ"


def assert_same_json(a, b, path="", ignore=()):
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - set(ignore), set(b) - set(ignore)
        assert ka == kb, f"{path or '/'}: keys differ: {sorted(ka ^ kb)[:5]}"
        for k in sorted(ka):
            assert_same_json(a[k], b[k], f"{path}/{k}", ignore)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_json(x, y, f"{path}[{i}]", ignore)
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12) or (a != a and b != b), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def load_json(p):
    return json.loads(Path(p).read_text())


def assert_same_csv(p, q, ignore_cols=()):
    ra = list(csv.DictReader(Path(p).open()))
    rb = list(csv.DictReader(Path(q).open()))
    assert len(ra) == len(rb), f"{p}: {len(ra)} rows != {len(rb)}"
    for i, (x, y) in enumerate(zip(ra, rb)):
        for k in set(x) | set(y):
            if k not in ignore_cols:
                assert x.get(k) == y.get(k), f"{Path(p).name} row {i} column {k}: {x.get(k)!r} != {y.get(k)!r}"


def orig_module(rel: str):
    """A module of the original thesis-model code, e.g. orig_module("training.schedule")."""
    import importlib
    return importlib.import_module(f"experiments.{EXP_NAME}.codes.{rel}")
