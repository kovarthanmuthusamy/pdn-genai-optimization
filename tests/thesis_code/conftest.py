"""Equivalence tests of thesis_code/ against the original code. Repo-only, not shipped."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from repo_paths import setup_path  # noqa: E402

setup_path()
EXP = REPO / "experiments/exp096_k1k10_balanced30k_ep700"
DATA = REPO / "datasets/pdn_k1-10_balanced_v2"
GUARDED_DEEP = [EXP / "checkpoints", EXP / "evals", REPO / "data/heatmaps", REPO / "configs",
                REPO / "Thesis_report/chapters", REPO / "Thesis_report/images"]
GUARDED_TOP = [DATA, DATA / "raw", DATA / "normalized", DATA / "test_pool", EXP,
               REPO / "datasets/_building_blocks/data_multifreq_balance30k_new"]
# Files other work in this repo writes while the tests may run (not written by thesis_code/).
GUARD_IGNORE = ("assistant_sweep",)


def _snapshot(roots, recursive):
    snap = {}
    for r in roots:
        if not r.is_dir():
            continue
        for p in (r.rglob("*") if recursive else r.iterdir()):
            if (p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
                    and not any(x in p.name for x in GUARD_IGNORE)):
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


@pytest.fixture(scope="session", autouse=True)
def guard_repo_results():
    before = {**_snapshot(GUARDED_DEEP, True), **_snapshot(GUARDED_TOP, False)}
    yield
    after = {**_snapshot(GUARDED_DEEP, True), **_snapshot(GUARDED_TOP, False)}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"files the thesis reads were changed or created: {changed[:10]}"


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    from thesis_code.common import settings
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path)
    return tmp_path
