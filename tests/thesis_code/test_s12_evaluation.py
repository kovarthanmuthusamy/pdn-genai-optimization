"""Chapter 4 checks of the copy against the results the thesis quotes.

Method: every check is re-run by the copy on the same inputs the original used (dataset, weights, the
saved eCADSTAR simulations) and each file it writes is compared, number by number, with the file of
the same name that the original wrote (floats to 1e-9 relative, arrays to 1e-6). This isolates one
question -- did trimming and moving change a result? -- and not whether the results are good. The
saved results were measured on the GPU, so these tests run on the GPU when one is free. No test may
start a process: eCADSTAR is a shared host, and new simulations are only written as PEB files.
"""
import importlib
import json
import subprocess
import sys

import numpy as np
import pytest

from tests.thesis_code.helpers import EXP, REPO, assert_close, assert_same_json, load_json

RESULTS = {                     # check -> the files it writes (saved under the same name in EXP/evals)
    "accuracy": ["ch4_accuracy.json", "ch4_accuracy.npz"],
    "beyond_peak": ["ch4_beyond_peak.json"],
    "board_laws": ["ch4_board_laws.json"],
    "cases": ["ch4_cases.json", "ch4_cases.npz"],
    "lookup_near": ["ch4_lookup_near.json"],
    "low_peaks": ["ch4_low_peaks.json"],
    "search_details": ["ch4_search_details.json"],
    "second_phase": ["ch4_second_phase.json"],
    "verdict": ["ch4_verdict.json"],
    # eCADSTAR checks: the analysis of the saved simulations
    "beyond_range_ecad": ["ch4_beyond_range.json", "ch4_beyond_range.npz"],
    "budget_proof_ecad": ["ch4_budget_proof.json", "ch4_budget_proof.npz"],
    "latent_ecad": ["ch4_latent.json", "ch4_latent.npz"],
    "nearest_pass_ecad": ["ch4_nearest_pass.json"],
    "next_decap_ecad": ["ch4_next_decap.json", "ch4_next_decap.npz"],
    "one_decap_ecad": ["ch4_one_decap.json", "ch4_one_decap.npz"],
    "search_ecad": ["ch4_search.json"],
    "search_maps_ecad": ["ch4_search_maps.json", "ch4_search_maps.npz"],
}
PREPARED = {"beyond_range_ecad": "beyond_range_ecad", "budget_proof_ecad": "budget_proof_ecad",
            "next_decap_ecad": "next_decap_ecad", "one_decap_ecad": "one_decap_ecad"}   # check -> its store
IGNORE = ("experiment",)        # the original records the experiment name; the copy names no experiment


@pytest.fixture
def no_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError(f"a process was started: {a[:1]}")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)


def _same_file(new, old):
    assert new.is_file(), f"{new.name} not written"
    if new.suffix == ".json":
        assert_same_json(load_json(new), load_json(old), new.name, ignore=IGNORE)
    else:
        a, b = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
        assert sorted(a.files) == sorted(b.files), f"{new.name}: arrays {sorted(set(a.files) ^ set(b.files))}"
        for k in a.files:
            assert_close(a[k], b[k], f"{new.name}:{k}")


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_check_reproduces_saved_result(name, out_dir, no_process):
    importlib.import_module(f"thesis_code.s12_evaluation.check_{name}").main()
    for fn in RESULTS[name]:
        _same_file(out_dir / "s12_evaluation" / fn, EXP / "evals" / fn)


def test_random_surrogate_reproduces_saved_proposals(out_dir, no_process):
    from thesis_code.s12_evaluation import check_random_surrogate as rs
    rs.main()
    runs = load_json(out_dir / "s12_evaluation/ch4_search_runs.json")
    saved = load_json(EXP / "evals/ch4_search_runs.json")
    assert set(runs) == set(saved)
    for key, info in runs.items():
        if not key.startswith("rand"):
            assert info == saved[key], key                    # the search runs are carried over unchanged
            continue
        new, old = REPO / info["run_dir"], REPO / saved[key]["run_dir"]
        assert new.is_relative_to(out_dir), key
        for k in range(2, 11):
            f = f"K{k:02d}/candidates.json"
            assert_same_json(load_json(new / f), load_json(old / f), f"{key}/{f}")


@pytest.mark.parametrize("name", sorted(PREPARED))
def test_new_simulations_are_written_as_peb_only(name, out_dir, no_process, monkeypatch, tmp_path_factory):
    """No saved store: the check draws its placements, writes the PEB files and commands.txt, and stops."""
    from thesis_code.common import settings
    monkeypatch.setattr(settings, "SIM_STORE_DIR", tmp_path_factory.mktemp("no_saved_store"))
    with pytest.raises(SystemExit, match="ECADSTAR_RUN"):
        importlib.import_module(f"thesis_code.s12_evaluation.check_{name}").main()
    new, old = out_dir / "s12_evaluation" / PREPARED[name], EXP / PREPARED[name]
    assert_same_json(load_json(new / "manifest.json"), load_json(old / "manifest.json"), "manifest", ignore=IGNORE)
    pebs = sorted(p.name for p in (old / "peb").glob("*.peb"))
    assert sorted(p.name for p in (new / "peb").glob("*.peb")) == pebs
    for p in pebs:
        assert (new / "peb" / p).read_bytes() == (old / "peb" / p).read_bytes(), p
    assert len((new / "peb/commands.txt").read_text().splitlines()) == 2 * len(pebs)


def test_nearest_pass_proposals_match_saved_manifest(out_dir, no_process, monkeypatch, tmp_path_factory):
    """No saved store: the assistant proposes the fixes again (same cases, same proposals), then the
    placements to simulate are written as PEB files and the check stops. The saved store started
    with simulations from earlier work, so its own run holds only the placements that were new to it:
    the copy must ask for every one of those, and for nothing the saved store does not hold."""
    from thesis_code.common import settings
    monkeypatch.setattr(settings, "SIM_STORE_DIR", tmp_path_factory.mktemp("no_saved_store"))
    with pytest.raises(SystemExit, match="ECADSTAR_RUN"):
        importlib.import_module("thesis_code.s12_evaluation.check_nearest_pass_ecad").main()
    new, old = out_dir / "s12_evaluation/nearest_pass_ecad", EXP / "nearest_pass_ecad"
    assert_same_json(load_json(new / "manifest.json"), load_json(old / "manifest.json"), "manifest",
                     ignore=IGNORE + ("seconds",))
    asked = {tuple(r["codes"]) for r in load_json(new / "spectra_run000/manifest.json")["rows"]}
    saved_run = {tuple(r["codes"]) for r in load_json(old / "spectra_run000/manifest.json")["rows"]}
    saved_all = {tuple(x) for x in np.load(old / "spectra.npz")["codes"]}
    assert saved_run <= asked <= saved_all, (len(saved_run - asked), len(asked - saved_all))
    n_batches = -(-len(asked) // 250)
    assert sorted(p.name for p in (new / "spectra_run000/peb").glob("*.peb")) == sorted(f"spectra{b}.peb" for b in range(n_batches))


def test_cached_store_dry_run_writes_original_peb(out_dir, no_process, tmp_path):
    """spectra_cached on a new store: manifest and PEB as the original would write them, then a stop."""
    sys.path.insert(0, str(REPO / "scrap"))
    import ch4_ecad as orig
    from pipelines.dataset_sim.peb_multitype import build_impedance_peb_multitype
    from thesis_code.s12_evaluation import common as c
    from thesis_code.s12_evaluation import ecad
    codes = c.codes_of(c.split_ids("test")[:5])
    codes[2, 0] = 1                                    # one placement that mounts the first slot
    with pytest.raises(SystemExit, match="ECADSTAR_RUN"):
        ecad.spectra_cached(ecad.store("dry"), codes)
    run = out_dir / "s12_evaluation/dry/spectra_run000"
    rows = orig.assign([{"codes": x.tolist()} for x in codes])
    assert load_json(run / "manifest.json")["rows"] == rows
    sel = sorted(rows, key=lambda r: r["pi"])
    build_impedance_peb_multitype(np.array([r["codes"] for r in sel]), tmp_path / "spectra0.peb", repo_root=REPO,
                                  powerbus=orig.POWERBUS, components=orig.IMPEDANCE_COMPONENTS)
    assert (run / "peb/spectra0.peb").read_bytes() == (tmp_path / "spectra0.peb").read_bytes()


def test_saved_store_is_read_only(out_dir):
    from thesis_code.s12_evaluation import ecad
    saved = ecad.store("budget_proof_ecad")
    assert saved == EXP / "budget_proof_ecad"
    with pytest.raises(SystemExit, match="read only"):
        ecad.writable(saved)


def test_run_search_records_runs(out_dir, monkeypatch):
    """Two tiny searches (K = 2, 4 vectors, 2 steps) through the runner: the plain one and the one without
    the straight-through gradient; both land in OUTPUT_DIR and in the runs file."""
    from thesis_code.s12_evaluation import run_search as rs
    monkeypatch.setattr(rs, "LABELS", ["s0", "noste"])
    monkeypatch.setattr(rs, "TARGETS", ("0.8",))
    monkeypatch.setattr(rs, "GPUS", [0])
    for k, v in {"SEARCH_NUM_STEPS": "2", "SEARCH_NUM_SEEDS": "4", "SEARCH_K_MAX": "2"}.items():
        monkeypatch.setenv(k, v)
    rs.main()
    runs = load_json(out_dir / "s12_evaluation/ch4_search_runs.json")
    saved = load_json(EXP / "evals/ch4_search_runs.json")
    assert set(runs) == set(saved)
    assert all(runs[k] == saved[k] for k in saved if k not in ("s0/0.8", "noste/0.8"))   # not run: saved run kept
    for key in ("s0/0.8", "noste/0.8"):
        info = runs[key]
        cfg = json.loads((REPO / info["run_dir"] / "run_config.json").read_text())
        assert (REPO / info["run_dir"]).is_relative_to(out_dir / "s10_search"), key
        assert cfg["num_steps"] == 2 and cfg["seed_info"]["master_seed"] == 20261008, key
    assert "read-out without the straight-through gradient" in (out_dir / "s12_evaluation/logs/ch4_search_noste_0.8.log").read_text()
