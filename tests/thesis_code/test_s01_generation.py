import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.thesis_code.helpers import REPO, assert_same_csv, assert_same_json, load_json

PLAN = REPO / "data/heatmaps"
RAW = "/mnt/c/Users/muthusamy/Desktop/PDN_raw_data"
CSVS = ["balance30k_planned_maps.csv", "balance30k_pool_ids.csv", "balance30k_setaside_ids.csv",
        "balance30k_drop_samples_pool.csv"] + [f"combinations_balance30k_{c}{s}.csv" for c in ("fill", "new")
                                               for s in ("", "_design_ids", "_freqs")]


def test_sampling_plan_reproduces_saved_plan(out_dir):
    from thesis_code.s01_generation import sample_plan
    sample_plan.main()
    for name in CSVS:
        assert_same_csv(out_dir / "s01_generation" / name, PLAN / name)
    assert_same_json(load_json(out_dir / "s01_generation/balance30k_report.json"),
                     load_json(PLAN / "balance30k_report.json"), ignore=("generated_at_utc", "unseen_boards_excluded"))


def test_peb_writer_matches_original(tmp_path):
    from pipelines.dataset_sim import peb_multitype as orig
    from thesis_code.s01_generation import peb
    from pipelines.dataset_sim.combinations_multitype import load_multitype_csv
    codes = load_multitype_csv(PLAN / "combinations_balance30k_new.csv")[:20]
    freqs = np.loadtxt(PLAN / "combinations_balance30k_new_freqs.csv", delimiter=",", skiprows=1, ndmin=2)[:20, -1]
    a = orig.build_impedance_peb_multitype(codes, tmp_path / "a.peb", repo_root=REPO, powerbus="Power_GND", components="IC1_Port1")
    b = peb.build_impedance_peb(codes, tmp_path / "b.peb", powerbus=peb.POWERBUS, components=peb.COMPONENTS)
    assert a.read_bytes() == b.read_bytes()
    a = orig.build_distribution_peb_multitype(codes, tmp_path / "c.peb", repo_root=REPO, per_sample_mhz=list(freqs), powerbus="Power_GND")
    b = peb.build_distribution_peb(codes, tmp_path / "d.peb", per_sample_mhz=list(freqs), powerbus=peb.POWERBUS)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("campaign", ["new", "fill"])
def test_simulate_dry_run_reproduces_campaign_pebs(out_dir, monkeypatch, campaign):
    from thesis_code.s01_generation import simulate
    monkeypatch.setattr(simulate, "CAMPAIGN", campaign)
    monkeypatch.setattr(simulate, "N_LAYOUTS", None)
    simulate.main()
    made = sorted((out_dir / f"s01_generation/peb_{campaign}").glob("*.peb"))
    saved = Path(RAW) / f"Raw_balance30k_{campaign}/peb"
    assert [p.name for p in made] == sorted(p.name for p in saved.glob("*.peb"))
    for p in made:
        assert p.read_bytes() == (saved / p.name).read_bytes(), p.name


def test_simulate_dry_run_starts_no_process(out_dir, monkeypatch):
    from thesis_code.s01_generation import simulate

    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(simulate, "CAMPAIGN", "new")
    monkeypatch.setattr(simulate, "N_LAYOUTS", 2)
    simulate.main()
    assert (out_dir / "s01_generation/peb_new/commands.txt").is_file()
