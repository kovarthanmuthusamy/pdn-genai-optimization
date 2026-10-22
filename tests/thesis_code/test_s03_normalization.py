import csv

import numpy as np

from tests.thesis_code.helpers import DATA, REPO, assert_close, assert_same_json, load_json

N_DESIGNS = 50


def _first_designs(n):
    with (DATA / "raw/manifest.csv").open() as fh:
        out = []
        for r in csv.DictReader(fh):
            if r["design_id"] not in out:
                out.append(r["design_id"])
            if len(out) == n:
                return set(out)


def test_normalized_store_matches_dataset(out_dir):
    from thesis_code.s03_normalization import normalize
    ids = _first_designs(N_DESIGNS)
    out = normalize.normalize_store(DATA / "raw", out_dir / "norm", REPO / "thesis_code/common/inputs/normalization_stats.json", ids)
    with (out / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["design_id"] for r in rows} == ids
    for r in rows:
        assert_close(np.load(out / "heatmap" / r["sample_name"]), np.load(DATA / "normalized/heatmap" / r["sample_name"]), r["sample_name"])
    for d in ids:
        assert_close(np.load(out / "layouts" / d / "imp.npy"), np.load(DATA / "normalized/layouts" / d / "imp.npy"), f"{d} spectrum")


def test_stats_from_full_raw_match_shipped(out_dir):
    """Recomputing the statistics from all raw maps gives the shipped file (slow: reads 330,000 maps)."""
    import os, pytest
    if not os.getenv("THESIS_CODE_SLOW"):
        pytest.skip("set THESIS_CODE_SLOW=1")
    from thesis_code.s03_normalization import normalize
    out = normalize.normalize_store(DATA / "raw", out_dir / "norm_full", None)
    assert_same_json(load_json(out / "normalization_stats.json"),
                     load_json(REPO / "thesis_code/common/inputs/normalization_stats.json"), ignore=("created",))


def test_norm_stats_objects_match_original():
    from src_vae.others.norm_stats import load_norm_stats as orig
    from thesis_code.s03_normalization.stats import load_norm_stats
    a, b = load_norm_stats(REPO / "thesis_code/common/inputs/normalization_stats.json"), orig(DATA / "normalized")
    assert (a.impedance.log_mean, a.impedance.log_std) == (b.impedance.log_mean, b.impedance.log_std)
    for mhz in (1.5, 24.0, 200.0, 580.0):
        x = np.linspace(-3, 3, 64 * 64, dtype=np.float32).reshape(64, 64)
        assert_close(a.heatmap.norm_to_physical_np(x, mhz=mhz), b.heatmap.norm_to_physical_np(x, mhz=mhz), f"denorm {mhz}")
