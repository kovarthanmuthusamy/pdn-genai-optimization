import csv

import numpy as np

from tests.thesis_code.helpers import REPO, assert_close, assert_same_json, load_json

RAW = __import__("pathlib").Path("/mnt/c/Users/muthusamy/Desktop/PDN_raw_data/Raw_balance30k_new")
STORE = REPO / "datasets/_building_blocks/data_multifreq_balance30k_new"
N = 50


def _rows(root):
    with (root / "manifest.csv").open() as fh:
        return {r["sample_name"]: r for r in csv.DictReader(fh)}


def test_processing_matches_campaign_store(out_dir):
    from thesis_code.s02_processing import process
    out = process.process(RAW, REPO / "data/heatmaps/combinations_balance30k_new.csv", out_dir / "p", n_layouts=N)
    new, old = _rows(out), _rows(STORE)
    assert new, "nothing processed"
    designs = {r["design_id"] for r in new.values()}
    for name, r in new.items():
        o = old[name]
        assert {k: r[k] for k in ("design_id", "freq_mhz", "K")} == {k: o[k] for k in ("design_id", "freq_mhz", "K")}, name
        assert_close(np.load(out / "heatmap" / name), np.load(STORE / "heatmap" / name), f"map {name}", rtol=0, atol=0)
    for d in designs:
        for f in ("imp.npy", "occ.npy"):
            assert_close(np.load(out / "layouts" / d / f), np.load(STORE / "layouts" / d / f), f"{d}/{f}", rtol=0, atol=0)


def test_reader_matches_original():
    from libs.data_creation.impedance import read_impedance_file
    from thesis_code.s02_processing import read_outputs
    f = sorted((RAW / "impedance").rglob("*.csv"))[0]
    assert_close(read_outputs.read_impedance_file(f), read_impedance_file(f), "spectrum", rtol=0, atol=0)


def test_merge_selection_matches_original():
    from pipelines.data import merge_raw_datasets as orig
    from thesis_code.s02_processing import merge
    pool = REPO / "datasets/pdn_k1-10_balanced_v1/test_pool/raw"
    rows = orig._read_manifest(pool)
    flt = {"keep_ids_csv": REPO / "data/heatmaps/balance30k_pool_ids.csv"}
    assert merge.select_designs(pool, rows, flt) == orig.select_designs(pool, rows, flt)
    assert set(merge.PRESETS) == {"balance30k", "balance30k_testpool"}


def _subset_store(src, dst, designs):
    import shutil
    rows = [r for r in csv.DictReader((src / "manifest.csv").open()) if r["design_id"] in designs]
    for sub in ("heatmap", "PI_freq", "layouts"):
        (dst / sub).mkdir(parents=True)
    for r in rows:
        for sub in ("heatmap", "PI_freq"):
            shutil.copy(src / sub / r["sample_name"], dst / sub / r["sample_name"])
    for d in designs:
        shutil.copytree(src / "layouts" / d, dst / "layouts" / d)
    with (dst / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_merge_matches_original_on_small_stores(tmp_path, monkeypatch):
    from pipelines.data import merge_raw_datasets as orig
    from thesis_code.s02_processing import merge
    fill = REPO / "datasets/_building_blocks/data_multifreq_balance30k_fill"
    designs = sorted({r["design_id"] for r in csv.DictReader((fill / "manifest.csv").open())})[:3]
    pool_rows = _subset_store(REPO / "datasets/pdn_k1-10_balanced_v1/test_pool/raw", tmp_path / "pool", designs)
    _subset_store(fill, tmp_path / "fill", designs)
    (tmp_path / "keep.csv").write_text("\n".join(designs) + "\n")
    (tmp_path / "drop.csv").write_text(pool_rows[0]["sample_name"] + "\n")
    sources = [("v1_pool", tmp_path / "pool"), ("balance30k_fill", tmp_path / "fill")]
    filters = {"v1_pool": {"keep_ids_csv": tmp_path / "keep.csv", "drop_samples_csv": tmp_path / "drop.csv"}}
    modes = {"balance30k_fill": "append"}
    for name, value in (("SOURCES", sources), ("SOURCE_FILTERS", filters), ("SOURCE_MODES", modes),
                        ("OUTPUT_DIR", tmp_path / "a"), ("PRESET", "balance30k")):
        monkeypatch.setattr(orig, name, value)
    orig.main()
    merge.merge(sources, filters, modes, tmp_path / "b", "balance30k")
    assert (tmp_path / "a/manifest.csv").read_text() == (tmp_path / "b/manifest.csv").read_text()
    for sub in ("heatmap", "PI_freq"):
        for f in sorted((tmp_path / "a" / sub).iterdir()):
            assert f.read_bytes() == (tmp_path / "b" / sub / f.name).read_bytes(), f"{sub}/{f.name}"
    ma, mb = (load_json(tmp_path / x / "dataset_meta.json") for x in ("a", "b"))
    assert_same_json(ma, mb, ignore=("generated_at_utc", "source_script"))
