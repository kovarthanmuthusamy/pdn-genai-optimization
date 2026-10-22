import numpy as np
import pytest
import torch

from tests.thesis_code.helpers import DATA, EXP_NAME, REPO, assert_close, assert_same_csv

LOADER_KW = dict(batch_size=112, num_workers=0, train_split=0.9, seed=42, pin_memory=False,
                 train_samples_per_epoch=64000, val_batch_size=320, n_decap_types=3)


def test_split_matches_original_and_package_lists(out_dir):
    from libs.experiment_code import code_module
    from thesis_code.s04_dataset import dataset, loader, package
    orig = code_module(EXP_NAME, "loader")
    ds = dataset.VAEDataset(str(DATA / "normalized"), precompute_k=True)
    meta = loader.precompute_multifreq_metadata(ds)
    assert loader._split_indices_by_design(meta["design_keys"], 0.9, 42) == orig._split_indices_by_design(meta["design_keys"], 0.9, 42)
    package.write_split_lists(DATA / "raw", out_dir / "pkg")
    for n in ("split_train.txt", "split_heldout.txt"):
        assert (out_dir / "pkg" / n).read_text() == (DATA / n).read_text(), n


def test_first_batches_match_original(monkeypatch):
    """Same split, same sampler draws, same cross-frequency partners (partner generators seeded alike)."""
    from libs.experiment_code import code_module
    from thesis_code.s04_dataset import loader
    orig = code_module(EXP_NAME, "loader")
    real = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: real(0) if not a and not k else real(*a, **k))
    loaders = [m.create_multifreq_data_loaders(str(DATA / "normalized"), **LOADER_KW) for m in (orig, loader)]
    for which in (0, 1):                                  # train, then held-out
        for i, (x, y) in enumerate(zip(loaders[0][which], loaders[1][which])):
            assert set(x) == set(y), f"batch {i} keys"
            for k in x:
                if torch.is_tensor(x[k]):
                    assert_close(x[k].numpy(), y[k].numpy(), f"loader {which} batch {i} {k}", rtol=0, atol=0)
                else:
                    assert x[k] == y[k], f"loader {which} batch {i} {k}"
            if i == 2:
                break


def test_missing_cache_goes_to_output_dir(out_dir, tmp_path):
    """A dataset folder without caches gets them in OUTPUT_DIR, never in the dataset folder."""
    import csv
    from thesis_code.s04_dataset import dataset, loader
    src = DATA / "normalized"
    small = tmp_path / "small"
    for sub in ("heatmap", "PI_freq"):
        (small / sub).mkdir(parents=True)
    with (src / "manifest.csv").open() as fh:
        rd = csv.DictReader(fh)
        rows = [r for _, r in zip(range(60), rd)]
        fields = rd.fieldnames
    with (small / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        for sub in ("heatmap", "PI_freq"):
            (small / sub / r["sample_name"]).symlink_to(src / sub / r["sample_name"])
    for name in ("layouts", "dataset_meta.json", "normalization_stats.json"):
        (small / name).symlink_to(src / name)
    ds = dataset.VAEDataset(str(small), precompute_k=True)
    loader.precompute_multifreq_metadata(ds)
    assert not (small / "k_values_cache.npy").exists() and not (small / "multifreq_meta.json").exists()
    made = sorted(p.name for p in (out_dir / "s04_dataset/cache").rglob("*") if p.is_file())
    assert made == ["k_values_cache.npy", "multifreq_meta.json"]


@pytest.mark.parametrize("pool", [False, True])
def test_indexes_match_package(out_dir, pool):
    from thesis_code.s04_dataset import package
    raw = DATA / "test_pool/raw" if pool else DATA / "raw"
    ref = DATA / "test_pool" if pool else DATA
    package.write_indexes(raw, out_dir / "pkg", test_pool=pool)
    for n in ("index_layouts.csv", "index_maps.csv"):
        assert_same_csv(out_dir / "pkg" / n, ref / n)


def test_readme_counts_from_indexes(tmp_path):
    import csv
    from thesis_code.s04_dataset import package
    (tmp_path / "normalized").symlink_to(DATA / "normalized")
    lay = [{**r, "K": int(r["K"]), "n_maps": int(r["n_maps"])} for r in csv.DictReader((DATA / "index_layouts.csv").open())]
    maps = list(csv.DictReader((DATA / "index_maps.csv").open()))
    tp = list(csv.DictReader((DATA / "test_pool/index_layouts.csv").open()))
    package.write_readme(tmp_path, lay, maps, tp)
    text = (tmp_path / "README.md").read_text()
    assert "30,000 placements, 330,000 maps" in text
    assert "27,000 / 3,000 placements" in text
    assert "18,666 simulated placements" in text
