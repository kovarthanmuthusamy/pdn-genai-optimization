import sys

import numpy as np
import pytest
import torch

from tests.thesis_code.helpers import EXP, EXP_NAME, REPO, assert_close
from tests.thesis_code.test_common import _codes


@pytest.fixture(scope="module")
def both():
    sys.path.insert(0, str(REPO / "scrap"))
    import ch4_common as c
    orig = c.Surrogate(device=torch.device("cpu"))
    from thesis_code.s09_inference import placement_encoder
    from thesis_code.s09_inference.inference import VAEInference
    return orig, VAEInference(device=torch.device("cpu")), placement_encoder.load(device=torch.device("cpu"))


def test_spectra_match_original(both):
    orig, _, enc = both
    codes = _codes(500, seed=3)
    assert_close(enc.predict_ohm(codes), orig.spectra(codes), "spectra")


def test_maps_match_original(both):
    orig, vae, enc = both
    from thesis_code.s12_evaluation.common import Surrogate
    codes = _codes(64, seed=4)
    mhz = np.full(64, 200.0)
    assert_close(Surrogate(device=torch.device("cpu")).maps(codes, mhz), orig.maps(codes, mhz), "200 MHz maps")


def test_latent_stats_and_denorm_match_original(both):
    orig, vae, _ = both
    assert vae.latent_stats == orig.vae.latent_stats
    assert vae.per_K_latent_stats == orig.vae.per_K_latent_stats
    hm = torch.randn(3, 1, 64, 64, generator=torch.Generator().manual_seed(5))
    for mhz in (2.3, 200.0, torch.tensor([3.0, 150.5, 580.0])):
        assert_close(vae.denorm_heatmap_physical(hm, mhz=mhz).numpy(),
                     orig.vae.denorm_heatmap_physical(hm, mhz=mhz).numpy(), f"denorm {mhz}")


def test_margin_matches_original():
    from pipelines.assistant.vae_impedance import zero_false_pass_margin
    from thesis_code.s09_inference import placement_encoder
    for v in ("0.5", "0.6", "0.8"):
        t = f"configs/target_impedance_{v}.npy"
        assert placement_encoder.margin(REPO / t) == zero_false_pass_margin(EXP_NAME, "H1_spec_hm200", t)


def test_head_stamp_accepts_copied_weights(tmp_path, capsys):
    import shutil
    from thesis_code.s09_inference.placement_encoder import check_head_matches
    p = tmp_path / "last_model.pt"
    shutil.copy(EXP / "checkpoints/last_model.pt", p)       # new mtime, same size
    ck = torch.load(EXP / "checkpoints/vae_route_finetune/H1_spec_hm200.pt", map_location="cpu", weights_only=False)
    check_head_matches(ck, p)
    assert "new modification time" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        check_head_matches({**ck, "vae_stamp": [1, ck["vae_stamp"][1]]}, p)


def test_missing_weights_names_setting(monkeypatch, tmp_path):
    from thesis_code.common import settings
    from thesis_code.s09_inference import placement_encoder
    monkeypatch.setattr(settings, "WEIGHTS_DIR", tmp_path / "nowhere")
    with pytest.raises(SystemExit, match="WEIGHTS_DIR"):
        placement_encoder.load(device=torch.device("cpu"))


def test_inference_runs_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from thesis_code.s09_inference import placement_encoder
    enc = placement_encoder.load()
    assert enc.device.type == "cpu" and enc.predict_ohm(_codes(4)).shape == (4, 231)


def test_margin_script_reproduces_saved_margins(out_dir):
    """Margins, pass counts and kept shares equal the saved file (floats to 1e-9 relative: the saved
    file was measured on the GPU, this run is on whatever device the test machine offers)."""
    from tests.thesis_code.helpers import assert_same_json, load_json
    from thesis_code.s09_inference import margin
    margin.main()
    new = load_json(out_dir / "s09_inference/H1_spec_hm200_margin.json")
    old = load_json(EXP / "checkpoints/vae_route_finetune/H1_spec_hm200_margin.json")
    assert_same_json(new, old, ignore=("path", "experiment"))
