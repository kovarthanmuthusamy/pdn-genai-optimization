import numpy as np
import pytest

from tests.thesis_code.helpers import EXP_NAME, REPO


@pytest.mark.parametrize("target,margin", [("0.8", None), ("0.6", 0.0)])
def test_assistant_matches_original(monkeypatch, target, margin):
    """Same failing placement, same target and margin: the same verdict and the same fixes in the same order."""
    from pipelines.assistant import suggest_layout as orig
    from thesis_code.s11_assistant import suggest
    monkeypatch.setattr(orig, "PREDICTOR", "vae")
    monkeypatch.setattr(orig, "VAE_HEAD", "H1_spec_hm200")
    monkeypatch.setattr(orig, "TARGET_PATH", f"configs/target_impedance_{target}.npy")
    monkeypatch.setattr(orig, "SAFETY_MARGIN", margin)
    codes = np.zeros(52, dtype=np.int64)
    codes[[5, 19, 30, 49, 50]] = 3
    band = np.arange(231)
    tpath = REPO / f"thesis_code/common/inputs/target_impedance_{target}.npy"
    tgt = np.load(tpath).reshape(-1)
    ja = orig.load_judge(EXP_NAME, tgt, band)
    jb = suggest.load_judge(tgt, band, margin=margin, target_path=tpath)
    pa, sa, ra = ja.assess(codes[None])
    pb, sb, rb = jb.assess(codes[None])
    assert pa.tolist() == pb.tolist() and np.allclose(sa, sb, rtol=1e-6) and list(ra) == list(rb)
    a = orig.find_fixes(codes, ja, 3, max_radius=2, min_suggestions=1, seed=42, log=None)
    b = suggest.find_fixes(codes, jb, 3, max_radius=2, min_suggestions=1, seed=42, log=None)
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert x["codes"].tolist() == y["codes"].tolist()
        assert (x["edits"], x["moves"], x["radius"]) == (y["edits"], y["moves"], y["radius"])
        assert np.isclose(x["score"], y["score"], rtol=1e-6)


def test_assistant_main_writes_report(out_dir):
    from thesis_code.s11_assistant import suggest
    suggest.main()
    import json
    out = json.loads((out_dir / "s11_assistant/suggestions.json").read_text())
    assert out["K"] == 5 and "verdict" in out
