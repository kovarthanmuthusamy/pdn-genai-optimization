import torch

from tests.thesis_code.helpers import EXP, EXP_NAME, assert_close

STEPS = 50


def _capture(monkeypatch):
    """The original writes into experiments/; keep those saves in memory instead."""
    saved = {}
    real = torch.save

    def fake(obj, path, *a, **k):
        if "experiments" in str(path):
            saved[str(path)] = obj
        else:
            real(obj, path, *a, **k)
    monkeypatch.setattr(torch, "save", fake)
    return saved


def test_finetune_matches_original(monkeypatch, out_dir):
    from pipelines.assistant import vae_impedance as orig
    saved = _capture(monkeypatch)
    ck = torch.load(EXP / "checkpoints/vae_route_finetune/H1_spec_hm200.pt", map_location="cpu", weights_only=False)
    kw = dict(steps=STEPS, w_imp=ck["losses"]["w_imp"], w_hm=ck["losses"]["w_hm"], w_anchor=ck["losses"]["w_anchor"],
              hm_ref_dir=ck["hm_ref_dir"], seed=7, device=torch.device("cpu"))
    orig.finetune_route(EXP_NAME, "last_model.pt", "equiv_test", verbose=False, **kw)
    a = next(v for k, v in saved.items() if k.endswith("equiv_test.pt"))
    from thesis_code.s07_training import phase2
    phase2.finetune("equiv_test", **kw)
    b = torch.load(out_dir / "s07_training/phase2/equiv_test.pt", map_location="cpu", weights_only=False)
    for part in ("impedance_decoder", "network"):
        assert list(a[part]) == list(b[part])
        for k in a[part]:
            assert_close(a[part][k].numpy(), b[part][k].numpy(), f"{part}.{k}")
    for key in ("heldout_before", "heldout_after"):
        for m in a[key]:
            assert_close(a[key][m], b[key][m], f"{key} {m}")


def test_distiller_matches_original(monkeypatch, out_dir):
    from pipelines.assistant import vae_impedance as orig
    saved = _capture(monkeypatch)
    torch.manual_seed(0)
    orig.load_or_train(EXP_NAME, "last_model.pt", device=torch.device("cpu"),
                       steps=STEPS, n_train=512, force=True, verbose=False)
    a = next(v for k, v in saved.items() if k.endswith("occ_to_z_distill.pt"))
    from thesis_code.s07_training import phase2
    torch.manual_seed(0)
    phase2.train_distiller(steps=STEPS, n_train=512, device=torch.device("cpu"))
    b = torch.load(out_dir / "s07_training/phase2/occ_to_z_distill.pt", map_location="cpu", weights_only=False)
    for k in a["state_dict"]:
        assert_close(a["state_dict"][k].numpy(), b["state_dict"][k].numpy(), k)
    for m in ("val_cos", "val_med_rel", "k_range", "n_train", "target"):
        assert_close(a[m], b[m], m) if isinstance(a[m], float) else None
        assert isinstance(a[m], float) or a[m] == b[m], m
