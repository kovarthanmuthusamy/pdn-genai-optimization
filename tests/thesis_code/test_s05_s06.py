import torch

from tests.thesis_code.helpers import EXP, EXP_NAME, REPO, assert_close

CONFIG = REPO / "thesis_code/common/inputs/config.yaml"


def _models():
    from libs.experiment_code import code_module
    cfg_mod = code_module(EXP_NAME, "config")
    from thesis_code.s05_model import MultiInputVAE
    from thesis_code.s07_training.config import load_train_config, vae_model_kwargs
    orig_cls = code_module(EXP_NAME, "model").MultiInputVAE
    sd = torch.load(EXP / "checkpoints/last_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    orig = orig_cls(**cfg_mod.vae_model_kwargs(cfg_mod.load_train_config([EXP / "config.yaml"]))).eval()
    orig.load_state_dict(sd)
    new = MultiInputVAE(**vae_model_kwargs(load_train_config([CONFIG]))).eval()
    new.load_state_dict(sd)
    return orig, new


def _batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(n, 52, dtype=torch.long)
    for i in range(n):
        idx = torch.randperm(52, generator=g)[: 1 + i % 10]
        codes[i, idx] = torch.randint(1, 4, (len(idx),), generator=g)
    from thesis_code.common.placement import ensure_batch_onehot
    occ = ensure_batch_onehot(codes, 3)
    hm = torch.randn(n, 1, 64, 64, generator=g)
    imp = torch.randn(n, 231, generator=g)
    K = (codes > 0).sum(1)
    pi = torch.rand(n, generator=g)
    return hm, occ, imp, K, pi


def test_same_state_dict_keys():
    orig, new = _models()
    assert list(orig.state_dict()) == list(new.state_dict())


def test_forward_matches_original():
    orig, new = _models()
    hm, occ, imp, K, pi = _batch()
    torch.manual_seed(0)
    with torch.inference_mode():
        a = orig(hm, occ, imp, K, pi)
    torch.manual_seed(0)
    with torch.inference_mode():
        b = new(hm, occ, imp, K, pi)
    flat_a = a if isinstance(a, (tuple, list)) else (a,)
    flat_b = b if isinstance(b, (tuple, list)) else (b,)
    assert len(flat_a) == len(flat_b)
    for i, (x, y) in enumerate(zip(flat_a, flat_b)):
        if torch.is_tensor(x):
            assert_close(x.numpy(), y.numpy(), f"output {i}")
        elif isinstance(x, dict):
            assert set(x) == set(y)
            for k in x:
                if torch.is_tensor(x[k]):
                    assert_close(x[k].numpy(), y[k].numpy(), f"output {i}[{k}]")


def test_losses_match_original():
    from libs.experiment_code import code_module
    from tests.thesis_code.helpers import orig_module
    lo, sched = orig_module("losses.objective"), orig_module("training.schedule")
    cfg = code_module(EXP_NAME, "config").load_train_config([EXP / "config.yaml"])
    cfg.background_value = -3.0
    from thesis_code.s06_losses import objective
    from thesis_code.s07_training.config import load_train_config
    c2 = load_train_config([CONFIG])
    c2.background_value = -3.0
    orig, new = _models()
    hm, occ, imp, K, pi = _batch(seed=1)
    with torch.inference_mode():
        rh, ro, ri, mu, lv, ex = orig(hm, occ, imp.unsqueeze(1), K, pi)
    kw = dict(beta=0.01, imp_log_std=1.0, weights=sched.phase_weights(650, cfg),
              penalty_scale=sched.penalty_scale(650, cfg), peak_scale=sched.peak_loss_scale(650, cfg))
    a = lo.vae_loss(rh, ro, ri, hm, occ, imp.unsqueeze(1), mu, lv, ex, K, c=cfg, **kw)
    b = objective.vae_loss(rh, ro, ri, hm, occ, imp.unsqueeze(1), mu, lv, ex, K, c=c2, **kw)
    assert set(a) == set(b)
    for k in a:
        assert_close(torch.as_tensor(a[k]).numpy(), torch.as_tensor(b[k]).numpy(), k)
