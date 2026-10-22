"""The thesis model architecture, built from the shipped config with random weights, on the CPU."""
import pytest
import torch

from thesis_code.common.frequency import pi_freq_mhz_to_norm
from thesis_code.common.placement import type_ids_to_onehot
from thesis_code.s05_model.vae import MultiInputVAE
from thesis_code.s07_training.config import CONFIG_FILE, load_experiment_config_dict, vae_model_kwargs

B = 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MultiInputVAE(**vae_model_kwargs(load_experiment_config_dict(CONFIG_FILE)))


def _batch():
    g = torch.Generator().manual_seed(1)
    codes = torch.zeros(B, 52, dtype=torch.long)
    k = torch.tensor([1, 3, 6, 10])
    for i in range(B):
        slots = torch.randperm(52, generator=g)[: int(k[i])]
        codes[i, slots] = torch.randint(1, 4, (int(k[i]),), generator=g)
    occ = type_ids_to_onehot(codes, 3).float()
    hm = torch.randn(B, 1, 64, 64, generator=g)
    imp = torch.randn(B, 1, 231, generator=g)
    freq = torch.full((B,), float(pi_freq_mhz_to_norm(100.0)))
    return hm, occ, imp, k, freq


def test_config_matches_the_thesis_model(model):
    kw = vae_model_kwargs(load_experiment_config_dict(CONFIG_FILE))
    assert (kw["latent_dim"], kw["n_occ_classes"], kw["k_max"]) == (128, 3, 10)
    assert sum(p.numel() for p in model.parameters()) > 1_000_000


def test_forward_shapes_and_backward(model):
    model.train()
    hm, occ, imp, k, freq = _batch()
    hm_r, occ_r, imp_r, mu, logvar, experts = model(hm, occ, imp, k, freq)
    assert hm_r.shape == (B, 1, 64, 64)
    assert occ_r.shape == (B, 52, 3)
    assert imp_r.shape == (B, 1, 231)
    assert mu.shape == logvar.shape == (B, 128)
    assert set(experts) == {"heatmap", "occupancy", "impedance"}
    loss = hm_r.pow(2).mean() + imp_r.pow(2).mean() + mu.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    model.zero_grad()


def test_eval_is_deterministic(model):
    model.eval()
    hm, occ, imp, k, freq = _batch()
    with torch.no_grad():
        _, mu1, _, _ = model.encode(hm, occ, imp, k, freq)
        _, mu2, _, _ = model.encode(hm, occ, imp, k, freq)
        out = model.decode(mu1, k, freq, occupancy=occ)
    assert torch.equal(mu1, mu2)
    assert all(torch.isfinite(t).all() for t in out)
