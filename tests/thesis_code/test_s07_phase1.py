import itertools

import numpy as np
import torch

from tests.thesis_code.helpers import DATA, EXP, REPO, assert_close, orig_module

N_BATCHES = 6
CONFIG = REPO / "thesis_code/common/inputs/config.yaml"


def _cpu(cfg):
    cfg.device = "cpu"
    cfg.batch_size = 16
    return cfg


def _batches(monkeypatch):
    real = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: real(0) if not a and not k else real(*a, **k))
    tl, _ = orig_module("data.loader").create_multifreq_data_loaders(
        str(DATA / "normalized"), batch_size=16, num_workers=0, seed=42, pin_memory=False,
        train_samples_per_epoch=64000, n_decap_types=3)
    return list(itertools.islice(tl, N_BATCHES))


def _run(loop_mod, cfg, batches, epoch):
    imp_log_std = loop_mod.apply_normalization_stats(cfg)
    torch.manual_seed(1)
    model = loop_mod.build_model(cfg)
    sd = torch.load(EXP / "checkpoints/last_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd)
    opt = loop_mod.build_optimizer(model, cfg)
    md = loop_mod.modality_dropout_at(epoch, cfg) if hasattr(loop_mod, "modality_dropout_at") else \
        orig_module("training.schedule").modality_dropout_at(epoch, cfg)
    torch.manual_seed(2)
    out = loop_mod.run_epoch(model, batches, cfg, epoch, float(cfg.beta_final), md, imp_log_std, train=True, optimizer=opt)
    return out, {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_phase1_matches_original_early_and_late_epoch(monkeypatch):
    oc = orig_module("config")
    cfg = _cpu(oc.load_train_config([EXP / "config.yaml"]))
    cfg.data_dir = str(DATA / "normalized")
    from thesis_code.s07_training import config as nc, loop
    cfg2 = _cpu(nc.load_train_config([CONFIG]))
    batches = _batches(monkeypatch)
    for epoch in (0, 650):            # 650: every curriculum stage is on (cross-frequency, anti-copy, focus)
        a, wa = _run(orig_module("training.loop"), cfg, batches, epoch)
        b, wb = _run(loop, cfg2, batches, epoch)
        for k in a:
            if isinstance(a[k], float):
                assert_close(a[k], b[k], f"epoch {epoch} {k}")
        for k in wa:
            assert_close(wa[k].numpy(), wb[k].numpy(), f"epoch {epoch} weight {k}")


def test_fresh_run_cleanup_stays_in_output(out_dir, tmp_path):
    import pytest
    from thesis_code.s07_training import checkpoints
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    (outside / "checkpoints").mkdir(parents=True)
    with pytest.raises(SystemExit, match="OUTPUT_DIR"):
        checkpoints.clean_previous_run(outside)
    assert (outside / "checkpoints").is_dir()
    inside = out_dir / "s07_training"
    (inside / "metrics").mkdir(parents=True)
    checkpoints.clean_previous_run(inside)
    assert not (inside / "metrics").exists()
