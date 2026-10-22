from tests.thesis_code.helpers import orig_module


def _epoch_dicts():
    from thesis_code.s06_losses.objective import LOSS_KEYS
    tr = {k: 1.0 + i / 7 for i, k in enumerate(LOSS_KEYS)}
    val = {k: 2.0 + i / 9 for i, k in enumerate(LOSS_KEYS)}
    val.update({"mu_mean": 0.01, "mu_std": 0.9, "std_mean": 0.3, "mu_min": -2.5, "mu_max": 2.7,
                "modality_stats": {m: {"mu_mean": 0.1 * j, "mu_std": 1.0, "std_mean": 0.2, "std_std": 0.05}
                                   for j, m in enumerate(("heatmap", "occupancy", "impedance"))}})
    return tr, val


def test_metric_files_match_original(tmp_path):
    from src_vae.others.vae_logger import VAETrainingLogger as OrigLogger
    from thesis_code.s08_monitoring import metrics
    from thesis_code.s08_monitoring.logger import VAETrainingLogger
    orig = orig_module("training.metrics")
    tr, val = _epoch_dicts()
    for d, logger_cls, mod in (("a", OrigLogger, orig), ("b", VAETrainingLogger, metrics)):
        m = tmp_path / d / "metrics"
        lg = logger_cls(str(m), str(tmp_path / d / "checkpoints"), str(m / "loss.csv"), checkpoint_interval=50)
        for ep in (1, 50, 100):
            mod.log_epoch_metrics(lg, None, m, ep, 0.01, tr, dict(val), True)
    names = sorted(p.name for p in (tmp_path / "a/metrics").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "b/metrics").iterdir())
    for f in names:
        assert (tmp_path / "a/metrics" / f).read_bytes() == (tmp_path / "b/metrics" / f).read_bytes(), f


def test_run_log_is_written_without_mlflow(tmp_path):
    import json
    from thesis_code.s08_monitoring.observability import RunObservability
    obs = RunObservability(exp_name="run", metrics_dir=tmp_path / "metrics", log_dir=tmp_path / "logs",
                           cfg={"seed": 1}, start_epoch=0, world_size=1, device="cpu")
    obs.start()
    obs.log_losses(1, train={"total_loss": 1.0}, val=None)
    obs.end(status="finished")
    assert json.loads((tmp_path / "metrics/run_meta.json").read_text())["exp_name"] == "run"
    events = [json.loads(x)["event"] for x in (tmp_path / "logs/train_events.jsonl").read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end"
