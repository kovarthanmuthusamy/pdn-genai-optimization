"""Guards for the silent-failure modes found on 2026-08-13.

Each test here corresponds to a bug that cost real time and produced numbers that looked
plausible. None of them raised an error at the time -- that is why they are tests now.
See docs/full-encode-reentry-findings.md sections 8 and 9.
"""
import json
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

EXPERIMENTS = ["exp059_capacity_freq", "exp061_fullencode_clean"]


def _config(exp: str) -> dict:
    path = REPO_ROOT / "experiments" / exp / "config.yaml"
    if not path.is_file():
        pytest.skip(f"{exp} has no config.yaml")
    text = path.read_text(encoding="utf-8")
    return json.loads("\n".join(
        l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")))


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_config_parses(exp):
    """config.yaml is JSON with # comments. A trailing-newline slip once ate the opening
    brace, and the file silently failed to load."""
    cfg = _config(exp)
    assert isinstance(cfg, dict) and cfg, f"{exp}: config did not parse to a non-empty dict"


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_resume_checkpoint_stays_inside_its_own_experiment(exp):
    """exp061 inherited exp059's resume path when its config was copied. It resumed epoch-465
    weights, saw 465 > num_epochs, exited having trained nothing, and would have invalidated
    the comparison had it run."""
    cfg = _config(exp)
    resume = cfg.get("resume_checkpoint")
    if not resume:
        return
    assert exp in str(resume), (
        f"{exp}: resume_checkpoint points at another experiment ({resume})")


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_experiment_dir_matches_folder(exp):
    """A copied config kept the wrong experiment_dir and wrote metrics into its source."""
    cfg = _config(exp)
    assert cfg.get("experiment_dir", "").rstrip("/").endswith(exp), (
        f"{exp}: experiment_dir is {cfg.get('experiment_dir')!r}")


def test_clip_floor_below_background():
    """THE bug: heatmap_z_clip_min (-1.6792) sat ABOVE background_value (-3.1792), so
    _prepare_batch's clamp lifted the background over the foreground threshold and every
    'foreground' mask selected 100% of the grid in training while eval used 83.8%."""
    cfg = _config("exp061_fullencode_clean")
    lo = cfg.get("heatmap_z_clip_min")
    bg = cfg.get("background_value")
    if lo is None or bg is None:
        pytest.skip("clip floor or background not set")
    assert float(lo) <= float(bg), (
        f"clip floor {lo} is above background {bg}: clamping will destroy the mask")


def test_board_mask_matches_expected_coverage():
    """Foreground is the board geometry, not a value test. If this mask ever changes shape,
    every masked loss and metric changes meaning."""
    import numpy as np

    mask = np.load(REPO_ROOT / "configs" / "binary_mask.npy")
    assert mask.shape == (64, 64), f"unexpected mask shape {mask.shape}"
    coverage = float((mask > 0).mean())
    assert 0.83 < coverage < 0.85, f"board coverage {coverage:.4f} is not the expected ~0.8379"


def test_split_seed_is_explicit_and_pinned():
    """train_core hardcoded seed=42 for the layout holdout while the config's `seed` (1234)
    drove only torch/numpy. Evaluation scripts that passed the config seed scored on a
    different split -- 89.6% of those samples were in the training set."""
    cfg = _config("exp061_fullencode_clean")
    assert "split_seed" in cfg, "split_seed must be explicit, never a literal in train_core"
    assert int(cfg["split_seed"]) == 42, (
        "split_seed is pinned at 42 so runs stay comparable; changing it changes the "
        "held-out set and silently invalidates every cross-run comparison")


def test_train_core_reads_split_seed_from_config():
    """The literal must not come back."""
    src = (REPO_ROOT / "experiments" / "exp061_fullencode_clean" / "codes"
           / "train_core.py").read_text(encoding="utf-8")
    assert "seed=42" not in src, "train_core still hardcodes the split seed"
    assert "split_seed" in src, "train_core does not read split_seed"


def test_training_does_not_overwrite_source_config():
    """train_core used to write the runtime state back over config.yaml on exit, destroying
    the comment header, reverting deliberate edits, and dropping architecture keys that
    TrainConfig does not carry (imp_peak_dim vanished this way)."""
    src = (REPO_ROOT / "experiments" / "exp061_fullencode_clean" / "codes"
           / "train_core.py").read_text(encoding="utf-8")
    assert '(exp / "config.yaml").write_text' not in src, (
        "train_core writes back over config.yaml; use config.runtime.yaml")


def test_full_encode_fork_has_no_dropped_paths():
    """exp061 exists to train ONE path. If the occupancy-only encode returns, 63% of samples
    go back to training a path the framework does not deploy."""
    codes = REPO_ROOT / "experiments" / "exp061_fullencode_clean" / "codes"
    banned = ("occ_only_encode_prob", "layout_train_prob", "cross_modal_weight",
              "latent_distill_weight", "output_distill_weight")
    offenders = []
    for f in codes.glob("*.py"):
        if f.name in ("board_mask.py", "inference_vae.py"):
            continue
        text = f.read_text(encoding="utf-8")
        for b in banned:
            # a mention inside an explanatory comment is fine; a real read is not
            for line in text.splitlines():
                if b in line and not line.strip().startswith("#"):
                    offenders.append(f"{f.name}: {line.strip()[:70]}")
    assert not offenders, "dropped training paths came back:\n" + "\n".join(offenders)


def test_background_value_config_survives_yaml_apply():
    """background_value=0 (geometric masking) was silently overwritten twice: once by the
    old config.yaml clobber (fixed, but only after it had already baked -3.1792 into the
    file), and independently by an unconditional stats-file override in train_core.py that
    ran on every launch regardless. Both together meant the entire first exp061 retrain
    trained with background_value=-3.1792, weakening peak-location loss terms that weight
    pixels by (value - background_value).clamp(min=0). This exercises the real load path."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from repo_paths import setup_path
    setup_path()
    from experiments.exp061_fullencode_clean.codes.config_schema import load_yaml_config
    from experiments.exp061_fullencode_clean.codes.train_vae_simple import Config
    import experiments.exp061_fullencode_clean.codes.train_core as tr

    cfg_path = REPO_ROOT / "experiments" / "exp061_fullencode_clean" / "config.yaml"
    cfg = load_yaml_config(cfg_path)
    assert cfg.get("background_value") == 0.0, "config.yaml no longer sets background_value=0"

    c = Config(**{k: v for k, v in cfg.items() if k in Config.model_fields})
    tr._apply_yaml_config(c)
    assert c.background_value == 0.0, (
        f"background_value={c.background_value} after _apply_yaml_config -- "
        "the config value did not survive to the Config object used by training")
