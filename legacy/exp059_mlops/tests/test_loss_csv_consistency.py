"""Guards on metrics/loss.csv — the source of truth for per-epoch numbers.

A CSV enforces nothing about what its columns MEAN. exp059_capacity_freq's val_total reads
4.87 at epoch 30, 0.84 at 305 and 2.48 at 450 in one file: same column, same run, but the
loss weights changed partway, so comparing those numbers was meaningless and nothing warned.
These tests catch the structural half of that problem, and flag disabled losses that are
still writing non-zero values.
"""
import csv
import json
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

TRACK = ["exp059_capacity_freq", "exp061_fullencode_clean"]


def _loss_rows(exp: str) -> list[dict]:
    p = REPO_ROOT / "experiments" / exp / "metrics" / "loss.csv"
    if not p.is_file():
        pytest.skip(f"{exp} has no loss.csv yet")
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        pytest.skip(f"{exp}: loss.csv is empty")
    return rows


def _config(exp: str) -> dict:
    p = REPO_ROOT / "experiments" / exp / "config.yaml"
    if not p.is_file():
        pytest.skip(f"{exp} has no config.yaml")
    text = p.read_text(encoding="utf-8")
    return json.loads("\n".join(
        l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")))


@pytest.mark.parametrize("exp", TRACK)
def test_every_row_has_every_column(exp):
    """A mid-run column change silently misaligns every downstream reader."""
    rows = _loss_rows(exp)
    width = len(rows[0])
    for i, r in enumerate(rows):
        assert len(r) == width, f"{exp}: row {i} has {len(r)} fields, header has {width}"
        assert None not in r.values(), f"{exp}: row {i} has more fields than the header"


@pytest.mark.parametrize("exp", TRACK)
def test_values_are_numeric_or_blank(exp):
    """A stray string in a metric column breaks plotting long after the run finished."""
    bad = []
    for i, r in enumerate(_loss_rows(exp)):
        for k, v in r.items():
            if v is None or v == "":
                continue
            try:
                float(v)
            except (TypeError, ValueError):
                bad.append(f"row {i} col {k!r} = {v!r}")
    assert not bad, f"{exp}: non-numeric values:\n" + "\n".join(bad[:10])


@pytest.mark.parametrize("exp", TRACK)
def test_epochs_strictly_increase(exp):
    """Duplicate or out-of-order epochs mean a resume appended over existing history, so
    'the curve' is really two runs stacked."""
    eps = [int(float(r["epoch"])) for r in _loss_rows(exp) if r.get("epoch")]
    dupes = {e for e in eps if eps.count(e) > 1}
    assert not dupes, f"{exp}: duplicate epochs {sorted(dupes)[:10]}"
    assert eps == sorted(eps), f"{exp}: epochs are not in order"


@pytest.mark.parametrize("exp", TRACK)
def test_disabled_losses_stay_zero(exp):
    """If a loss is switched off in the config, its column must not carry signal. exp059 ran
    with physics=None hard-coded while physics_* weights were non-zero in the config -- the
    columns existed and read 0.0, which is correct but was easy to misread as 'physics is
    training and contributing nothing'."""
    rows = _loss_rows(exp)
    cfg = _config(exp)
    checks = {
        "heatmap_contrast_weight": "contrast",
        "cross_freq_weight": "cross_freq",
    }
    problems = []
    for key, frag in checks.items():
        weight = cfg.get(key)
        if weight is None or float(weight) > 0:
            continue
        for col in rows[0]:
            if frag not in col:
                continue
            vals = [abs(float(r[col])) for r in rows if r.get(col) not in (None, "")]
            if vals and max(vals) > 1e-8:
                problems.append(f"{col} is non-zero (max {max(vals):.3g}) but {key}={weight}")
    assert not problems, f"{exp}: " + "; ".join(problems)


def test_physics_columns_zero_when_physics_inactive():
    """run_epoch_encode passes physics=None, so the physics loss is not computed. Its columns
    should therefore be exactly zero -- if they ever aren't, the wiring changed."""
    rows = _loss_rows("exp061_fullencode_clean")
    cols = [c for c in rows[0] if "physics" in c]
    if not cols:
        pytest.skip("no physics columns")
    for c in cols:
        vals = [abs(float(r[c])) for r in rows if r.get(c) not in (None, "")]
        assert not vals or max(vals) < 1e-8, (
            f"{c} is non-zero (max {max(vals):.3g}) but physics is passed as None")


def test_crossfreq_metric_is_being_recorded():
    """The metric exp061 is judged on. Baseline to beat: 0.5588 (exp059 ep545, seed-42
    split, geometric mask). If this file stops appearing the run cannot be evaluated."""
    p = REPO_ROOT / "experiments" / "exp061_fullencode_clean" / "metrics" / "crossfreq_eval.csv"
    if not p.is_file():
        pytest.skip("run has not reached its first eval yet")
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "crossfreq_eval.csv exists but is empty"
    assert "pearson_at_target" in rows[0], "the scored-against-truth column is missing"
    for r in rows:
        v = float(r["pearson_at_target"])
        assert -1.0 <= v <= 1.0, f"pearson out of range: {v}"
