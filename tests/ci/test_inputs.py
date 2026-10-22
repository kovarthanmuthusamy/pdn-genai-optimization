"""The small inputs that ship with thesis_code/ (board, frequency grid, targets, config)."""
import json

import numpy as np

from thesis_code.common.board import INPUTS
from thesis_code.common.frequency import freqs_mhz
from thesis_code.s07_training.config import CONFIG_FILE, load_experiment_config_dict


def test_frequency_grid():
    f = freqs_mhz()
    assert f.shape == (231,) and np.all(np.diff(f) > 0)
    assert abs(f[0] - 1.0) < 1e-6 and abs(f[-1] - 600.0) < 1e-3


def test_board_mask_is_an_h():
    m = np.load(INPUTS / "binary_mask.npy").astype(bool)
    assert m.shape == (64, 64) and 0.5 < m.mean() < 0.95
    assert not m[5, 32] and not m[57, 32]                 # notches in the middle of the bottom and top edges
    assert m[32, 32] and m[5, 5] and m[57, 5] and m[57, 58]   # centre and the four arms


def test_slots_and_targets():
    board = json.loads((INPUTS / "slot_xy_mm.json").read_text())
    assert len(board["slots"]) == 52 and len(board["ic_port"]) == 2
    for name in ("0.8", "0.6", "0.5"):
        t = np.load(INPUTS / f"target_impedance_{name}.npy").reshape(-1)
        assert t.shape == (231,) and (t > 0).all()
        assert abs(t[0] - float(name)) < 0.05 * float(name)      # the target allows its value at 1 MHz
    bare = np.load(INPUTS / "bare_impedance.npy").reshape(-1)
    assert bare.shape == (231,) and (bare > 0).all()


def test_config_is_json_with_comments():
    cfg = load_experiment_config_dict(CONFIG_FILE)
    assert cfg["latent_dim"] == 128 and cfg["n_decap_types"] == 3
