import numpy as np
import torch

from tests.thesis_code.helpers import DATA, EXP, EXP_NAME, REPO, assert_close

INPUTS = REPO / "thesis_code/common/inputs"
SOURCES = {"binary_mask.npy": REPO / "configs/binary_mask.npy",
           "Frequency_data_hz.npy": REPO / "configs/Frequency_data_hz.npy",
           "slot_xy_mm.json": REPO / "configs/slot_xy_mm.json",
           "target_impedance_0.5.npy": REPO / "configs/target_impedance_0.5.npy",
           "target_impedance_0.6.npy": REPO / "configs/target_impedance_0.6.npy",
           "target_impedance_0.8.npy": REPO / "configs/target_impedance_0.8.npy",
           "normalization_stats.json": DATA / "normalized/normalization_stats.json",
           "dataset_meta.json": DATA / "normalized/dataset_meta.json",
           "bare_impedance.npy": REPO / "configs/bare_impedance.npy"}


def _codes(n=256, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 52), dtype=np.int64)
    for i in range(n):
        k = rng.integers(1, 11)
        out[i, rng.choice(52, k, replace=False)] = rng.integers(1, 4, k)
    return out


def test_inputs_are_byte_copies():
    for name, src in SOURCES.items():
        assert (INPUTS / name).read_bytes() == src.read_bytes(), name


def test_shipped_config_equals_training_config():
    """Same keys and values as the thesis run's config.yaml; only the # comment lines differ."""
    import json

    def parse(p):
        return json.loads("\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#")))
    assert parse(INPUTS / "config.yaml") == parse(EXP / "config.yaml")


def test_placement_formats_match_original():
    from libs.experiment_code import code_module
    from libs.data_creation.occupancy import LABELS_ORDERED, _LABEL_TO_COORD
    from libs.data_creation.csv_to_occupancy import labels_v1
    from thesis_code.common import placement
    orig = code_module(EXP_NAME, "occupancy")
    c = torch.as_tensor(_codes())
    assert_close(placement.ensure_batch_onehot(c, 3).numpy(), orig.ensure_batch_onehot(c, 3).numpy(), "one-hot")
    assert placement.LABELS_ORDERED == LABELS_ORDERED and placement.LABEL_TO_COORD == _LABEL_TO_COORD
    assert placement.LABELS_ORDERED == labels_v1


def test_board_mask_matches_original():
    from libs.experiment_code import code_module
    from thesis_code.common import board
    orig = code_module(EXP_NAME, "board")
    x = torch.randn(4, 1, 64, 64, generator=torch.Generator().manual_seed(1))
    assert_close(board.zero_offboard(x).numpy(), orig.zero_offboard(x).numpy(), "zero_offboard")


def test_frequency_helpers_match_original():
    from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm
    from src_vae.others.multifreq_anchors import load_anchors_mhz
    from thesis_code.common import frequency
    for mhz in (1.0, 3.7, 200.0, 599.1):
        assert frequency.pi_freq_mhz_to_norm(mhz) == pi_freq_mhz_to_norm(mhz)
    assert_close(frequency.freqs_mhz(), np.load(REPO / "configs/Frequency_data_hz.npy").reshape(-1) / 1e6, "grid")
    assert frequency.load_anchors_mhz(DATA / "normalized") == load_anchors_mhz(DATA / "normalized")


def test_make_targets_reproduces_shipped_targets(out_dir):
    from thesis_code.common import make_targets
    make_targets.main()
    for v in ("0.5", "0.6", "0.8"):
        assert_close(np.load(out_dir / "common" / f"target_impedance_{v}.npy"),
                     np.load(REPO / f"configs/target_impedance_{v}.npy"), f"target {v}", rtol=0, atol=0)
