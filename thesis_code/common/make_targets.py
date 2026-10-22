"""Build the target spectra: a flat floor from 1 to 10 MHz, rising to an end value at 600 MHz.

What: log-log linear interpolation through (1 MHz, floor), (10 MHz, floor) and (600 MHz, end),
    evaluated at the 231 spectrum frequencies and clipped to [floor, end]. Three targets:
    0.8 ohm -> 50 ohm, 0.6 ohm -> 36 ohm, 0.5 ohm -> 30 ohm. The same files ship in common/inputs/.
Usage: python -m thesis_code.common.make_targets
    Writes OUTPUT_DIR/common/target_impedance_<floor>.npy, shape (231, 1) float64.
Config keys: LOW_FREQ_MHZ, CORNER_FREQ_MHZ, HIGH_FREQ_MHZ, PLATEAU_VALUES_OHM.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from thesis_code.common.frequency import freqs_mhz
from thesis_code.common.paths import out_dir

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
LOW_FREQ_MHZ = 1.0
CORNER_FREQ_MHZ = 10.0
HIGH_FREQ_MHZ = 600.0
PLATEAU_VALUES_OHM = [(0.8, 50.0), (0.6, 36.0), (0.5, 30.0)]   # (floor, value at 600 MHz) in ohm
# =============================================================================


def build(freq_mhz: np.ndarray, plateau_ohm: float, end_ohm: float) -> np.ndarray:
    """Target in ohm at ``freq_mhz``."""
    tar_freq = [LOW_FREQ_MHZ, CORNER_FREQ_MHZ, HIGH_FREQ_MHZ]
    tar_imp = [plateau_ohm, plateau_ohm, end_ohm]
    f = interp1d(np.log10(tar_freq), np.log10(tar_imp), kind="linear", fill_value="extrapolate")
    imp = 10 ** f(np.log10(freq_mhz))
    return np.clip(imp, plateau_ohm, end_ohm)


def main() -> None:
    f_mhz = freqs_mhz()
    out = out_dir("common")
    for plateau, end_ohm in PLATEAU_VALUES_OHM:
        imp = build(f_mhz, plateau, end_ohm).reshape(-1, 1).astype(np.float64)
        path = out / f"target_impedance_{plateau:g}.npy"
        np.save(path, imp)
        print(f"floor {plateau} ohm, end {end_ohm} ohm -> {path}")


if __name__ == "__main__":
    main()
