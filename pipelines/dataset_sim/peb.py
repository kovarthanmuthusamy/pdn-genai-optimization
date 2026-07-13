"""Build ECADStar .peb files for combinations batch simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pipelines.dataset_sim.combinations import N_DECAPS


def _import_generate_peb(repo_root: Path):
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from scrap.generation.generate_peb import generate_peb  # noqa: E402

    return generate_peb


def mhz_to_freq_hz(mhz: float) -> str:
    return f"{int(round(float(mhz)))}e6"


def build_impedance_peb(
    occupancy: np.ndarray,
    output_path: Path,
    *,
    repo_root: Path,
    powerbus: str,
    components: str,
) -> Path:
    """CreatePISpectrum only — one PI per layout row."""
    if occupancy.ndim == 1:
        occupancy = occupancy.reshape(1, -1)
    if occupancy.shape[1] != N_DECAPS:
        raise ValueError(f"expected (*, {N_DECAPS}), got {occupancy.shape}")

    generate_peb = _import_generate_peb(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_peb(
        occupancy=occupancy,
        output_path=str(output_path),
        powerbus=powerbus,
        freq="1e6",  # unused when distribution disabled
        components=components,
        include_distribution=False,
        include_spectrum=True,
    )
    return output_path


def build_distribution_peb(
    occupancy: np.ndarray,
    output_path: Path,
    *,
    repo_root: Path,
    mhz: float,
    powerbus: str,
) -> Path:
    """PI-Distribution only — one PI per layout row at ``mhz``."""
    if occupancy.ndim == 1:
        occupancy = occupancy.reshape(1, -1)
    if occupancy.shape[1] != N_DECAPS:
        raise ValueError(f"expected (*, {N_DECAPS}), got {occupancy.shape}")

    freq_hz = mhz_to_freq_hz(mhz)
    generate_peb = _import_generate_peb(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_peb(
        occupancy=occupancy,
        output_path=str(output_path),
        powerbus=powerbus,
        freq=freq_hz,
        components="IC1_Port1,IC2_Port2",
        include_distribution=True,
        include_spectrum=False,
    )
    return output_path
