"""Build ECADStar .peb files for multi-type combinations batch simulation.

Mirrors ``pipelines/dataset_sim/peb.py`` but each slot carries a type code
(0 empty / 1 type-1 / 2 type-2) and the PEB sets per-component C / ESR / ESL
from the type catalog (see ``scrap/generation/generate_peb_multitype.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pipelines.dataset_sim.combinations_multitype import N_DECAPS


def _import_generate(repo_root: Path):
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from scrap.generation.generate_peb_multitype import (  # noqa: E402
        TYPE_CATALOG,
        generate_peb_multitype,
    )

    return generate_peb_multitype, TYPE_CATALOG


def mhz_to_freq_hz(mhz: float) -> str:
    return f"{int(round(float(mhz)))}e6"


def build_impedance_peb_multitype(
    type_codes: np.ndarray,
    output_path: Path,
    *,
    repo_root: Path,
    powerbus: str,
    components: str,
) -> Path:
    """CreatePISpectrum only — one PI per layout row (type-aware)."""
    if type_codes.ndim == 1:
        type_codes = type_codes.reshape(1, -1)
    if type_codes.shape[1] != N_DECAPS:
        raise ValueError(f"expected (*, {N_DECAPS}), got {type_codes.shape}")

    generate_peb_multitype, _ = _import_generate(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_peb_multitype(
        type_codes=type_codes,
        output_path=str(output_path),
        powerbus=powerbus,
        freq="1e6",  # unused when distribution disabled
        components=components,
        include_distribution=False,
        include_spectrum=True,
    )
    return output_path


def build_distribution_peb_multitype(
    type_codes: np.ndarray,
    output_path: Path,
    *,
    repo_root: Path,
    mhz: float,
    powerbus: str,
) -> Path:
    """PI-Distribution only — one PI per layout row at ``mhz`` (type-aware)."""
    if type_codes.ndim == 1:
        type_codes = type_codes.reshape(1, -1)
    if type_codes.shape[1] != N_DECAPS:
        raise ValueError(f"expected (*, {N_DECAPS}), got {type_codes.shape}")

    freq_hz = mhz_to_freq_hz(mhz)
    generate_peb_multitype, _ = _import_generate(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_peb_multitype(
        type_codes=type_codes,
        output_path=str(output_path),
        powerbus=powerbus,
        freq=freq_hz,
        components="IC1_Port1,IC2_Port2",
        include_distribution=True,
        include_spectrum=False,
    )
    return output_path
