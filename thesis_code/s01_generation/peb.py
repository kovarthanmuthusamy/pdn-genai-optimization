"""eCADSTAR batch files (PEB) for decap placements.

What: a placement is a row of 52 type codes (0 empty, 1..3 a part). For every slot the PEB sets
    four values: mounted or not, capacitance, ESR and ESL, taken from TYPE_CATALOG (SI units):
        T1 100 nF, ESR  8.9 mOhm, ESL 222 pH
        T2  47 nF, ESR 21.4 mOhm, ESL 154 pH
        T3  22 nF, ESR 25.2 mOhm, ESL 142 pH
    Empty slots write ``false`` and 0. One analysis group per placement: a spectrum (CreatePISpectrum)
    or a map at one frequency (PI-Distribution). Each placement may have its own map frequency in
    the same file; eCADSTAR bills per group, so this costs the same as one shared frequency.
Usage (import only):
    build_impedance_peb(codes, path, powerbus=POWERBUS, components=COMPONENTS) -> path
    build_distribution_peb(codes, path, mhz=... | per_sample_mhz=[...], powerbus=POWERBUS) -> path
    load_multitype_csv(path) -> (N, 52) int8 codes from a plan CSV
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

N_DECAPS = 52
N_DECAP_TYPES = 3
VALID_CODES = set(range(N_DECAP_TYPES + 1))
POWERBUS = "Power_GND"
COMPONENTS = "IC1_Port1"                     # IC port whose spectrum is exported

TYPE_CATALOG: dict[int, dict[str, float]] = {
    1: {"C": 100e-9, "ESR": 8.9e-3, "ESL": 222e-12},
    2: {"C": 47e-9, "ESR": 21.4e-3, "ESL": 154e-12},
    3: {"C": 22e-9, "ESR": 25.2e-3, "ESL": 142e-12},
}


def load_multitype_csv(path: Path) -> np.ndarray:
    """Return (N, 52) int8 rows of type codes in ``VALID_CODES``."""
    rows: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != N_DECAPS:
                raise ValueError(f"{path}: expected {N_DECAPS} columns, got {len(parts)}")
            rows.append([int(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    arr = np.array(rows, dtype=np.int8)
    bad = set(np.unique(arr).tolist()) - VALID_CODES
    if bad:
        raise ValueError(f"{path}: values must be in {sorted(VALID_CODES)}; found extra {sorted(bad)}")
    return arr


def mhz_to_freq_hz(mhz: float) -> str:
    """200.0 -> "200e6", 168.5 -> "168.5e6" (eCADSTAR accepts fractional MHz)."""
    return f"{float(mhz):g}e6"


def _fmt(v: float) -> str:
    """Format a value eCADSTAR-style (e.g. 1e-07, 0.0089, 0)."""
    if v == 0:
        return "0"
    return f"{v:g}"


def build_component_edits(type_row: np.ndarray, indent: str = "      ") -> str:
    """The four Edit lines of each of the 52 slots for one type-code row."""
    lines: list[str] = []
    for idx, code in enumerate(type_row):
        code = int(code)
        name = f"C{idx + 1}"
        if code == 0:
            enabled, cval, esr, esl = "false", 0.0, 0.0, 0.0
        else:
            spec = TYPE_CATALOG[code]
            enabled, cval, esr, esl = "true", spec["C"], spec["ESR"], spec["ESL"]
        lines.append(f'{indent}<Edit Table="Component" Name="{name}" Column="Name" Value="{enabled}" />')
        lines.append(f'{indent}<Edit Table="Component" Name="{name}" Column="Value" Value="{_fmt(cval)}"/>')
        lines.append(f'{indent}<Edit Table="PI Decap" Name="{name}" Column="ESR" Value="{_fmt(esr)}"/>')
        lines.append(f'{indent}<Edit Table="PI Decap" Name="{name}" Column="ESL" Value="{_fmt(esl)}"/>')
        lines.append("")
    return "\n".join(lines)


def _distribution_group(type_row: np.ndarray, freq: str) -> str:
    return ("    <Group>\n"
            f"{build_component_edits(type_row)}\n"
            '      <Loop Analysis="PI-Distribution" Count="1">\n'
            f'        <EditPIDistribution Frequency="{freq}"/>\n'
            "      </Loop>\n"
            "    </Group>")


def _spectrum_group(type_row: np.ndarray, components: str) -> str:
    return ("    <Group>\n"
            f"{build_component_edits(type_row)}\n"
            f'      <CreatePISpectrum Component="{components}" Format="ARV,CSV" />\n'
            "    </Group>")


def _write_batch(groups: list[str], output_path: Path, powerbus: str) -> Path:
    body = "\n".join(groups)
    content = (f'<Batch Analysis="PI" Version="2" PowerBus="{powerbus}">\n'
               "  <Iterate>\n"
               f"{body}\n"
               "  </Iterate>\n"
               "</Batch>\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated {output_path}  ({len(groups)} group(s))")
    return output_path


def _check_codes(type_codes: np.ndarray) -> np.ndarray:
    type_codes = np.asarray(type_codes)
    if type_codes.ndim == 1:
        type_codes = type_codes.reshape(1, -1)
    if type_codes.shape[1] != N_DECAPS:
        raise ValueError(f"expected (*, {N_DECAPS}), got {type_codes.shape}")
    bad = set(np.unique(type_codes).tolist()) - VALID_CODES
    if bad:
        raise ValueError(f"type_codes contain unknown codes {sorted(bad)}; valid={sorted(VALID_CODES)}")
    return type_codes


def build_impedance_peb(type_codes: np.ndarray, output_path: Path, *, powerbus: str, components: str) -> Path:
    """One spectrum group per placement row."""
    type_codes = _check_codes(type_codes)
    return _write_batch([_spectrum_group(row, components) for row in type_codes], Path(output_path), powerbus)


def build_distribution_peb(
    type_codes: np.ndarray,
    output_path: Path,
    *,
    mhz: float | None = None,
    per_sample_mhz: Sequence[float] | None = None,
    powerbus: str,
) -> Path:
    """One map group per placement row: ``mhz`` for all rows, or ``per_sample_mhz`` per row."""
    type_codes = _check_codes(type_codes)
    if (mhz is None) == (per_sample_mhz is None):
        raise ValueError("pass exactly one of mhz= or per_sample_mhz=")
    if per_sample_mhz is None:
        freqs = [mhz_to_freq_hz(mhz)] * type_codes.shape[0]
    else:
        if len(per_sample_mhz) != type_codes.shape[0]:
            raise ValueError(f"per_sample_mhz length {len(per_sample_mhz)} != rows {type_codes.shape[0]}")
        freqs = [mhz_to_freq_hz(f) for f in per_sample_mhz]
    return _write_batch([_distribution_group(row, f) for row, f in zip(type_codes, freqs)],
                        Path(output_path), powerbus)
