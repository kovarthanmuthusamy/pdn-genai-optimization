"""Generate ECADStar Batch PEB for multi-type decap layouts.

Each slot carries an integer *type code* (not just 0/1):

    0 = empty
    1 = type-1 decap
    2 = type-2 decap
    ...

For every component C{i}, four Edit lines are written (matching
``configs/Batch_Decap_example.peb``)::

    <Edit Table="Component" Name="Ci" Column="Name"  Value="true|false"/>
    <Edit Table="Component" Name="Ci" Column="Value" Value="{C in Farads}"/>
    <Edit Table="PI Decap"  Name="Ci" Column="ESR"   Value="{ESR in Ohms}"/>
    <Edit Table="PI Decap"  Name="Ci" Column="ESL"   Value="{ESL in Henries}"/>

Empty slots write ``false`` + ``0`` for Value/ESR/ESL.

The type → (C, ESR, ESL) mapping lives in ``TYPE_CATALOG`` (SI units).
"""
from __future__ import annotations

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH = "type_codes.npy"          # (N x 52) or (52,) int codes in {0,1,2,...}
OUTPUT_PATH = "new_multitype.peb"
POWERBUS = "Power_GND"
FREQ = "63e6"                          # PI-Distribution analysis frequency (Hz)
COMPONENTS = "IC1_Port1,IC2_Port2"     # IC ports for CreatePISpectrum export
# ──────────────────────────────────────────────────────────────────────────────

N_COMPONENTS = 52  # C1 .. C52

# Type catalog in SI units (Farads, Ohms, Henries).
#   Type1: 100 nF, ESL 222 pH, ESR 8.9 mΩ
#   Type2:  47 nF, ESL 154 pH, ESR 21.4 mΩ
TYPE_CATALOG: dict[int, dict[str, float]] = {
    1: {"C": 100e-9, "ESR": 8.9e-3, "ESL": 222e-12},
    2: {"C": 47e-9, "ESR": 21.4e-3, "ESL": 154e-12},
}


def _fmt(v: float) -> str:
    """Format a numeric value ECADStar-style (e.g. 1e-07, 0.0089, 0)."""
    if v == 0:
        return "0"
    return f"{v:g}"


def catalog_for_type(type_id: int) -> dict[str, float]:
    if type_id not in TYPE_CATALOG:
        raise KeyError(
            f"type_id {type_id} not in TYPE_CATALOG (known: {sorted(TYPE_CATALOG)})"
        )
    return TYPE_CATALOG[type_id]


def build_component_edits(type_row: np.ndarray, indent: str = "      ") -> str:
    """Return the 4-line Edit block for all 52 components from a type-code row."""
    lines: list[str] = []
    for idx, code in enumerate(type_row):
        code = int(code)
        name = f"C{idx + 1}"
        if code == 0:
            enabled, cval, esr, esl = "false", 0.0, 0.0, 0.0
        else:
            spec = catalog_for_type(code)
            enabled, cval, esr, esl = "true", spec["C"], spec["ESR"], spec["ESL"]
        lines.append(
            f'{indent}<Edit Table="Component" Name="{name}" Column="Name" Value="{enabled}" />'
        )
        lines.append(
            f'{indent}<Edit Table="Component" Name="{name}" Column="Value" Value="{_fmt(cval)}"/>'
        )
        lines.append(
            f'{indent}<Edit Table="PI Decap" Name="{name}" Column="ESR" Value="{_fmt(esr)}"/>'
        )
        lines.append(
            f'{indent}<Edit Table="PI Decap" Name="{name}" Column="ESL" Value="{_fmt(esl)}"/>'
        )
        lines.append("")  # blank line between components (matches example)
    return "\n".join(lines)


def build_distribution_group(type_row: np.ndarray, freq: str) -> str:
    edits = build_component_edits(type_row)
    return (
        "    <Group>\n"
        f"{edits}\n"
        f'      <Loop Analysis="PI-Distribution" Count="1">\n'
        f'        <EditPIDistribution Frequency="{freq}"/>\n'
        "      </Loop>\n"
        "    </Group>"
    )


def build_spectrum_group(type_row: np.ndarray, components: str) -> str:
    edits = build_component_edits(type_row)
    return (
        "    <Group>\n"
        f"{edits}\n"
        f'      <CreatePISpectrum Component="{components}" Format="ARV,CSV" />\n'
        "    </Group>"
    )


def generate_peb_multitype(
    type_codes: np.ndarray,
    output_path: str,
    powerbus: str = "Power_GND",
    freq: str = "63e6",
    components: str = "IC1_Port1,IC2_Port2",
    per_sample_freqs: list[str] | None = None,
    include_distribution: bool = True,
    include_spectrum: bool = True,
) -> None:
    """Write a multi-type Batch PEB.

    type_codes       : (N, 52) or (52,) integer codes in {0} ∪ TYPE_CATALOG keys.
    per_sample_freqs : optional length-N list of PI-Distribution freq strings.
    include_distribution / include_spectrum : which analysis groups to emit.
    """
    type_codes = np.asarray(type_codes)
    if type_codes.ndim == 1:
        type_codes = type_codes.reshape(1, -1)
    if type_codes.shape[1] != N_COMPONENTS:
        raise ValueError(
            f"Expected {N_COMPONENTS} components per row, got {type_codes.shape[1]}"
        )

    valid = {0, *TYPE_CATALOG.keys()}
    bad = set(np.unique(type_codes)) - valid
    if bad:
        raise ValueError(f"type_codes contain unknown codes {sorted(bad)}; valid={sorted(valid)}")

    n_samples = type_codes.shape[0]
    if per_sample_freqs is not None and len(per_sample_freqs) != n_samples:
        raise ValueError(
            f"per_sample_freqs length {len(per_sample_freqs)} != rows {n_samples}"
        )
    if not include_distribution and not include_spectrum:
        raise ValueError("At least one of include_distribution / include_spectrum must be True")

    groups: list[str] = []
    for i, row in enumerate(type_codes):
        row_freq = per_sample_freqs[i] if per_sample_freqs is not None else freq
        if include_distribution:
            groups.append(build_distribution_group(row, row_freq))
        if include_spectrum:
            groups.append(build_spectrum_group(row, components))

    body = "\n".join(groups)
    content = (
        f'<Batch Analysis="PI" Version="2" PowerBus="{powerbus}">\n'
        f"  <Iterate>\n"
        f"{body}\n"
        f"  </Iterate>\n"
        f"</Batch>\n"
    )
    with open(output_path, "w") as f:
        f.write(content)

    groups_per_sample = int(include_distribution) + int(include_spectrum)
    parts = []
    if include_distribution:
        parts.append("PI-Distribution")
    if include_spectrum:
        parts.append("PI-Spectrum")
    print(
        f"Generated {output_path}  ({n_samples} samples → "
        f"{n_samples * groups_per_sample} group(s), {', '.join(parts) or 'none'})"
    )


if __name__ == "__main__":
    codes = np.load(INPUT_PATH)
    generate_peb_multitype(
        type_codes=codes,
        output_path=OUTPUT_PATH,
        powerbus=POWERBUS,
        freq=FREQ,
        components=COMPONENTS,
    )
