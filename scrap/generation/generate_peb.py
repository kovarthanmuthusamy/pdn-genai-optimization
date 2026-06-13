"""
Generate ECADStar batch .peb file for PI analysis
from an occupancy vector input (N x 52), values 0 or 1.
"""

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH  = "occupancy.npy"          # path to input .npy file (N x 52) or (52,)
OUTPUT_PATH = "new.peb"                # output .peb file path
POWERBUS    = "Power_GND"              # PowerBus name in the design
FREQ        = "63e6"                   # PI-Distribution analysis frequency (Hz)
COMPONENTS  = "IC1_Port1,IC2_Port2"    # IC ports for CreatePISpectrum export
# ──────────────────────────────────────────────────────────────────────────────

N_COMPONENTS = 52  # C1 .. C52


def build_edit_lines(occ_row: np.ndarray, indent: str = "      ") -> str:
    """Return XML Edit lines for all 52 components based on a 0/1 occupancy row."""
    lines = []
    for idx, val in enumerate(occ_row):
        name = f"C{idx + 1}"
        value = "true" if int(val) == 1 else "false"
        lines.append(f'{indent}<Edit Table="Component" Name="{name}" Column="Name" Value="{value}"/>')
    return "\n".join(lines)


def build_distribution_group(occ_row: np.ndarray, freq: str) -> str:
    edits = build_edit_lines(occ_row)
    return (
        "    <Group>\n"
        f"{edits}\n"
        f'      <Loop Analysis="PI-Distribution" Count="1">\n'
        f'        <EditPIDistribution Frequency="{freq}"/>\n'
        "      </Loop>\n"
        "    </Group>"
    )


def build_spectrum_group(occ_row: np.ndarray, components: str) -> str:
    edits = build_edit_lines(occ_row)
    return (
        "    <Group>\n"
        f"{edits}\n"
        f'      <CreatePISpectrum Component="{components}" Format="ARV,CSV" />\n'
        "    </Group>"
    )


def generate_peb(
    occupancy: np.ndarray,
    output_path: str,
    powerbus: str = "Power_GND",
    freq: str = "63e6",
    components: str = "IC1_Port1,IC2_Port2",
    per_sample_freqs: list[str] | None = None,
    include_distribution: bool = True,
    include_spectrum: bool = True,
) -> None:
    """
    occupancy        : np.ndarray of shape (N, 52) with 0/1 values
                       or shape (52,) for a single sample
    per_sample_freqs : if given, a list of length N with the PI-Distribution
                       frequency string for each row (overrides `freq`).
    include_distribution : PI-Distribution (heatmap) groups.
    include_spectrum     : CreatePISpectrum (impedance) groups.
    """
    if occupancy.ndim == 1:
        occupancy = occupancy.reshape(1, -1)

    assert occupancy.shape[1] == N_COMPONENTS, (
        f"Expected {N_COMPONENTS} components per row, got {occupancy.shape[1]}"
    )

    n_samples = occupancy.shape[0]
    if per_sample_freqs is not None:
        assert len(per_sample_freqs) == n_samples, (
            f"per_sample_freqs length {len(per_sample_freqs)} != occupancy rows {n_samples}"
        )
    if not include_distribution and not include_spectrum:
        raise ValueError("At least one of include_distribution / include_spectrum must be True")

    groups = []

    for i, row in enumerate(occupancy):
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
    occupancy = np.load(INPUT_PATH)
    generate_peb(
        occupancy=occupancy,
        output_path=OUTPUT_PATH,
        powerbus=POWERBUS,
        freq=FREQ,
        components=COMPONENTS,
    )
