"""Precompute per-decap impedance profiles analytically and save to configs/decap_profiles.npy.

═══════════════════════════════════════════════════════════════════════════════
PHYSICS BACKGROUND
═══════════════════════════════════════════════════════════════════════════════

Every physical decoupling capacitor (decap) is NOT a perfect capacitor.
It is a series RLC circuit with three parasitic elements:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ─── ESL (nH) ─── ESR (mΩ) ─── C (nF) ───                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

  C   (Capacitance)        — stores charge, lowers impedance at mid frequencies
  ESR (Equivalent Series Resistance) — internal loss, sets the minimum impedance floor
  ESL (Equivalent Series Inductance) — from bond wires, pads, vias; dominates at HF

The complex impedance of this series RLC at frequency f is:

    Z_decap(f) = ESR  +  j * ( 2π·f·ESL  −  1/(2π·f·C) )
                 ───       ──────────────   ─────────────
                 real      inductive term   capacitive term

The imaginary part is:
  - NEGATIVE at low f  → component is CAPACITIVE (X_C > X_L)
  - ZERO at f_SRF      → component is PURELY RESISTIVE (X_C = X_L)
  - POSITIVE at high f → component is INDUCTIVE  (X_L > X_C)

The Self-Resonant Frequency (SRF) is the crossover point:

    f_SRF = 1 / (2π · sqrt(ESL · C))

Below SRF  : the decap is doing its job (low |Z|, capacitive behaviour)
At SRF     : |Z| = ESR (minimum, purely resistive — best decoupling point)
Above SRF  : the decap becomes an inductor — it INCREASES impedance with frequency

For this design:
    C   = 100 nF
    ESR = 36 mΩ
    ESL = 0.45 nH
    → f_SRF ≈ 23.7 MHz

═══════════════════════════════════════════════════════════════════════════════
WHY MAGNITUDE ONLY (not complex)?
═══════════════════════════════════════════════════════════════════════════════

The full complex Z_decap(f) carries both magnitude and phase. For the VAE
input feature, we use |Z_decap(f)| (the magnitude). Reasons:

  1. The measured impedance spectra in the dataset (the ground-truth channel 0)
     are also stored as magnitudes — keeping the same representation is consistent.

  2. Neural networks operating on real-valued inputs cannot directly process
     complex numbers. Splitting Re/Im into two separate channels is possible
     but doubles the input size for marginal gain in this context.

  3. The physics information that matters most — WHERE the SRF is, how LOW the
     minimum is (=ESR), how steeply it rises on either side — is fully captured
     in the magnitude curve. Phase information is redundant given magnitude for
     a minimum-phase system like a passive RLC.

NOTE: _compute_z_eff in train_vae_simple.py then computes the effective parallel
impedance as a real-valued scalar at each frequency point:

    Y_eff(f) = sum_i [ occ_i / |Z_i(f)| ]        ← real admittance sum
    Z_eff(f) = 1 / Y_eff(f)                        ← real effective impedance

This is an approximation of the true complex parallel combination. It is exact
only when all decaps are in the same phase regime (all capacitive OR all
inductive). Near SRF, where phase varies rapidly, there is a small error.
For the purpose of the VAE encoder this is acceptable: the model is learning
a continuous latent representation, not performing exact circuit simulation.

═══════════════════════════════════════════════════════════════════════════════
HOW Z_eff IS USED IN THE MODEL (exp031+)
═══════════════════════════════════════════════════════════════════════════════

At training time (train_vae_simple.py → _compute_z_eff):

  Step 1 — admittance profiles:
      admittance_profiles[i, f] = 1 / decap_profiles[i, f]   shape: (52, 231)

  Step 2 — weighted sum by occupancy (binary vector, shape B×52):
      Y_eff[b, f] = sum_i ( occ[b, i] * admittance_profiles[i, f] )
                  = occ @ admittance_profiles                 shape: (B, 231)

  Step 3 — invert to get effective impedance:
      Z_eff[b, f] = 1 / Y_eff[b, f]                          shape: (B, 231)

  Step 4 — log + z-score normalize (same stats as channel 0):
      log_z_eff = log(Z_eff)
      norm_z_eff = (log_z_eff - imp_log_mean) / imp_log_std   shape: (B, 231)

  Step 5 — append as channel 3:
      imp_input = cat([ch0, ch1, ch2, norm_z_eff], dim=1)     shape: (B, 4, 231)

The impedance encoder first Linear expands from 3×231=693 to 4×231=924 inputs.
The DECODER is unchanged — still outputs (B, 3, 231). Z_eff is encoder-only.

By injecting Z_eff, the model no longer needs to "discover" circuit theory from
data. It can focus its capacity on learning spatial parasitics: the difference
between the ideal decap network impedance and the real PCB plane impedance.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT DOES
═══════════════════════════════════════════════════════════════════════════════

Since all 52 slots use the SAME capacitor type (C/ESR/ESL identical), every
row of decap_profiles.npy is the same curve — the analytical |Z_decap(f)|.

Output: configs/decap_profiles.npy
  shape : (52, 231)   — 52 slots × 231 frequency points
  dtype : float32
  units : Ohms (impedance magnitude)

Usage:
    python scripts/precompute_decap_profiles.py

Re-run this script whenever C, ESR, or ESL values change.
The output file must exist before launching any exp031+ training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Project root detection ────────────────────────────────────────────────────
_here = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in _here.parents if (p / "datasets").is_dir() and (p / "configs").is_dir()),
    _here.parents[1],
)

# ── Decap component parameters ────────────────────────────────────────────────
# These are the measured/datasheet values for the physical decap used on all
# 52 slots of the PCB design. Update here and re-run the script if the
# component type is changed.
#
#   C   : nominal capacitance — determines the low-frequency capacitive slope
#         and, together with ESL, sets the SRF.
#   ESR : equivalent series resistance — sets the impedance floor at SRF.
#         Lower ESR = better decoupling at the resonant frequency.
#   ESL : equivalent series inductance — from package geometry + via inductance.
#         Dominates above SRF; smaller ESL → higher SRF → wider useful BW.
C   = 100e-9   # Capacitance  [F]   100 nF   (MLCC, nominal)
ESR = 0.036    # ESR          [Ω]   36 mΩ    (datasheet spec)
ESL = 0.45e-9  # ESL          [H]   0.45 nH  (package + via estimate)

# ── Paths ─────────────────────────────────────────────────────────────────────
FREQ_PATH   = PROJECT_ROOT / "configs" / "Frequency_data_hz.npy"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "decap_profiles.npy"

# Must match the dataset impedance spectrum dimensions.
N_SLOTS = 52    # number of decap placement slots on the PCB
N_FREQ  = 231   # number of frequency points (1 MHz – 600 MHz, linear steps)


def compute_rlc_magnitude(freq: np.ndarray) -> np.ndarray:
    """
    Compute |Z_decap(f)| for a single series-RLC decap.

    Full complex formula:
        Z(f) = ESR  +  j * (X_L(f) − X_C(f))

    where:
        X_L(f) = 2π·f·ESL     [inductive reactance, grows linearly with f]
        X_C(f) = 1/(2π·f·C)   [capacitive reactance, falls with 1/f]

    Magnitude (what we store — real-valued, Ohms):
        |Z(f)| = sqrt( ESR²  +  (X_L − X_C)² )

    Key behaviour:
        f << f_SRF : X_C >> X_L  →  |Z| ≈ X_C  (capacitive, high impedance)
        f = f_SRF  : X_C = X_L   →  |Z| = ESR   (minimum, purely resistive)
        f >> f_SRF : X_L >> X_C  →  |Z| ≈ X_L   (inductive, rising impedance)

    Args:
        freq: (N,) frequency array in Hz (must be positive, non-zero)
    Returns:
        (N,) impedance magnitude in Ohms, float32
    """
    omega = 2.0 * np.pi * freq          # angular frequency [rad/s]
    x_l = omega * ESL                   # inductive reactance [Ω]; proportional to f
    x_c = 1.0 / (omega * C)            # capacitive reactance [Ω]; inversely proportional to f
    # Net reactance: negative below SRF (cap dominates), positive above SRF (ind dominates)
    x_net = x_l - x_c
    return np.sqrt(ESR**2 + x_net**2).astype(np.float32)


def build_profiles() -> np.ndarray:
    """
    Build the (52, 231) decap_profiles array.

    Since all 52 slots have identical components, every row is the same
    analytical RLC magnitude curve. The array is still shaped (52, 231)
    so that the training code (_compute_z_eff) can support heterogeneous
    components in future without any interface change — just swap in
    per-row values here.

    Returns:
        (52, 231) float32 array, units = Ohms
    """
    # Load the project frequency grid (1 MHz – 600 MHz, 231 points, linear spacing)
    freq = np.load(FREQ_PATH).flatten()
    assert len(freq) == N_FREQ, f"Expected {N_FREQ} frequency points, got {len(freq)}"

    # Compute the analytical RLC impedance magnitude for one decap
    z_single = compute_rlc_magnitude(freq)   # shape: (231,)

    # ── Sanity diagnostics ────────────────────────────────────────────────────
    # f_SRF analytical: frequency where X_L = X_C → minimum |Z|
    f_srf = 1.0 / (2.0 * np.pi * np.sqrt(ESL * C))
    # Find the closest grid point to confirm our formula matches the discrete grid
    idx_srf = np.argmin(z_single)
    print(f"Component: C={C*1e9:.1f} nF  ESR={ESR*1e3:.1f} mΩ  ESL={ESL*1e9:.2f} nH")
    print(f"SRF (analytical): {f_srf/1e6:.2f} MHz  (nearest freq grid point: {freq[idx_srf]/1e6:.2f} MHz)")
    # At SRF, |Z| should equal ESR exactly (X_L = X_C cancel). Confirm:
    print(f"|Z| at SRF  : {z_single[idx_srf]:.6f} Ω  (expected ≈ ESR = {ESR:.3f} Ω)")
    # At 1 MHz, strongly capacitive: X_C = 1/(2π·1M·100n) ≈ 1.59 Ω
    print(f"|Z| at 1 MHz: {z_single[0]:.4f} Ω  (capacitive; X_C ≈ {1/(2*np.pi*freq[0]*C):.4f} Ω)")
    # At 600 MHz, strongly inductive: X_L = 2π·600M·0.45n ≈ 1.70 Ω
    print(f"|Z| at 600 MHz: {z_single[-1]:.6f} Ω  (inductive; X_L ≈ {2*np.pi*freq[-1]*ESL:.6f} Ω)")

    # ── Replicate across all 52 slots ─────────────────────────────────────────
    # np.tile repeats the (231,) row N_SLOTS times → (52, 231).
    # If different slot types are ever introduced, replace np.tile with a loop
    # that computes compute_rlc_magnitude(freq) per-slot with its own C/ESR/ESL.
    profiles = np.tile(z_single, (N_SLOTS, 1))   # shape: (52, 231), dtype: float32
    print(f"\nProfiles shape : {profiles.shape}  dtype: {profiles.dtype}")
    print(f"Value range    : {profiles.min():.6f} Ω  ..  {profiles.max():.4f} Ω")
    return profiles


def main() -> None:
    print("=" * 60)
    print("Precomputing per-decap impedance profiles (analytical RLC)")
    print(f"  Freq file : {FREQ_PATH}")
    print(f"  Output    : {OUTPUT_PATH}")
    print("=" * 60)

    profiles = build_profiles()

    # Save as float32 .npy — loaded once at the start of each training run
    # and kept as a GPU tensor for fast batched Z_eff computation.
    np.save(OUTPUT_PATH, profiles)

    print(f"\nSaved: {OUTPUT_PATH}  shape={profiles.shape}  dtype={profiles.dtype}")
    print("Re-run this script if C, ESR, or ESL values change.")


if __name__ == "__main__":
    main()
