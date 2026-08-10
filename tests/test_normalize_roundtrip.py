"""Tests for normalization round-trips (data pipeline sanity check)."""
import numpy as np
import torch

from src_vae.others.heatmap_gmax_norm import (
    physical_to_gmax_norm,
    gmax_norm_to_physical,
    zscore_to_physical,
)


def test_gmax_roundtrip():
    """Test that physical → gmax_norm → physical recovers original heatmap."""
    # Create a realistic heatmap (64x64 spatial) with impedance values
    np.random.seed(42)
    phys_original = np.random.uniform(0.1, 10.0, size=(64, 64)).astype(np.float32)

    # Normalization parameters
    global_max_ohm = 15.0  # Maximum impedance observed
    bg_ohm = 0.0            # Background (foreground starts above this)

    # Forward: physical → [0, 1] norm
    norm = physical_to_gmax_norm(phys_original, global_max_ohm=global_max_ohm, bg_ohm=bg_ohm)

    # Check: normalized values are in [0, 1]
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0), "Normalized values should be in [0, 1]"

    # Inverse: norm → physical
    phys_recovered = gmax_norm_to_physical(torch.from_numpy(norm), global_max_ohm).numpy()

    # Roundtrip: should recover original (within floating-point tolerance)
    np.testing.assert_allclose(phys_recovered, phys_original, rtol=1e-5, atol=1e-6)


def test_zscore_roundtrip():
    """Test that zscore → physical and back recovers original z-scores (for non-clamped values)."""
    # Create synthetic z-score normalized heatmap with z ∈ [-2, 3] to avoid clamp at 0
    # This represents realistic normalized impedance data (centered around mean, ±3σ)
    np.random.seed(123)
    z_original = np.random.normal(0.5, 0.8, size=(64, 64)).astype(np.float32)
    # Clip to avoid extreme values that get clamped
    z_original = np.clip(z_original, -2.0, 3.0)

    # Normalization parameters (from a typical multifreq dataset)
    log_mean = 2.5   # Log-space mean impedance
    log_std = 0.8    # Log-space std impedance

    # Forward: z-score → physical Ω
    phys = zscore_to_physical(z_original, log_mean=log_mean, log_std=log_std)

    # Check: physical values are non-negative
    assert np.all(phys >= 0.0), "Physical impedance should be non-negative"

    # Inverse: physical → z-score
    # Formula: z = (log(phys + 1) - log_mean) / log_std
    z_recovered = (np.log(phys + 1.0) - log_mean) / log_std

    # Roundtrip: should recover original z-scores (within floating-point tolerance)
    # Note: clamping in forward means we exactly recover for all non-clamped values
    np.testing.assert_allclose(z_recovered, z_original, rtol=1e-4, atol=1e-5)
