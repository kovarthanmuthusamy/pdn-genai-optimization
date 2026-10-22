"""Tests for the straight-through estimator top-K function (Stage 2 inverse design)."""
import torch

from pipelines.latent.optimize import _ste_topk


def test_forward_discrete():
    """Test that _ste_topk forward pass produces hard binary top-K."""
    # Create occupancy probabilities (52 slots)
    occ_prob = torch.rand(1, 52, dtype=torch.float32)
    K = 5

    # Forward pass: should return hard binary with exactly K ones
    output = _ste_topk(occ_prob, K)

    # Check: output is binary (only 0s and 1s)
    # Use allclose to handle floating-point precision (values may be very close to 0 or 1)
    rounded = torch.round(output)
    assert torch.allclose(output, rounded, atol=1e-6), "Output should be binary (0 or 1)"

    # Check: output sums to K (exactly K slots selected)
    assert torch.allclose(output.sum(dim=-1), torch.tensor(float(K))), f"Output should sum to {K}"


def test_backward_identity():
    """Test that _ste_topk backward pass has identity gradient."""
    # Create occupancy probabilities with requires_grad=True
    occ_prob = torch.rand(2, 52, dtype=torch.float32, requires_grad=True)
    K = 3

    # Forward pass
    output = _ste_topk(occ_prob, K)

    # Compute a scalar loss and backward
    loss = output.sum()
    loss.backward()

    # Backward pass (via STE): gradient should be ≈ 1 (identity)
    # The straight-through estimator should flow gradient directly to occ_prob
    assert occ_prob.grad is not None, "Gradients should flow to occ_prob"

    # Gradient should be non-zero everywhere (since identity gradient)
    # and approximately 1 per element (since loss = sum(output) and output is binary)
    expected_grad = torch.ones_like(occ_prob)
    assert torch.allclose(occ_prob.grad, expected_grad, atol=1e-5), \
        "STE should have identity gradient (≈ 1 everywhere)"
