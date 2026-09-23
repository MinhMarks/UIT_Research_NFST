"""
Unit tests for LUNAR MLP, k-NN Distance Feature Extractor,
Distance-Ranking BCE Loss, and SimpleAutoEncoder.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.autoencoder import SimpleAutoEncoder


def test_lunar_mlp_init_and_shapes():
    """Test LUNAR MLP layer instantiation and output tensor shapes."""
    k = 10
    batch_size = 16
    model = LUNAR_MLP(k=k, hidden_dims=[64, 32, 16], dropout=0.1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 0, "Model should have trainable parameters"

    # Forward pass with sorted distances
    dummy_dists = torch.sort(torch.rand(batch_size, k) * 5.0, dim=1)[0]
    logits = model(dummy_dists)
    assert logits.shape == (batch_size, 1), f"Expected shape (16, 1), got {logits.shape}"

    # Anomaly probabilities in [0, 1]
    probs = model.predict_proba(dummy_dists)
    assert probs.shape == (batch_size, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Probabilities must be in [0, 1]"

    # Anomaly score alias
    scores = model.predict_score(dummy_dists)
    assert torch.equal(probs, scores)


def test_lunar_mlp_invalid_k_and_dimensions():
    """Test exception raising for invalid k and mismatched input dimensions."""
    with pytest.raises(ValueError):
        LUNAR_MLP(k=0)

    model = LUNAR_MLP(k=10)
    mismatched_input = torch.rand(8, 7)  # Expected 10, got 7
    with pytest.raises(ValueError, match="Expected input with 10 features"):
        model(mismatched_input)


def test_knn_distance_extractor_accuracy():
    """Test k-NN distance extraction accuracy against brute-force Euclidean distance."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Reference data: 50 samples in R^4
    ref_data = np.random.randn(50, 4).astype(np.float32)
    # Query data: 5 samples in R^4
    queries = np.random.randn(5, 4).astype(np.float32)

    k = 6
    extractor = KNNDistanceExtractor(ref_data, device="cpu", batch_size=16)
    extracted_dists = extractor.extract_distances(queries, k=k, is_reference_member=False)

    assert extracted_dists.shape == (5, k)

    # Verify monotonic ordering: 0 <= d_1 <= d_2 <= ... <= d_k
    for i in range(5):
        row = extracted_dists[i].numpy()
        for j in range(k - 1):
            assert row[j] <= row[j + 1] + 1e-6, f"Distances not sorted ascending at row {i}"

    # Brute-force verification for the first query
    q0 = queries[0]
    brute_dists = np.sort(np.linalg.norm(ref_data - q0, axis=1))[:k]
    extracted_q0 = extracted_dists[0].numpy()
    np.testing.assert_allclose(extracted_q0, brute_dists, rtol=1e-5, atol=1e-5)


def test_knn_distance_extractor_self_exclusion():
    """Test that querying reference members with is_reference_member=True omits self-distance (0)."""
    np.random.seed(123)
    ref_data = np.random.randn(30, 4).astype(np.float32)
    extractor = KNNDistanceExtractor(ref_data, device="cpu")

    # Query the first 5 reference members
    subset = ref_data[:5]

    # Without self-exclusion: first distance is ~0 (distance to self)
    dists_with_self = extractor.extract_distances(subset, k=3, is_reference_member=False)
    np.testing.assert_allclose(dists_with_self[:, 0].numpy(), 0.0, atol=1e-3)

    # With self-exclusion: first distance is > 0 (distance to closest distinct neighbor)
    dists_without_self = extractor.extract_distances(subset, k=3, is_reference_member=True)
    assert dists_without_self.shape == (5, 3)
    assert np.all(dists_without_self[:, 0].numpy() > 1e-4), "Self-distance was not excluded!"


def test_lunar_distance_ranking_loss():
    """Test BCE distance-ranking loss numerical stability, gradients, and directional behavior."""
    criterion = LunarDistanceRankingLoss(lambda_anom=1.0)

    # Case 1: Well-separated logits (normal logits < 0, anom logits > 0)
    logits_norm = torch.tensor([-2.0, -3.0, -1.5], requires_grad=True)
    logits_anom = torch.tensor([2.0, 3.5, 1.8], requires_grad=True)

    loss = criterion(logits_norm, logits_anom)
    assert loss.item() > 0.0
    assert torch.isfinite(loss), "Loss must be finite"

    # Backward pass
    loss.backward()
    assert logits_norm.grad is not None and torch.all(torch.isfinite(logits_norm.grad))
    assert logits_anom.grad is not None and torch.all(torch.isfinite(logits_anom.grad))

    # Normal gradient should be positive (increasing normal logit increases loss)
    assert torch.all(logits_norm.grad > 0.0)
    # Anomaly gradient should be negative (increasing anomaly logit decreases loss)
    assert torch.all(logits_anom.grad < 0.0)

    # Case 2: Soft debiased sample weights
    weights = torch.tensor([1.0, 0.5, 0.0])
    loss_weighted = criterion(logits_norm.detach(), logits_anom.detach(), weights_anom=weights)
    assert torch.isfinite(loss_weighted)


def test_lunar_mlp_training_convergence_toy():
    """Test that LUNAR MLP can be trained to distinguish normal from outlier distance vectors."""
    torch.manual_seed(42)
    k = 8
    model = LUNAR_MLP(k=k, hidden_dims=[32, 16], dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = LunarDistanceRankingLoss()

    # Synthetic normal: close distances (small values around 0.1 to 0.5)
    # Synthetic anomalies: distant points (large values around 2.0 to 5.0)
    d_norm = torch.sort(torch.rand(50, k) * 0.4 + 0.1, dim=1)[0]
    d_anom = torch.sort(torch.rand(50, k) * 2.0 + 2.0, dim=1)[0]

    initial_norm_score = model.predict_proba(d_norm).mean().item()
    initial_anom_score = model.predict_proba(d_anom).mean().item()

    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(d_norm), model(d_anom))
        loss.backward()
        optimizer.step()

    final_norm_score = model.predict_proba(d_norm).mean().item()
    final_anom_score = model.predict_proba(d_anom).mean().item()

    # Model should learn to rank anomalies higher than normal points
    assert final_anom_score > final_norm_score
    assert final_norm_score < initial_norm_score or final_anom_score > initial_anom_score


def test_simple_autoencoder():
    """Test SimpleAutoEncoder forward pass, bottleneck dimension, and reconstruction error."""
    torch.manual_seed(42)
    input_dim = 20
    code_size = 8
    batch_size = 12

    ae = SimpleAutoEncoder(input_dim=input_dim, code_size=code_size)

    x = torch.randn(batch_size, input_dim)
    reconstructed, code = ae(x)

    assert reconstructed.shape == (batch_size, input_dim)
    assert code.shape == (batch_size, code_size)

    # Encode and decode methods
    c2 = ae.encode(x)
    assert torch.equal(code, c2)
    r2 = ae.decode(c2)
    assert torch.equal(reconstructed, r2)

    # Reconstruction error
    errors = ae.reconstruction_error(x, reduction="none")
    assert errors.shape == (batch_size,)
    assert torch.all(errors >= 0.0)

    # Predict score interface
    scores = ae.predict_score(x.numpy())
    assert scores.shape == (batch_size,)
    np.testing.assert_allclose(scores, errors.numpy(), rtol=1e-5)
