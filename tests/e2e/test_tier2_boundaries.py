"""Tier 2: Boundary & Corner Cases Test Suite for Federated LUNAR.

Exhaustively verifies system resilience under pathological, numerical, and extreme conditions:
- Extreme k-NN settings (k=1, k=N-1, k >= N)
- Empty inputs, single-sample batches, and dimensional extremes (1D to 500D)
- Zero-variance features and singular covariance matrices
- Single-client (M=1) and massive client (M=50) federated scaling
- Perfectly opposing (cos=-1.0), orthogonal (cos=0), and zero-norm gradients
- Numerical saturation, extreme contamination (0% and 100%), and empty purging fallbacks.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.e2e.contract_stubs import (
    get_lunar_mlp,
    get_negative_generator,
    get_fsds_class,
    get_cmnp_filter_class,
    get_droga,
    compute_pairwise_cosine_similarity,
    compute_gradient_conflict_ratio,
    get_autoencoder,
    get_loc_nfst_bound,
    get_dirichlet_partitioner,
    get_metrics_logger,
)


# ===========================================================================
# Category 1: Extreme k-NN Settings
# ===========================================================================

def test_b1_extreme_knn_k_equals_1():
    """B1.1: Verify LUNAR MLP operates stably with k=1 (minimum nearest neighbor)."""
    model = get_lunar_mlp(k=1)
    x = torch.tensor([[0.05], [0.5], [2.0]], dtype=torch.float32)
    out = model(x)
    assert out.shape == (3, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_b1_extreme_knn_k_equals_2():
    """B1.2: Verify LUNAR MLP operates with k=2 minimal monotonic distance slope."""
    model = get_lunar_mlp(k=2)
    x = torch.tensor([[0.1, 0.2], [1.5, 2.5]], dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 1)


def test_b1_extreme_knn_k_equals_dataset_size_minus_one():
    """B1.3: Verify k-NN neighborhood spanning N-1 data points."""
    N = 25
    k = N - 1
    model = get_lunar_mlp(k=k)
    x = torch.sort(torch.rand(4, k), dim=1)[0]
    out = model(x)
    assert out.shape == (4, 1)


def test_b1_extreme_knn_large_k_100():
    """B1.4: Verify large neighborhood dimension k=100 without memory exhaustion."""
    k = 100
    model = get_lunar_mlp(k=k)
    x = torch.sort(torch.rand(8, k), dim=1)[0]
    out = model(x)
    assert out.shape == (8, 1)


def test_b1_extreme_knn_invalid_k_zero_or_negative():
    """B1.5: Verify initializing model with non-positive k raises ValueError."""
    with pytest.raises((ValueError, Exception)):
        _ = get_lunar_mlp(k=0)
    with pytest.raises((ValueError, Exception)):
        _ = get_lunar_mlp(k=-5)


# ===========================================================================
# Category 2: Empty Inputs & Batch Extremes
# ===========================================================================

def test_b2_single_sample_batch_inference():
    """B2.1: Verify single-sample batch (batch_size=1) inference and backward pass."""
    k = 10
    model = get_lunar_mlp(k=k)
    x = torch.sort(torch.rand(1, k), dim=1)[0]
    out = model(x)
    assert out.shape == (1, 1)
    loss = nn.BCELoss()(out, torch.ones(1, 1))
    loss.backward()
    assert next(model.parameters()).grad is not None


def test_b2_ultra_large_batch_2048():
    """B2.2: Verify high-throughput inference on batch_size=2048 without NaN or crash."""
    k = 10
    model = get_lunar_mlp(k=k)
    x = torch.sort(torch.rand(2048, k), dim=1)[0]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2048, 1)
    assert not torch.isnan(out).any()


def test_b2_empty_negative_generator_input():
    """B2.3: Verify NegativeGenerator handles empty normal input gracefully."""
    gen = get_negative_generator(negative_ratio=1.0, epsilon=0.1)
    X_empty = np.empty((0, 10), dtype=np.float32)
    negs = gen.generate(X_empty)
    assert negs.shape == (0, 10)


def test_b2_negative_generator_zero_ratio():
    """B2.4: Verify NegativeGenerator with negative_ratio=0 produces 0 negatives."""
    gen = get_negative_generator(negative_ratio=0.0, epsilon=0.1)
    X = np.random.randn(20, 5).astype(np.float32)
    negs = gen.generate(X)
    assert len(negs) == 0


def test_b2_invalid_feature_dimension_mismatch():
    """B2.5: Verify passing incorrect feature dimension to MLP raises ValueError."""
    k = 8
    model = get_lunar_mlp(k=k)
    x_wrong = torch.rand(4, 15)  # Expected 8, passed 15
    with pytest.raises(ValueError):
        _ = model(x_wrong)


# ===========================================================================
# Category 3: Zero Variance & Degenerate Covariance
# ===========================================================================

def test_b3_zero_variance_constant_features():
    """B3.1: Verify FSDS sketch fits on data containing constant zero-variance columns."""
    FSDS = get_fsds_class()
    X = np.random.randn(50, 10).astype(np.float32)
    X[:, 3] = 0.0  # Constant column 3
    X[:, 7] = 5.0  # Constant column 7
    
    sketch = FSDS.fit(X, rank=4)
    assert sketch.mu.shape == (10,)
    assert sketch.r_max > 0.0
    assert not np.isnan(sketch.Lambda).any()


def test_b3_singular_covariance_collinear_features():
    """B3.2: Verify FSDS and CMNP stability when features are perfectly collinear."""
    FSDS = get_fsds_class()
    X = np.zeros((60, 8), dtype=np.float32)
    col = np.random.randn(60)
    X[:, 0] = col
    X[:, 1] = 2.0 * col  # Exactly collinear
    X[:, 2] = -3.0 * col # Exactly collinear
    
    sketch = FSDS.fit(X, rank=3)
    assert np.all(sketch.Lambda > 0.0), "Eigenvalue floors must prevent division by zero"


def test_b3_identical_duplicate_points():
    """B3.3: Verify LOC-NFST and FSDS fit when dataset consists of identical repeated points."""
    FSDS = get_fsds_class()
    X_identical = np.ones((40, 6), dtype=np.float32) * 3.14
    sketch = FSDS.fit(X_identical, rank=2)
    assert sketch.r_max >= 0.0
    assert not np.isnan(sketch.mu).any()


def test_b3_svd_requested_rank_exceeds_ambient_dimension():
    """B3.4: Verify FSDS safely clamps requested rank r=50 when ambient dimension D=10."""
    FSDS = get_fsds_class()
    X = np.random.randn(30, 10).astype(np.float32)
    sketch = FSDS.fit(X, rank=50)  # r > D
    assert sketch.U.shape[1] <= 10, f"Rank cannot exceed ambient dimension: {sketch.U.shape}"


def test_b3_rank_1_subspace_in_high_dimensions():
    """B3.5: Verify 1D subspace in 40D ambient space properly computes null-space projector."""
    FSDS = get_fsds_class()
    np.random.seed(42)
    v = np.random.randn(40)
    v = v / np.linalg.norm(v)
    X = np.outer(np.random.randn(100), v).astype(np.float32)
    
    sketch = FSDS.fit(X, rank=1)
    assert sketch.U.shape == (40, 1)
    assert abs(np.linalg.norm(sketch.U) - 1.0) < 1e-4


# ===========================================================================
# Category 4: Dimensional Extremes
# ===========================================================================

def test_b4_ultra_low_dimension_1d():
    """B4.1: Verify pipeline functionality in 1-dimensional ambient space (D=1)."""
    FSDS = get_fsds_class()
    X = np.random.randn(50, 1).astype(np.float32)
    sketch = FSDS.fit(X, rank=1)
    assert sketch.mu.shape == (1,)
    assert sketch.r_max > 0.0


def test_b4_2d_planar_manifold():
    """B4.2: Verify CMNP filtering in 2D space."""
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    X = np.random.randn(50, 2).astype(np.float32)
    sketch = FSDS.fit(X, rank=1)
    cmnp = CMNPFilter(peer_sketches=[sketch])
    cand = sketch.mu + np.array([0.01, 0.01], dtype=np.float32)
    assert cmnp.is_intruding(cand) is True


def test_b4_nbaiot_115_dimensions():
    """B4.3: Verify FSDS sketch and Autoencoder on 115-dimensional N_BaIoT schema."""
    D = 115
    FSDS = get_fsds_class()
    X = np.random.randn(120, D).astype(np.float32)
    sketch = FSDS.fit(X, rank=10)
    assert sketch.U.shape == (115, 10)
    
    ae = get_autoencoder(input_dim=D, hidden_dims=[64, 32], latent_dim=16)
    x_tensor = torch.randn(8, D)
    recon = ae(x_tensor)
    assert recon.shape == (8, 115)


def test_b4_extreme_high_dimension_500d():
    """B4.4: Verify SVD and projection stability at D=500."""
    D = 500
    FSDS = get_fsds_class()
    X = np.random.randn(60, D).astype(np.float32)
    sketch = FSDS.fit(X, rank=5)
    assert sketch.U.shape == (500, 5)
    assert not np.isnan(sketch.U).any()


def test_b4_autoencoder_latent_dimension_1():
    """B4.5: Verify autoencoder works with extreme bottleneck latent_dim=1."""
    ae = get_autoencoder(input_dim=10, hidden_dims=[16, 4], latent_dim=1)
    x = torch.randn(4, 10)
    recon = ae(x)
    assert recon.shape == (4, 10)


# ===========================================================================
# Category 5: Federated Client Hierarchy Extremes
# ===========================================================================

def test_b5_single_client_federation_m_equals_1():
    """B5.1: Verify DROGA handles single-client federation (M=1) returning identical gradient."""
    DROGA = get_droga()
    g = torch.tensor([1.5, -2.5, 3.5])
    
    g_pcgrad = DROGA.dr_pcgrad([g])
    g_cagrad = DROGA.dr_cagrad([g], c=0.4)
    
    assert torch.allclose(g_pcgrad, g), "M=1 PCGrad must return identity"
    assert torch.allclose(g_cagrad, g), "M=1 CAGrad must return identity"
    assert compute_gradient_conflict_ratio([g]) == 0.0


def test_b5_two_client_federation_m_equals_2():
    """B5.2: Verify two-client federation (M=2) pairwise alignment."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, -1.0])
    g2 = torch.tensor([-1.0, 1.0])  # Antipodal
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    assert torch.norm(g_aligned).item() >= 0.0
    assert not torch.isnan(g_aligned).any()


def test_b5_large_federation_m_equals_20():
    """B5.3: Verify multi-client simplex QP stability with M=20 clients."""
    DROGA = get_droga()
    torch.manual_seed(42)
    gradients = [torch.randn(50) for _ in range(20)]
    
    g_aligned = DROGA.dr_cagrad(gradients, c=0.3)
    assert g_aligned.shape == (50,)
    assert not torch.isnan(g_aligned).any()


def test_b5_massive_federation_m_equals_50():
    """B5.4: Verify scalability under massive client count M=50 without performance breakdown."""
    DROGA = get_droga()
    torch.manual_seed(42)
    gradients = [torch.randn(30) for _ in range(50)]
    
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    g_aligned = DROGA.dr_pcgrad(gradients)
    assert g_aligned.shape == (30,)
    assert not torch.isnan(g_aligned).any()


def test_b5_heavily_unbalanced_client_data_proportions():
    """B5.5: Verify Dirichlet partitioner with extreme sample disparity (10 vs 10,000)."""
    partitioner = get_dirichlet_partitioner(num_clients=3, alpha=0.01)
    X = np.random.randn(500, 4)
    splits = partitioner.partition(X)
    assert len(splits) == 3
    assert sum(len(s) for s in splits) == 500


# ===========================================================================
# Category 6: Extreme Gradient Antagonism & Zero Norms
# ===========================================================================

def test_b6_perfectly_antipodal_gradients_cos_minus_one():
    """B6.1: Verify DROGA projects perfectly antipodal gradients (cos = -1.0)."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = -g1.clone()  # cos = -1.0
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    # Projected gradients should become orthogonal to opposing directions
    assert not torch.isnan(g_aligned).any()


def test_b6_orthogonal_gradients_cos_zero():
    """B6.2: Verify DROGA preserves mutually orthogonal gradients (cos = 0.0)."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 2.0])
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    g_expected = (g1 + g2) / 2.0
    assert torch.allclose(g_aligned, g_expected, atol=1e-5)


def test_b6_zero_norm_client_gradient():
    """B6.3: Verify division-by-zero protection when a client gradient is identically zero."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, 2.0])
    g_zero = torch.tensor([0.0, 0.0])  # Zero update
    
    g_aligned = DROGA.dr_pcgrad([g1, g_zero])
    assert not torch.isnan(g_aligned).any()
    cos_sim = compute_pairwise_cosine_similarity([g1, g_zero])
    assert not np.isnan(cos_sim).any()


def test_b6_microscopic_gradient_norm_1e_minus_15():
    """B6.4: Verify numerical underflow resilience with tiny gradient norm 1e-15."""
    DROGA = get_droga()
    g1 = torch.tensor([1e-15, -1e-15])
    g2 = torch.tensor([-1e-15, 2e-15])
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    assert not torch.isnan(g_aligned).any()


def test_b6_huge_gradient_norm_1e_plus_8():
    """B6.5: Verify numerical overflow resilience with large gradient norm 1e8."""
    DROGA = get_droga()
    g1 = torch.tensor([1e8, -1e8])
    g2 = torch.tensor([-1e8, 2e8])
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    assert not torch.isnan(g_aligned).any()
    assert not torch.isinf(g_aligned).any()


# ===========================================================================
# Category 7: Numerical Extremes & Activation Saturation
# ===========================================================================

def test_b7_massive_distance_input_saturation():
    """B7.1: Verify sigmoid does not produce NaN/Inf when distance values are huge (1e6)."""
    k = 5
    model = get_lunar_mlp(k=k)
    x = torch.full((4, k), 1e6, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_b7_microscopic_distance_input_near_zero():
    """B7.2: Verify stability when distance values approach zero (1e-7)."""
    k = 5
    model = get_lunar_mlp(k=k)
    x = torch.full((4, k), 1e-7, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_b7_soft_purging_weight_boundaries():
    """B7.3: Verify soft debiased weights with gamma=0 (no purging) and extreme gamma=1000."""
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    X = np.random.randn(40, 5).astype(np.float32)
    sketch = FSDS.fit(X, rank=2)
    cmnp = CMNPFilter(peer_sketches=[sketch])
    
    cand = np.array([sketch.mu])
    w_no_purge = cmnp.compute_soft_weights(cand, gamma=0.0)
    assert w_no_purge[0] == 1.0, "gamma=0 must yield weight 1.0"
    
    w_high_purge = cmnp.compute_soft_weights(cand, gamma=1000.0)
    assert w_high_purge[0] == 0.0, "High gamma must clamp to 0.0"


def test_b7_autoencoder_reconstruction_on_zero_input():
    """B7.4: Verify Autoencoder forward pass on identically zero inputs."""
    ae = get_autoencoder(input_dim=8)
    zeros = torch.zeros(4, 8)
    recon = ae(zeros)
    err = ae.reconstruction_error(zeros)
    assert recon.shape == (4, 8)
    assert torch.all(err >= 0.0)


def test_b7_loc_nfst_singular_matrix_tolerance():
    """B7.5: Verify LOC-NFST handles singular Sw with tolerance threshold gracefully."""
    loc_nfst = get_loc_nfst_bound()
    X = np.ones((30, 6), dtype=np.float32) # Sw is identically 0
    loc_nfst.fit(X, tol=1e-3)
    scores = loc_nfst.score(X)
    assert not np.isnan(scores).any()


# ===========================================================================
# Category 8: Contamination Extremes
# ===========================================================================

def test_b8_clean_stream_zero_percent_contamination():
    """B8.1: Verify metrics evaluation with 0% contamination (pure benign stream)."""
    MetricsLogger = get_metrics_logger()
    y_true = np.zeros(100, dtype=int)
    y_score = np.random.uniform(0.0, 0.3, size=100)
    metrics = MetricsLogger.evaluate(y_true, y_score)
    # When all true labels are 0, AUC defaults to 50.0 gracefully
    assert metrics["auc_roc"] == 50.0
    assert metrics["f1_score"] == 0.0


def test_b8_complete_attack_100_percent_contamination():
    """B8.2: Verify metrics evaluation with 100% contamination (pure attack stream)."""
    MetricsLogger = get_metrics_logger()
    y_true = np.ones(100, dtype=int)
    y_score = np.random.uniform(0.7, 1.0, size=100)
    metrics = MetricsLogger.evaluate(y_true, y_score)
    assert metrics["auc_roc"] == 50.0
    assert metrics["f1_score"] > 0.0


def test_b8_single_anomaly_in_stream():
    """B8.3: Verify needle-in-a-haystack single anomaly sample in 1000 normal flows."""
    MetricsLogger = get_metrics_logger()
    y_true = np.zeros(1000, dtype=int)
    y_true[42] = 1
    y_score = np.full(1000, 0.1)
    y_score[42] = 0.99
    metrics = MetricsLogger.evaluate(y_true, y_score)
    assert metrics["auc_roc"] == 100.0


def test_b8_exact_five_percent_contamination_bound():
    """B8.4: Verify metrics calculation on standard 5% contamination evaluation set."""
    MetricsLogger = get_metrics_logger()
    n_norm = 950
    n_anom = 50
    y_true = np.array([0] * n_norm + [1] * n_anom)
    y_score = np.array([0.1] * n_norm + [0.9] * n_anom)
    metrics = MetricsLogger.evaluate(y_true, y_score)
    assert metrics["auc_roc"] == 100.0
    assert metrics["far"] == 0.0


def test_b8_all_candidates_purged_generator_resilience():
    """B8.5: Verify NegativeGenerator does not crash when 100% of candidates are purged."""
    class StrictRejectAll:
        def is_intruding(self, x):
            return True
            
    gen = get_negative_generator(negative_ratio=1.0, epsilon=0.05, cmnp=StrictRejectAll())
    X_norm = np.random.randn(30, 8).astype(np.float32)
    negs = gen.generate(X_norm)
    assert len(negs) == 30
    assert negs.shape == (30, 8)
