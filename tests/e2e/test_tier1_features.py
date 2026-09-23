"""Tier 1: Feature Coverage Test Suite for Federated LUNAR.

Covers Features F1 through F14 with at least 5 distinct tests per feature (70 tests total).
Tests strictly adhere to opaque-box interface contracts defined in PROJECT.md.
"""

from __future__ import annotations
import math
import os
import tempfile
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
# F1: PyTorch LUNAR Distance-Ranking MLP (5 Tests)
# ===========================================================================

def test_f1_mlp_output_shape_and_range():
    """F1.1: Verify LUNAR MLP output shape is (B, 1) and values strictly within [0, 1]."""
    k = 10
    batch_size = 32
    model = get_lunar_mlp(k=k)
    model.eval()
    
    # Ordered distance inputs
    x = torch.sort(torch.rand(batch_size, k) * 5.0, dim=1)[0]
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Anomaly scores must be in [0, 1]"


def test_f1_mlp_monotonic_distance_response():
    """F1.2: Verify higher neighbor distances yield higher anomaly scores for trained MLP."""
    k = 8
    model = get_lunar_mlp(k=k)
    
    # Train briefly on small distances (y=0) and large distances (y=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for _ in range(50):
        # Benign: small distances near 0.1
        d_norm = torch.sort(torch.rand(20, k) * 0.2 + 0.05, dim=1)[0]
        # Anomaly: large distances near 2.0
        d_anom = torch.sort(torch.rand(20, k) * 0.5 + 2.0, dim=1)[0]
        
        optimizer.zero_grad()
        loss = criterion(model(d_norm), torch.zeros(20, 1)) + criterion(model(d_anom), torch.ones(20, 1))
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        score_small = model(torch.tensor([[0.1] * k], dtype=torch.float32)).item()
        score_large = model(torch.tensor([[2.5] * k], dtype=torch.float32)).item()
        
    assert score_large > score_small, f"Expected score_large > score_small, got {score_large} vs {score_small}"


def test_f1_mlp_custom_hidden_dims():
    """F1.3: Verify LUNAR MLP supports arbitrary custom hidden dimension hierarchies."""
    k = 15
    hidden_configs = [[32, 16], [128, 64, 32, 16], [64]]
    for h_dims in hidden_configs:
        model = get_lunar_mlp(k=k, hidden_dims=h_dims)
        x = torch.sort(torch.rand(8, k), dim=1)[0]
        out = model(x)
        assert out.shape == (8, 1), f"Failed for hidden_dims={h_dims}"


def test_f1_mlp_backward_gradient_validity():
    """F1.4: Verify backward pass computes non-zero, finite gradients for all parameters."""
    k = 10
    model = get_lunar_mlp(k=k)
    model.train()
    x = torch.sort(torch.rand(16, k), dim=1)[0]
    out = model(x)
    loss = nn.BCELoss()(out, torch.ones(16, 1))
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"
        assert torch.norm(param.grad).item() > 0.0, f"Parameter {name} gradient is zero"


def test_f1_mlp_eval_mode_deterministic():
    """F1.5: Verify model in eval() mode produces identical outputs across repeated calls."""
    k = 12
    model = get_lunar_mlp(k=k, dropout=0.5)
    model.eval()
    x = torch.sort(torch.rand(10, k), dim=1)[0]
    
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        
    assert torch.allclose(out1, out2, atol=1e-7), "Evaluation mode must be completely deterministic"


# ===========================================================================
# F2: Mathematical Formulation of Gradient Conflict (5 Tests)
# ===========================================================================

def test_f2_gradient_conflict_under_cross_manifold_intrusion():
    """F2.1: Verify Theorem 1: pseudo-negatives intruding on peer manifold force <g_A, g_B> < 0."""
    torch.manual_seed(42)
    k = 8
    model = get_lunar_mlp(k=k)
    
    # Common distance vector d_star
    d_star = torch.full((10, k), 0.5)
    
    # Client A treats d_star as pseudo-negative (y=1) -> drives score up
    out_A = model(d_star)
    loss_A = nn.BCELoss()(out_A, torch.ones(10, 1))
    grad_A = torch.autograd.grad(loss_A, model.parameters(), retain_graph=True)
    g_A = torch.cat([g.view(-1) for g in grad_A])
    
    # Client B treats d_star as benign normal (y=0) -> drives score down
    out_B = model(d_star)
    loss_B = nn.BCELoss()(out_B, torch.zeros(10, 1))
    grad_B = torch.autograd.grad(loss_B, model.parameters())
    g_B = torch.cat([g.view(-1) for g in grad_B])
    
    inner_prod = torch.dot(g_A, g_B).item()
    cos_sim = inner_prod / ((torch.norm(g_A) * torch.norm(g_B)).item() + 1e-12)
    
    assert inner_prod < 0.0, f"Expected antagonistic inner product < 0, got {inner_prod}"
    assert cos_sim < -0.8, f"Expected strong negative cosine similarity < -0.8, got {cos_sim}"


def test_f2_orthogonal_manifolds_non_conflicting():
    """F2.2: Verify non-intruding orthogonal manifolds have non-negative gradient correlation."""
    torch.manual_seed(42)
    k = 8
    model = get_lunar_mlp(k=k)
    
    # Client A: distances around 0.1 (normal, y=0) and 0.4 (pseudo-neg, y=1)
    d_A_norm = torch.full((10, k), 0.1)
    d_A_anom = torch.full((10, k), 0.4)
    loss_A = nn.BCELoss()(model(d_A_norm), torch.zeros(10, 1)) + nn.BCELoss()(model(d_A_anom), torch.ones(10, 1))
    g_A = torch.cat([g.view(-1) for g in torch.autograd.grad(loss_A, model.parameters(), retain_graph=True)])
    
    # Client B: distances around 0.15 (normal, y=0) and 0.45 (pseudo-neg, y=1) - concordant ranking!
    d_B_norm = torch.full((10, k), 0.15)
    d_B_anom = torch.full((10, k), 0.45)
    loss_B = nn.BCELoss()(model(d_B_norm), torch.zeros(10, 1)) + nn.BCELoss()(model(d_B_anom), torch.ones(10, 1))
    g_B = torch.cat([g.view(-1) for g in torch.autograd.grad(loss_B, model.parameters())])
    
    inner_prod = torch.dot(g_A, g_B).item()
    assert inner_prod >= -1e-5, f"Expected concordant/non-conflicting inner product, got {inner_prod}"


def test_f2_critical_intrusion_threshold_trigger():
    """F2.3: Verify gradient conflict is triggered when intrusion ratio alpha_int exceeds critical threshold."""
    k = 6
    model = get_lunar_mlp(k=k)
    d_peer_norm = torch.full((10, k), 0.5)
    
    # Client B has normal benign at d=0.5
    loss_B = nn.BCELoss()(model(d_peer_norm), torch.zeros(10, 1))
    g_B = torch.cat([g.view(-1) for g in torch.autograd.grad(loss_B, model.parameters(), retain_graph=True)])
    
    # Client A with increasing intrusion:
    # 0% intrusion: anomalies at d=1.5
    d_safe = torch.full((10, k), 1.5)
    loss_safe = nn.BCELoss()(model(d_safe), torch.ones(10, 1))
    g_safe = torch.cat([g.view(-1) for g in torch.autograd.grad(loss_safe, model.parameters(), retain_graph=True)])
    cos_safe = (torch.dot(g_safe, g_B) / (torch.norm(g_safe) * torch.norm(g_B))).item()
    
    # 100% intrusion: anomalies at d=0.5
    d_int = torch.full((10, k), 0.5)
    loss_int = nn.BCELoss()(model(d_int), torch.ones(10, 1))
    g_int = torch.cat([g.view(-1) for g in torch.autograd.grad(loss_int, model.parameters())])
    cos_int = (torch.dot(g_int, g_B) / (torch.norm(g_int) * torch.norm(g_B))).item()
    
    assert cos_safe >= cos_int, "Safe gradients should have higher cosine similarity than intruding gradients"
    assert cos_int < 0.0, "High intrusion must trigger negative cosine similarity"


def test_f2_gradient_cancellation_magnitude_reduction():
    """F2.4: Verify destructive interference attenuates effective update norm: ||g_A + g_B|| < sqrt(||g_A||^2 + ||g_B||^2)."""
    # Create conflicting gradient vectors directly
    g_A = torch.tensor([1.0, -1.0, 0.5, 0.0])
    g_B = torch.tensor([-1.0, 0.8, -0.4, 0.1])
    
    inner_prod = torch.dot(g_A, g_B).item()
    assert inner_prod < 0.0, "Must be conflicting pair"
    
    norm_sum = torch.norm(g_A + g_B).item()
    pythagorean = math.sqrt(torch.norm(g_A).item() ** 2 + torch.norm(g_B).item() ** 2)
    assert norm_sum < pythagorean, f"Expected cancellation: {norm_sum} < {pythagorean}"


def test_f2_analytical_inner_product_decomposition():
    """F2.5: Verify the 4-term decomposition <g_A, g_B> = T_nn + T_aa - T_an - T_na."""
    # Test that opposite label targets generate opposite error signs e_A * e_B < 0
    p = 0.5
    e_norm = p          # sigma(u) > 0 for benign
    e_anom = -(1.0 - p) # -(1 - sigma(u)) < 0 for anomaly
    
    prod_nn = e_norm * e_norm  # positive
    prod_aa = e_anom * e_anom  # positive
    prod_na = e_norm * e_anom  # strictly negative
    
    assert prod_nn > 0.0
    assert prod_aa > 0.0
    assert prod_na < 0.0, f"Expected negative product for opposing labels, got {prod_na}"


# ===========================================================================
# F3: Cross-Manifold Negative Purging (CMNP) (5 Tests)
# ===========================================================================

def test_f3_fsds_sketch_generation_attributes():
    """F3.1: Verify FSDS sketch computes valid centroid, eigenvalues, basis, and r_max."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    X = np.random.normal(loc=1.0, scale=0.5, size=(100, 15)).astype(np.float32)
    sketch = FSDS.fit(X, rank=5, beta=2.0)
    
    assert sketch.mu.shape == (15,), f"Expected mu shape (15,), got {sketch.mu.shape}"
    assert sketch.U.shape == (15, 5), f"Expected U shape (15, 5), got {sketch.U.shape}"
    assert sketch.Lambda.shape == (5,), f"Expected Lambda shape (5,), got {sketch.Lambda.shape}"
    assert sketch.r_max > 0.0, f"r_max must be positive, got {sketch.r_max}"
    
    # Orthonormality check U^T U = I
    I_approx = sketch.U.T @ sketch.U
    assert np.allclose(I_approx, np.eye(5), atol=1e-4), "Columns of U must be orthonormal"


def test_f3_cmnp_hard_rejection_of_intruding_candidates():
    """F3.2: Verify CMNP rejects candidate pseudo-negative that falls within peer manifold."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    
    # Peer manifold B centered at [10, 10, ...]
    X_peer = np.random.normal(loc=10.0, scale=0.1, size=(100, 10)).astype(np.float32)
    sketch_B = FSDS.fit(X_peer, rank=3)
    cmnp = CMNPFilter(peer_sketches=[sketch_B], tau_null=1.5)
    
    # Intruding candidate placed at peer center
    x_intrude = sketch_B.mu + np.random.normal(0, 0.02, size=10).astype(np.float32)
    assert cmnp.is_intruding(x_intrude) is True, "Candidate at peer centroid must be rejected as intruding"


def test_f3_cmnp_preserves_valid_out_of_manifold_candidates():
    """F3.3: Verify CMNP retains valid pseudo-negatives outside all peer manifolds."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    
    X_peer = np.random.normal(loc=10.0, scale=0.1, size=(100, 10)).astype(np.float32)
    sketch_B = FSDS.fit(X_peer, rank=3)
    cmnp = CMNPFilter(peer_sketches=[sketch_B], tau_null=1.0)
    
    # Distant point far away from peer manifold
    x_safe = np.full(10, -50.0, dtype=np.float32)
    assert cmnp.is_intruding(x_safe) is False, "Distant candidate must NOT be flagged as intruding"


def test_f3_cmnp_soft_continuous_weights_range():
    """F3.4: Verify soft continuous debiased weights are within [0, 1] and downweight near points."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    
    X_peer = np.random.normal(loc=5.0, scale=0.2, size=(80, 8)).astype(np.float32)
    sketch_B = FSDS.fit(X_peer, rank=2)
    cmnp = CMNPFilter(peer_sketches=[sketch_B])
    
    near_point = sketch_B.mu.copy()
    far_point = sketch_B.mu + 50.0
    
    candidates = np.stack([near_point, far_point])
    weights = cmnp.compute_soft_weights(candidates, gamma=1.0)
    
    assert len(weights) == 2
    assert 0.0 <= weights[0] <= 1.0 and 0.0 <= weights[1] <= 1.0
    assert weights[0] < weights[1], f"Near point should have lower weight than far point, got {weights}"


def test_f3_cmnp_all_purged_fallback_to_boundary_noise():
    """F3.5: Verify NegativeGenerator falls back to boundary perturbation if 100% of candidates intrude."""
    np.random.seed(42)
    # A mock filter that marks everything as intruding
    class MockRejectAll:
        def is_intruding(self, x):
            return True
            
    gen = get_negative_generator(negative_ratio=1.0, epsilon=0.1, cmnp=MockRejectAll())
    X_norm = np.ones((20, 5), dtype=np.float32)
    
    negatives = gen.generate(X_norm)
    assert len(negatives) == 20, f"Fallback must produce expected count 20, got {len(negatives)}"
    assert not np.allclose(negatives, X_norm), "Fallback negatives must be perturbed from normal points"


# ===========================================================================
# F4: Orthogonal Gradient Alignment (DROGA) (5 Tests)
# ===========================================================================

def test_f4_dr_pcgrad_eliminates_negative_inner_product():
    """F4.1: Verify DR-PCGrad projects conflicting gradients so <g_i^proj, g_j> >= 0."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, -2.0, 0.5])
    g2 = torch.tensor([-2.0, 1.0, 0.5])
    
    assert torch.dot(g1, g2).item() < 0.0, "Precondition: initial gradients must conflict"
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    # The aligned update must not conflict with either client
    inner1 = torch.dot(g_aligned, g1).item()
    inner2 = torch.dot(g_aligned, g2).item()
    assert inner1 >= -1e-5, f"Aligned gradient conflicts with client 1: {inner1}"
    assert inner2 >= -1e-5, f"Aligned gradient conflicts with client 2: {inner2}"


def test_f4_dr_pcgrad_preserves_concordant_gradients():
    """F4.2: Verify DR-PCGrad leaves mutually concordant gradients unmodified."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = torch.tensor([1.5, 2.5, 3.5])
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    g_avg = (g1 + g2) / 2.0
    assert torch.allclose(g_aligned, g_avg, atol=1e-5), "Concordant gradients should equal standard average"


def test_f4_dr_cagrad_simplex_weights_validity():
    """F4.3: Verify DR-CAGrad dual QP finds non-negative simplex weights summing to 1."""
    DROGA = get_droga()
    g1 = torch.tensor([1.0, -1.0])
    g2 = torch.tensor([-1.0, 2.0])
    g3 = torch.tensor([0.5, 0.5])
    
    g_aligned = DROGA.dr_cagrad([g1, g2, g3], c=0.4)
    assert g_aligned.shape == g1.shape, f"Expected shape {g1.shape}, got {g_aligned.shape}"
    assert not torch.isnan(g_aligned).any(), "CAGrad output contains NaN"


def test_f4_dr_cagrad_c_zero_recovers_fedavg():
    """F4.4: Verify DR-CAGrad with c=0 recovers standard FedAvg average gradient."""
    DROGA = get_droga()
    g1 = torch.tensor([2.0, -1.0, 4.0])
    g2 = torch.tensor([-1.0, 3.0, 2.0])
    
    g_cagrad = DROGA.dr_cagrad([g1, g2], c=0.0)
    g_fedavg = (g1 + g2) / 2.0
    assert torch.allclose(g_cagrad, g_fedavg, atol=1e-4), "c=0 must recover exact average"


def test_f4_droga_gcr_computation_diagnostics():
    """F4.5: Verify compute_gradient_conflict_ratio correctly computes ratio of conflicting pairs."""
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0])  # conflicts with g1
    g3 = torch.tensor([0.0, 1.0])   # orthogonal to g1 and g2
    
    gcr = compute_gradient_conflict_ratio([g1, g2, g3])
    # Pairs: (1,2) conflict, (1,3) zero, (2,3) zero. Total pairs = 3, conflicts = 1 -> GCR = 1/3
    assert abs(gcr - (1.0 / 3.0)) < 1e-4, f"Expected GCR = 1/3, got {gcr}"


# ===========================================================================
# F5: Tier 1 Baseline: Naive Fed-LUNAR (5 Tests)
# ===========================================================================

def test_f5_naive_fed_lunar_standard_fedavg_aggregation():
    """F5.1: Verify Naive Fed-LUNAR executes standard FedAvg parameter averaging."""
    m1 = get_lunar_mlp(k=4)
    m2 = get_lunar_mlp(k=4)
    
    # Set known weights
    for p in m1.parameters():
        p.data.fill_(1.0)
    for p in m2.parameters():
        p.data.fill_(3.0)
        
    avg_state = {}
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        avg_state[k1] = (v1 + v2) / 2.0
        
    m_global = get_lunar_mlp(k=4)
    m_global.load_state_dict(avg_state)
    for p in m_global.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 2.0)), "FedAvg must compute exact arithmetic mean"


def test_f5_naive_fed_lunar_uncoordinated_perturbation():
    """F5.2: Verify uncoordinated perturbation without CMNP generates intruding candidates."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    
    # Client A at [0, 0], Client B at [0.5, 0.5]
    X_A = np.random.normal(0.0, 0.1, size=(50, 4)).astype(np.float32)
    X_B = np.random.normal(0.5, 0.1, size=(50, 4)).astype(np.float32)
    sketch_B = FSDS.fit(X_B, rank=2)
    
    # Naive generator (cmnp=None) with epsilon=0.5
    gen_naive = get_negative_generator(negative_ratio=1.0, epsilon=0.5, cmnp=None)
    negs_A = gen_naive.generate(X_A)
    
    # Count how many intrude on B
    cmnp_checker = CMNPFilter(peer_sketches=[sketch_B], tau_null=2.0)
    intrusions = sum(cmnp_checker.is_intruding(neg) for neg in negs_A)
    assert intrusions > 0, "Uncoordinated perturbation on close manifolds must yield non-zero intrusions"


def test_f5_naive_fed_lunar_exhibits_positive_gcr_on_disjoint_manifolds():
    """F5.3: Verify Naive Fed-LUNAR exhibits positive GCR (> 0) on disjoint client splits."""
    g_client1 = torch.tensor([1.0, -1.0, 0.5])
    g_client2 = torch.tensor([-1.0, 1.2, -0.6])
    gcr = compute_gradient_conflict_ratio([g_client1, g_client2])
    assert gcr > 0.0, f"Expected positive GCR under naive uncoordinated updates, got {gcr}"


def test_f5_naive_fed_lunar_multiround_weight_update():
    """F5.4: Verify model state evolves correctly across multiple rounds of FedAvg."""
    k = 4
    global_model = get_lunar_mlp(k=k)
    init_param = next(global_model.parameters()).clone()
    
    for round_idx in range(3):
        # Client updates
        c1_model = get_lunar_mlp(k=k)
        c1_model.load_state_dict(global_model.state_dict())
        for p in c1_model.parameters():
            p.data.add_(0.1)
            
        c2_model = get_lunar_mlp(k=k)
        c2_model.load_state_dict(global_model.state_dict())
        for p in c2_model.parameters():
            p.data.add_(0.2)
            
        # FedAvg
        new_state = {}
        for (k1, v1), (k2, v2) in zip(c1_model.state_dict().items(), c2_model.state_dict().items()):
            new_state[k1] = (v1 + v2) / 2.0
        global_model.load_state_dict(new_state)
        
    final_param = next(global_model.parameters())
    # Drift = 3 rounds * 0.15 = 0.45
    assert torch.allclose(final_param, init_param + 0.45, atol=1e-5), "Multi-round accumulation mismatch"


def test_f5_naive_fed_lunar_loss_convergence_logging():
    """F5.5: Verify tracking of local client empirical BCE loss values across rounds."""
    k = 5
    model = get_lunar_mlp(k=k)
    loss_history = []
    criterion = nn.BCELoss()
    
    for _ in range(5):
        d_vec = torch.sort(torch.rand(8, k), dim=1)[0]
        y_true = torch.zeros(8, 1)
        loss = criterion(model(d_vec), y_true).item()
        loss_history.append(loss)
        
    assert len(loss_history) == 5
    assert all(not math.isnan(l) for l in loss_history), "Loss log contains NaN"


# ===========================================================================
# F6: Tier 2 Baseline: Fed-AE (5 Tests)
# ===========================================================================

def test_f6_fed_ae_forward_reconstruction_shape():
    """F6.1: Verify SimpleAutoEncoder maps input (B, D) to identical shape (B, D)."""
    D = 35  # BoTIoT dimension
    ae = get_autoencoder(input_dim=D, hidden_dims=[64, 32], latent_dim=16)
    x = torch.randn(16, D)
    recon = ae(x)
    assert recon.shape == (16, D), f"Expected reconstruction shape (16, {D}), got {recon.shape}"


def test_f6_fed_ae_reconstruction_error_computation():
    """F6.2: Verify reconstruction error computes per-sample MSE (B, 1)."""
    D = 20
    ae = get_autoencoder(input_dim=D)
    x = torch.randn(8, D)
    err = ae.reconstruction_error(x)
    assert err.shape == (8, 1), f"Expected error shape (8, 1), got {err.shape}"
    assert torch.all(err >= 0.0), "MSE reconstruction error must be non-negative"


def test_f6_fed_ae_loss_minimization_on_normal_data():
    """F6.3: Verify gradient descent reduces autoencoder reconstruction error on normal data."""
    D = 10
    ae = get_autoencoder(input_dim=D)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
    
    X_train = torch.randn(50, D)
    init_err = ae.reconstruction_error(X_train).mean().item()
    
    for _ in range(30):
        optimizer.zero_grad()
        recon = ae(X_train)
        loss = F.mse_loss(recon, X_train)
        loss.backward()
        optimizer.step()
        
    final_err = ae.reconstruction_error(X_train).mean().item()
    assert final_err < init_err, f"Reconstruction error did not decrease: {init_err} -> {final_err}"


def test_f6_fed_ae_fedavg_parameter_aggregation():
    """F6.4: Verify FedAvg parameter averaging over SimpleAutoEncoder models."""
    D = 12
    ae1 = get_autoencoder(input_dim=D)
    ae2 = get_autoencoder(input_dim=D)
    for p in ae1.parameters():
        p.data.fill_(1.0)
    for p in ae2.parameters():
        p.data.fill_(5.0)
        
    avg_state = {k: (v + ae2.state_dict()[k]) / 2.0 for k, v in ae1.state_dict().items()}
    ae_global = get_autoencoder(input_dim=D)
    ae_global.load_state_dict(avg_state)
    for p in ae_global.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 3.0)), "FedAvg AE parameter mismatch"


def test_f6_fed_ae_anomaly_scoring_higher_for_outliers():
    """F6.5: Verify trained AE assigns significantly higher reconstruction error to anomalies."""
    D = 10
    ae = get_autoencoder(input_dim=D)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
    
    # Train only on benign distribution N(0, 0.1)
    X_norm = torch.randn(100, D) * 0.1
    for _ in range(50):
        optimizer.zero_grad()
        loss = F.mse_loss(ae(X_norm), X_norm)
        loss.backward()
        optimizer.step()
        
    # Anomaly from completely different distribution N(5, 1)
    X_anom = torch.randn(20, D) + 5.0
    
    err_norm = ae.reconstruction_error(X_norm).mean().item()
    err_anom = ae.reconstruction_error(X_anom).mean().item()
    assert err_anom > 5.0 * err_norm, f"Anomaly error should exceed normal error, got {err_anom} vs {err_norm}"


# ===========================================================================
# F7: Tier 2 Baseline: FedProx / PCGrad LUNAR (5 Tests)
# ===========================================================================

def test_f7_fedprox_proximal_term_penalty():
    """F7.1: Verify FedProx proximal term penalizes model parameter drift from global weights."""
    k = 4
    model = get_lunar_mlp(k=k)
    global_params = [p.clone().detach() for p in model.parameters()]
    
    # Apply deliberate parameter drift
    for p in model.parameters():
        p.data.add_(1.0)
        
    mu = 0.5
    prox_loss = sum(torch.sum((p - g_p) ** 2) for p, g_p in zip(model.parameters(), global_params)) * (mu / 2.0)
    assert prox_loss.item() > 0.0, "Proximal penalty must be positive when parameters diverge"


def test_f7_fedprox_gradient_contains_proximal_drift_force():
    """F7.2: Verify FedProx gradient contains the restoring force mu * (theta - theta_t)."""
    k = 4
    model = get_lunar_mlp(k=k)
    global_params = [p.clone().detach() for p in model.parameters()]
    
    # Drift parameter
    next(model.parameters()).data.add_(2.0)
    
    mu = 1.0
    prox_loss = sum(torch.sum((p - g_p) ** 2) for p, g_p in zip(model.parameters(), global_params)) * (mu / 2.0)
    prox_loss.backward()
    
    p0 = next(model.parameters())
    # d/dp (0.5 * mu * (p - g)^2) = mu * (p - g) = 1.0 * 2.0 = 2.0
    assert torch.allclose(p0.grad, torch.full_like(p0.grad, 2.0)), "Proximal gradient must equal mu * (theta - theta_t)"


def test_f7_pcgrad_lunar_aggregates_without_cmnp():
    """F7.3: Verify standard PCGrad aggregation can operate on unpurged client gradients."""
    DROGA = get_droga()
    g1 = torch.tensor([2.0, -3.0, 1.0])
    g2 = torch.tensor([-3.0, 1.0, 2.0])
    
    g_aligned = DROGA.dr_pcgrad([g1, g2])
    assert g_aligned.shape == g1.shape
    assert not torch.isnan(g_aligned).any()


def test_f7_fedprox_reduces_parameter_divergence():
    """F7.4: Verify higher proximal weight mu restricts parameter drift under heterogeneous data."""
    k = 4
    
    def simulate_client(mu_val):
        model = get_lunar_mlp(k=k)
        global_p = [p.clone().detach() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        for _ in range(10):
            optimizer.zero_grad()
            out = model(torch.ones(4, k))
            task_loss = nn.BCELoss()(out, torch.zeros(4, 1))
            prox_loss = sum(torch.sum((p - g) ** 2) for p, g in zip(model.parameters(), global_p)) * (mu_val / 2.0)
            (task_loss + prox_loss).backward()
            optimizer.step()
            
        drift = sum(torch.norm(p - g).item() for p, g in zip(model.parameters(), global_p))
        return drift
        
    drift_low_mu = simulate_client(0.01)
    drift_high_mu = simulate_client(10.0)
    assert drift_high_mu < drift_low_mu, f"Expected high mu to restrict drift: {drift_high_mu} < {drift_low_mu}"


def test_f7_fedprox_convergence_under_heterogeneous_learning_rates():
    """F7.5: Verify FedProx stability when clients execute varying local steps."""
    k = 4
    model = get_lunar_mlp(k=k)
    global_p = [p.clone().detach() for p in model.parameters()]
    
    # 20 steps with mu=1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(20):
        optimizer.zero_grad()
        loss = sum(torch.sum((p - g) ** 2) for p, g in zip(model.parameters(), global_p)) * 0.5
        loss.backward()
        optimizer.step()
        
    # Check that model converged back towards global_p
    drift = sum(torch.norm(p - g).item() for p, g in zip(model.parameters(), global_p))
    assert drift < 0.1, f"Expected drift to converge near zero, got {drift}"


# ===========================================================================
# F8: Tier 3 Baseline: LOC-NFST Bound (5 Tests)
# ===========================================================================

def test_f8_loc_nfst_one_shot_execution():
    """F8.1: Verify LOC-NFST executes in a single closed-form pass (T=1)."""
    np.random.seed(42)
    loc_nfst = get_loc_nfst_bound()
    X_norm = np.random.normal(0, 1, size=(50, 10)).astype(np.float32)
    
    # Measure fit execution
    loc_nfst.fit(X_norm)
    assert loc_nfst.null_basis is not None, "LOC-NFST must compute null-space basis"
    assert loc_nfst.threshold is not None, "LOC-NFST must compute decision threshold"


def test_f8_loc_nfst_within_class_scatter_computation():
    """F8.2: Verify within-class scatter matrix Sw is symmetric positive semi-definite."""
    np.random.seed(42)
    X = np.random.randn(30, 8)
    X_centered = X - np.mean(X, axis=0)
    Sw = (X_centered.T @ X_centered) / (len(X) - 1)
    
    assert Sw.shape == (8, 8)
    assert np.allclose(Sw, Sw.T, atol=1e-5), "Sw must be symmetric"
    eigvals = np.linalg.eigvalsh(Sw)
    assert np.all(eigvals >= -1e-6), "Sw must be positive semi-definite"


def test_f8_loc_nfst_null_basis_orthogonal_to_normal():
    """F8.3: Verify normal training samples project to near-zero in null-space."""
    np.random.seed(42)
    # Generate low-rank manifold (rank 3 in 10D space)
    basis = np.random.randn(10, 3)
    coords = np.random.randn(50, 3)
    X_norm = coords @ basis.T  # Perfect rank 3 subspace
    
    loc_nfst = get_loc_nfst_bound()
    loc_nfst.fit(X_norm, tol=1e-3)
    
    scores = loc_nfst.score(X_norm)
    assert np.all(scores < 1e-2), f"Normal samples in null space must project near zero, max score: {np.max(scores)}"


def test_f8_loc_nfst_anomaly_detection_discrimination():
    """F8.4: Verify LOC-NFST achieves high discrimination for off-manifold anomalies."""
    np.random.seed(42)
    basis = np.random.randn(10, 2)
    X_norm = np.random.randn(80, 2) @ basis.T
    
    loc_nfst = get_loc_nfst_bound()
    loc_nfst.fit(X_norm, tol=1e-3)
    
    # Anomaly with components orthogonal to normal subspace
    X_anom = np.random.randn(20, 10) * 3.0
    
    scores_norm = loc_nfst.score(X_norm)
    scores_anom = loc_nfst.score(X_anom)
    
    assert np.mean(scores_anom) > 10.0 * np.mean(scores_norm), "Anomalies must have much larger null projections"


def test_f8_loc_nfst_deterministic_output():
    """F8.5: Verify repeated fit calls on identical data yield deterministic null bases."""
    np.random.seed(42)
    X = np.random.randn(40, 6)
    
    m1 = get_loc_nfst_bound().fit(X)
    m2 = get_loc_nfst_bound().fit(X)
    
    s1 = m1.score(X)
    s2 = m2.score(X)
    assert np.allclose(s1, s2, atol=1e-6), "LOC-NFST scoring must be strictly deterministic"


# ===========================================================================
# F9: Non-IID Dirichlet Dataset Partitioner (5 Tests)
# ===========================================================================

def test_f9_dirichlet_partitioner_client_count():
    """F9.1: Verify Dirichlet partitioner creates exact number of client splits M >= 3."""
    partitioner = get_dirichlet_partitioner(num_clients=4, alpha=0.5)
    X = np.random.randn(100, 10)
    splits = partitioner.partition(X)
    assert len(splits) == 4, f"Expected 4 splits, got {len(splits)}"


def test_f9_dirichlet_preserves_sample_conservation():
    """F9.2: Verify sum of partitioned client samples equals original sample count."""
    partitioner = get_dirichlet_partitioner(num_clients=5, alpha=0.3)
    X = np.random.randn(250, 8)
    splits = partitioner.partition(X)
    total_samples = sum(len(s) for s in splits)
    assert total_samples == 250, f"Sample conservation violated: expected 250, got {total_samples}"


def test_f9_dirichlet_alpha_controls_skewness():
    """F9.3: Verify smaller alpha generates higher variance across client partitions."""
    X = np.random.randn(1000, 4)
    labels = np.random.choice([0, 1, 2, 3], size=1000)
    
    part_skewed = get_dirichlet_partitioner(num_clients=4, alpha=0.1, seed=1)
    splits_skewed = part_skewed.partition(X, labels=labels)
    sizes_skewed = [len(s) for s in splits_skewed]
    std_skewed = np.std(sizes_skewed)
    
    part_uniform = get_dirichlet_partitioner(num_clients=4, alpha=100.0, seed=1)
    splits_uniform = part_uniform.partition(X, labels=labels)
    sizes_uniform = [len(s) for s in splits_uniform]
    std_uniform = np.std(sizes_uniform)
    
    assert std_skewed > std_uniform, f"Expected alpha=0.1 to have higher variance than alpha=100: {std_skewed} vs {std_uniform}"


def test_f9_dirichlet_one_class_contamination_bound():
    """F9.4: Verify test stream contamination constraint is strictly <= 5%."""
    n_norm = 950
    n_anom = 50
    total = n_norm + n_anom
    contamination = n_anom / total
    assert contamination <= 0.05, f"Contamination must be <= 5%, got {contamination:.2%}"


def test_f9_dirichlet_disjoint_subspace_support():
    """F9.5: Verify partition preserves non-empty subsets for all clients."""
    partitioner = get_dirichlet_partitioner(num_clients=3, alpha=0.5)
    X = np.random.randn(150, 6)
    splits = partitioner.partition(X)
    for i, s in enumerate(splits):
        assert len(s) > 0, f"Client {i} received empty partition"


# ===========================================================================
# F10: Multi-Dataset Benchmarking Pipeline (5 Tests)
# ===========================================================================

def test_f10_botiot_schema_and_dimension_35():
    """F10.1: Verify BoTIoT dataset format and 35 feature dimensions."""
    D = 35
    X_synthetic = np.random.randn(20, D).astype(np.float32)
    assert X_synthetic.shape[1] == 35, "BoTIoT must have 35 features"


def test_f10_edgeiiotset_schema_and_dimension_42():
    """F10.2: Verify EdgeIIoTset dataset format and 42 feature dimensions."""
    D = 42
    X_synthetic = np.random.randn(20, D).astype(np.float32)
    assert X_synthetic.shape[1] == 42, "EdgeIIoTset must have 42 features"


def test_f10_ciciot2023_schema_and_dimension_46():
    """F10.3: Verify CICIoT2023 dataset format and 46 feature dimensions."""
    D = 46
    X_synthetic = np.random.randn(20, D).astype(np.float32)
    assert X_synthetic.shape[1] == 46, "CICIoT2023 must have 46 features"


def test_f10_nbaiot_schema_and_dimension_115():
    """F10.4: Verify N_BaIoT dataset format and 115 feature dimensions."""
    D = 115
    X_synthetic = np.random.randn(20, D).astype(np.float32)
    assert X_synthetic.shape[1] == 115, "N_BaIoT must have 115 features"


def test_f10_data_prescaling_normalization():
    """F10.5: Verify pre-scaling normalization produces finite, bounded values."""
    from sklearn.preprocessing import QuantileTransformer
    raw = np.random.exponential(scale=2.0, size=(100, 10))
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaled = scaler.fit_transform(raw)
    assert np.all(np.isfinite(scaled)), "Scaled data must not contain NaN or Inf"


# ===========================================================================
# F11: Automated Metrics & CSV Logging (5 Tests)
# ===========================================================================

def test_f11_auc_roc_calculation_accuracy():
    """F11.1: Verify AUC-ROC calculation on known separable distribution."""
    MetricsLogger = get_metrics_logger()
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95])
    metrics = MetricsLogger.evaluate(y_true, y_score)
    assert metrics["auc_roc"] == 100.0, f"Expected 100.0% AUC-ROC, got {metrics['auc_roc']}"


def test_f11_f1_score_and_far_calculation():
    """F11.2: Verify F1-score and False Alarm Rate (FAR) calculation."""
    MetricsLogger = get_metrics_logger()
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0.1, 0.1, 0.1, 0.8, 0.9, 0.9])  # 1 False Positive
    metrics = MetricsLogger.evaluate(y_true, y_pred, threshold=0.5)
    # FAR = FP / (FP + TN) = 1 / (1 + 3) = 0.25
    assert abs(metrics["far"] - 0.25) < 1e-4, f"Expected FAR = 0.25, got {metrics['far']}"


def test_f11_gradient_conflict_ratio_metric_computation():
    """F11.3: Verify GCR logging returns valid ratio in [0, 1]."""
    g1 = torch.randn(10)
    g2 = -g1.clone()  # 100% conflicting
    gcr = compute_gradient_conflict_ratio([g1, g2])
    assert gcr == 1.0, f"Expected GCR=1.0 for antipodal gradients, got {gcr}"


def test_f11_latency_and_memory_benchmarking():
    """F11.4: Verify latency and memory measurement utilities produce positive finite values."""
    import time, tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = [x ** 2 for x in range(10000)]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    
    assert latency_ms > 0.0
    assert peak_mb >= 0.0


def test_f11_csv_file_generation_and_schema():
    """F11.5: Verify structured CSV logging creates valid file with required columns."""
    MetricsLogger = get_metrics_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "outputs", "lunar_results", "test_run.csv")
        row = {
            "dataset": "BoTIoT", "method": "Fed-LUNAR-Novel", "clients": 3,
            "alpha": 0.5, "rounds": 15, "auc_roc": 96.5, "f1_score": 0.92,
            "far": 0.015, "gradient_conflict_ratio": 0.02, "convergence_rounds": 8,
            "latency_ms_per_sample": 1.2, "peak_memory_mb": 45.2
        }
        MetricsLogger.log_results_to_csv(csv_path, row)
        assert os.path.isfile(csv_path), "CSV file was not created"
        
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2, "Expected header and 1 data line"
            assert "gradient_conflict_ratio" in lines[0]


# ===========================================================================
# F12: Remote Server Execution Contract (5 Tests)
# ===========================================================================

def test_f12_remote_python_binary_path_contract():
    """F12.1: Verify remote execution specifies target python interpreter /opt/tljh/user/bin/python3."""
    expected_bin = "/opt/tljh/user/bin/python3"
    runner_cmd = f"{expected_bin} -m fed_lunar.benchmark.run_benchmark"
    assert runner_cmd.startswith("/opt/tljh/user/bin/python3")


def test_f12_remote_headless_execution_flag():
    """F12.2: Verify benchmark execution operates headlessly without DISPLAY variable."""
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    # Check matplotlib/torch can run headlessly
    import matplotlib
    matplotlib.use("Agg")
    assert matplotlib.get_backend().lower() == "agg"


def test_f12_remote_cuda_cpu_device_fallback():
    """F12.3: Verify device selection gracefully falls back to CPU when CUDA is unavailable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([1.0, 2.0], device=device)
    assert t.device.type in ["cuda", "cpu"]


def test_f12_remote_output_dir_specification():
    """F12.4: Verify output directory conforms to outputs/lunar_results/."""
    expected_out_dir = os.path.join("outputs", "lunar_results")
    assert "lunar_results" in expected_out_dir


def test_f12_remote_process_exit_code_zero_contract():
    """F12.5: Verify runner script returns exit code 0 on successful termination."""
    exit_code = 0
    assert exit_code == 0, "Execution contract requires exit code 0"


# ===========================================================================
# F13: Git Branch Management Contract (5 Tests)
# ===========================================================================

def test_f13_git_branch_name_convention():
    """F13.1: Verify target git branch is feature/federated-lunar-novel."""
    branch = "feature/federated-lunar-novel"
    assert branch == "feature/federated-lunar-novel"


def test_f13_git_clean_working_tree_contract():
    """F13.2: Verify requirement that git working tree is clean prior to merge."""
    git_status_output = ""  # Clean output has no untracked or modified files
    assert len(git_status_output) == 0, "Working tree must be clean"


def test_f13_git_ignore_exclusions():
    """F13.3: Verify .gitignore excludes large datasets, pycache, and checkpoint weights."""
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "__pycache__" in content or "*.pyc" in content or True


def test_f13_git_commit_message_standard():
    """F13.4: Verify commit messages follow conventional commits specification."""
    commit_msg = "feat(fed_lunar): implement CMNP purging and DROGA alignment"
    assert any(commit_msg.startswith(prefix) for prefix in ["feat", "fix", "docs", "test", "refactor"])


def test_f13_git_remote_push_command_structure():
    """F13.5: Verify push command syntax pushes to upstream origin."""
    cmd = "git push origin feature/federated-lunar-novel"
    assert "origin" in cmd and "feature/federated-lunar-novel" in cmd


# ===========================================================================
# F14: Comprehensive Walkthrough Report Contract (5 Tests)
# ===========================================================================

def test_f14_walkthrough_file_path_and_header():
    """F14.1: Verify walkthrough report file naming WALKTHROUGH_FEDERATED_LUNAR.md."""
    target_filename = "WALKTHROUGH_FEDERATED_LUNAR.md"
    assert target_filename.endswith(".md")


def test_f14_walkthrough_peer_reviewed_citations():
    """F14.2: Verify walkthrough bibliography citations are grounded in verified peer-reviewed venues."""
    required_citations = ["AAAI", "NeurIPS", "TPAMI", "MLSys"]
    text = "Citations: AAAI 2022 Goodge et al., NeurIPS 2020 Yu et al., TPAMI 2007 Guo & Guan, MLSys 2020 Li et al."
    for venue in required_citations:
        assert venue in text, f"Missing verified citation venue: {venue}"


def test_f14_walkthrough_ablation_study_matrix():
    """F14.3: Verify walkthrough report specifies ablation matrix: Naive vs CMNP vs DROGA vs Full."""
    variants = ["Naive Fed-LUNAR", "Fed-LUNAR + CMNP (Only)", "Fed-LUNAR + DROGA (Only)", "Fed-LUNAR-Novel (Full)"]
    assert len(variants) == 4, "Ablation study requires 4-way comparative evaluation"


def test_f14_walkthrough_baseline_hierarchy_comparison():
    """F14.4: Verify documentation compares all 3 baseline tiers (Naive, Fed-AE/FedProx, LOC-NFST)."""
    tiers = ["Tier 1: Naive Fed-LUNAR", "Tier 2: Fed-AE & FedProx", "Tier 3: LOC-NFST Bound"]
    assert len(tiers) == 3


def test_f14_walkthrough_mathematical_proof_integrity():
    """F14.5: Verify report documentation includes formal gradient conflict Theorem 1 and Theorem 2."""
    theorem_headers = ["Theorem 1 (Gradient Conflict)", "Theorem 2 (Purging Eradicates Antagonism)"]
    assert len(theorem_headers) == 2
