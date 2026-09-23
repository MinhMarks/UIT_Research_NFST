"""
Unit tests for Distance-Ranking Orthogonal Gradient Alignment (DROGA).

Validates:
1. Exact computation of Pairwise Cosine Similarity and Gradient Conflict Ratio (GCR).
2. DR-PCGrad pairwise gradient projection (<g_i^proj, g_j> >= 0).
3. DR-CAGrad dual simplex QP solution and monotonic descent guarantee (<g*, g_i> >= 0).
4. DROGAStrategy server aggregation across multi-client federations.
5. End-to-end client training round and gradient aggregation.

Reference:
    Explorer 3 Report (Mathematical Foundations), Section 5 & Algorithm 2.
"""

import pytest
import numpy as np
import torch

from fed_lunar.federated.strategy import (
    DROGAStrategy,
    dr_pcgrad,
    dr_cagrad,
    compute_gradient_conflict_metrics,
)
from fed_lunar.federated.client import LunarClient
from fed_lunar.models.lunar_mlp import LUNAR_MLP


def test_compute_gradient_conflict_metrics():
    """Test exact computation of cosine similarities and GCR across known geometric vectors."""
    # 1. Concordant gradients: colinear, same direction
    g_concordant = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    ]
    m_conc = compute_gradient_conflict_metrics(g_concordant)
    assert m_conc["gcr"] == 0.0
    assert m_conc["conflicting_pairs"] == 0
    assert m_conc["total_pairs"] == 1
    np.testing.assert_allclose(m_conc["mean_cosine"], 1.0, atol=1e-5)

    # 2. Antagonistic gradients: opposing directions (cosine = -1.0)
    g_opposing = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
    ]
    m_opp = compute_gradient_conflict_metrics(g_opposing)
    assert m_opp["gcr"] == 1.0
    assert m_opp["conflicting_pairs"] == 1
    assert m_opp["total_pairs"] == 1
    np.testing.assert_allclose(m_opp["mean_cosine"], -1.0, atol=1e-5)

    # 3. 4-Client mixed configuration:
    # Clients 0 & 1 along +x: [1, 0]
    # Clients 2 & 3 along -x: [-1, 0]
    # Total pairs = 4*3/2 = 6 pairs.
    # Pairs with opposing cosine: (0, 2), (0, 3), (1, 2), (1, 3) = 4 conflicting pairs.
    g_4clients = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
    ]
    m_4 = compute_gradient_conflict_metrics(g_4clients)
    assert m_4["total_pairs"] == 6
    assert m_4["conflicting_pairs"] == 4
    np.testing.assert_allclose(m_4["gcr"], 4.0 / 6.0, atol=1e-5)
    np.testing.assert_allclose(m_4["gcr_percent"], (4.0 / 6.0) * 100.0, atol=1e-5)


def test_dr_pcgrad_projection():
    """
    Test DR-PCGrad pairwise projection:
    When <g_i, g_j> < 0, g_i is projected onto the normal plane of g_j such that <g_i^proj, g_j> >= 0.
    """
    # Create two conflicting 2D gradients
    # g1 = [1.0, 0.0]
    # g2 = [-0.6, 0.8] -> <g1, g2> = -0.6 < 0
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-0.6, 0.8])

    inner_pre = torch.dot(g1, g2).item()
    assert inner_pre < 0.0, "Input gradients must be conflicting"

    # Run DR-PCGrad
    g_aligned = dr_pcgrad([g1, g2], seed=42)

    # In 2D, DR-PCGrad projects g1 onto the orthogonal complement of g2:
    # g1' = g1 - (<g1, g2>/||g2||^2) g2
    # Then <g1', g2> = 0.
    # Check that aggregated direction has non-negative inner product with both clients
    ip1 = torch.dot(g_aligned, g1).item()
    ip2 = torch.dot(g_aligned, g2).item()

    assert ip1 >= -1e-6, f"Expected non-negative inner product with g1, got {ip1}"
    assert ip2 >= -1e-6, f"Expected non-negative inner product with g2, got {ip2}"


def test_dr_cagrad_dual_simplex_qp():
    """
    Test DR-CAGrad dual simplex QP solver:
    min_alpha 1/2 ||g_0 + sum_i alpha_i g_i||^2 s.t. alpha >= 0, sum alpha_i = phi.
    Verifies that the aligned direction provides simultaneous descent for conflicting clients.
    """
    # 3 conflicting clients in 3D:
    # g0 points weakly along +z, but clients strongly pull in opposing x and y directions
    g1 = torch.tensor([2.0, 0.0, 0.5])
    g2 = torch.tensor([-2.0, 0.0, 0.5])
    g3 = torch.tensor([0.0, 1.5, 0.5])

    grads = [g1, g2, g3]
    pre_metrics = compute_gradient_conflict_metrics(grads)
    assert pre_metrics["conflicting_pairs"] > 0, "Must have conflicting pairs"

    # Solve DR-CAGrad
    g_cagrad = dr_cagrad(grads, c_param=0.5)

    # Verify that g_cagrad is a valid descent direction for all clients:
    # <g_cagrad, g_i> >= 0 for all i
    for i, g in enumerate(grads):
        ip = torch.dot(g_cagrad, g).item()
        assert ip >= -1e-5, f"Client {i} has negative inner product {ip} with aligned gradient!"


def test_droga_strategy_server_aggregation():
    """Test DROGAStrategy server interface across CAGrad, PCGrad, and FedAvg modes."""
    np.random.seed(42)
    torch.manual_seed(42)
    P = 50
    # Create 4 synthetic client gradients with engineered conflict
    g1 = torch.randn(P)
    g2 = -g1 + torch.randn(P) * 0.1  # Strongly opposing g1
    g3 = torch.randn(P)
    g4 = -g3 + torch.randn(P) * 0.1  # Strongly opposing g3
    gradients = [g1, g2, g3, g4]

    # 1. Test CAGrad strategy
    strat_cagrad = DROGAStrategy(mode="CAGrad", c_param=0.4)
    g_ca, summary_ca = strat_cagrad.aggregate(gradients, round_idx=1)

    assert g_ca.shape == (P,)
    assert summary_ca["pre_gcr"] > 0.0
    assert summary_ca["aligned_norm"] > 0.0
    assert len(strat_cagrad.history) == 1

    # 2. Test PCGrad strategy
    strat_pcgrad = DROGAStrategy(mode="PCGrad")
    g_pc, summary_pc = strat_pcgrad.aggregate(gradients, round_idx=1)

    assert g_pc.shape == (P,)
    assert summary_pc["pre_gcr"] == summary_ca["pre_gcr"]

    # 3. Test FedAvg strategy
    strat_fedavg = DROGAStrategy(mode="FedAvg")
    g_avg, summary_avg = strat_fedavg.aggregate(gradients, round_idx=1)

    assert g_avg.shape == (P,)
    assert summary_avg["pre_gcr"] == summary_ca["pre_gcr"]
    # DROGA produces valid non-zero descent direction and logs conflict diagnostics
    assert summary_ca["aligned_norm"] > 0.0
    assert summary_pc["aligned_norm"] > 0.0
    assert summary_ca["post_conflicts_with_clients"] <= summary_avg["post_conflicts_with_clients"]


def test_client_training_round_and_aggregation():
    """Test end-to-end local training round on LunarClient and server aggregation."""
    np.random.seed(42)
    torch.manual_seed(42)

    D = 8
    N = 60

    # Client 0 data: cluster around [0, ..., 0]
    data_0 = np.random.randn(N, D).astype(np.float32) * 0.2
    # Client 1 data: cluster around [4, ..., 4]
    data_1 = np.random.randn(N, D).astype(np.float32) * 0.2 + 4.0

    client_0 = LunarClient(client_id=0, data_normal=data_0, k=5, rank=3, device="cpu")
    client_1 = LunarClient(client_id=1, data_normal=data_1, k=5, rank=3, device="cpu")

    # Round 0: Sketch exchange
    sketch_0 = client_0.export_sketch()
    sketch_1 = client_1.export_sketch()

    client_0.receive_peer_sketches([sketch_1])
    client_1.receive_peer_sketches([sketch_0])

    # Round 1: Local training
    g0, metrics_0 = client_0.train_round(epochs=2, batch_size=32)
    g1, metrics_1 = client_1.train_round(epochs=2, batch_size=32)

    assert g0.dim() == 1 and g0.shape[0] > 0
    assert g1.dim() == 1 and g1.shape[0] > 0
    assert metrics_0["client_id"] == 0
    assert metrics_1["client_id"] == 1

    # Server aggregation with DROGA
    strategy = DROGAStrategy(mode="CAGrad", c_param=0.4)
    g_aligned, summary = strategy.aggregate([g0, g1], round_idx=1)

    assert g_aligned.shape == g0.shape
    assert summary["round"] == 1

    # Test evaluation
    test_normal = data_0[:10]
    test_anom = np.random.randn(10, D).astype(np.float32) * 2.0 + 10.0
    X_test = np.vstack([test_normal, test_anom])
    y_test = np.array([0] * 10 + [1] * 10)

    eval_results = client_0.evaluate(X_test, y_test)
    assert "auc_roc" in eval_results
    assert "f1_score" in eval_results
    assert "far" in eval_results
    assert eval_results["auc_roc"] >= 0.0 and eval_results["auc_roc"] <= 100.0
