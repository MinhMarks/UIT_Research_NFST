"""
Unit tests for Standardized 3-Tier Baseline Hierarchy (Milestone M2).

Covers:
- Tier 1: NaiveFedLunar (Standard FedAvg on LUNAR with uncoordinated perturbation)
- Tier 2: FedAutoEncoder (Federated Deep Autoencoder with MSE loss)
- Tier 2: FedProxLunar (Federated LUNAR with proximal regularization)
- Tier 2: PCGradFedLunar (Federated LUNAR with standard PCGrad aggregation)
- Tier 3: LOC_NFST_Bound (LOC-NFST Closed-Form Null-Space Theoretical Bound)
"""

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fed_lunar.baselines import (
    NaiveFedLunar,
    FedAutoEncoder,
    FedProxLunar,
    PCGradFedLunar,
    LOC_NFST_Bound,
)


@pytest.fixture
def synthetic_multiclient_data():
    """Create synthetic non-IID 3-client normal dataset with test anomalies."""
    np.random.seed(42)
    torch.manual_seed(42)

    D = 8
    N_per_client = 60

    # 3 distinct normal sub-manifolds
    c1 = np.random.randn(N_per_client, D).astype(np.float32) * 0.2 + np.array([2.0] * D)
    c2 = np.random.randn(N_per_client, D).astype(np.float32) * 0.2 + np.array([-2.0] * D)
    c3 = np.random.randn(N_per_client, D).astype(np.float32) * 0.2 + np.array([0.0] * D)

    client_train = [c1, c2, c3]

    # Test set: normal test points + anomalies far from normal clusters
    test_norm1 = np.random.randn(20, D).astype(np.float32) * 0.2 + np.array([2.0] * D)
    test_norm2 = np.random.randn(20, D).astype(np.float32) * 0.2 + np.array([-2.0] * D)
    test_anom = np.random.uniform(low=-8.0, high=8.0, size=(40, D)).astype(np.float32)

    X_test = np.vstack([test_norm1, test_norm2, test_anom])
    y_test = np.array([0] * 40 + [1] * 40)

    return client_train, X_test, y_test


def test_naive_fed_lunar(synthetic_multiclient_data):
    """Test Tier 1 NaiveFedLunar baseline training and inference."""
    client_train, X_test, y_test = synthetic_multiclient_data

    model = NaiveFedLunar(
        k=5,
        hidden_dims=[32, 16],
        dropout=0.0,
        negative_ratio=1.0,
        sigma_pert=0.2,
        lr=0.005,
        batch_size=32,
        local_epochs=2,
        device="cpu",
        seed=42,
    )

    # Calling decision_function before fit should raise RuntimeError
    with pytest.raises(RuntimeError):
        model.decision_function(X_test)

    # Fit for 2 rounds
    model.fit(client_train, rounds=2, verbose=False)
    assert len(model.history) == 2

    # Decision function test
    scores = model.decision_function(X_test)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X_test),)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    # Predict proba test
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)

    # Predict binary labels
    preds = model.predict(X_test, threshold=0.5)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    # Detection performance check
    auc = roc_auc_score(y_test, scores)
    assert auc > 0.65, f"Naive Fed-LUNAR AUC {auc:.4f} should be better than random"


def test_fed_autoencoder(synthetic_multiclient_data):
    """Test Tier 2 FedAutoEncoder baseline training and inference."""
    client_train, X_test, y_test = synthetic_multiclient_data

    model = FedAutoEncoder(
        code_size=4,
        hidden1=16,
        hidden2=8,
        lr=0.01,
        batch_size=32,
        local_epochs=2,
        device="cpu",
        seed=42,
    )

    with pytest.raises(RuntimeError):
        model.decision_function(X_test)

    model.fit(client_train, rounds=2, verbose=False)
    assert len(model.history) == 2

    scores = model.decision_function(X_test)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X_test),)
    assert np.all(scores >= 0.0), "Reconstruction error must be non-negative"

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    # Autoencoder should separate extreme outliers from normal clusters
    auc = roc_auc_score(y_test, scores)
    assert auc > 0.70, f"Fed-AE AUC {auc:.4f} should separate anomalies"


def test_fedprox_lunar(synthetic_multiclient_data):
    """Test Tier 2 FedProxLunar baseline with proximal regularization."""
    client_train, X_test, y_test = synthetic_multiclient_data

    model = FedProxLunar(
        k=5,
        mu=0.05,
        hidden_dims=[32, 16],
        dropout=0.0,
        sigma_pert=0.2,
        lr=0.005,
        batch_size=32,
        local_epochs=2,
        device="cpu",
        seed=42,
    )

    model.fit(client_train, rounds=2, verbose=False)
    assert len(model.history) == 2

    scores = model.decision_function(X_test)
    assert scores.shape == (len(X_test),)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)

    auc = roc_auc_score(y_test, scores)
    assert auc > 0.65, f"FedProx-LUNAR AUC {auc:.4f} should be better than random"


def test_pcgrad_fed_lunar(synthetic_multiclient_data):
    """Test Tier 2 PCGradFedLunar baseline with standard PCGrad aggregation."""
    client_train, X_test, y_test = synthetic_multiclient_data

    model = PCGradFedLunar(
        k=5,
        hidden_dims=[32, 16],
        dropout=0.0,
        sigma_pert=0.2,
        lr=0.005,
        batch_size=32,
        local_epochs=2,
        device="cpu",
        seed=42,
    )

    model.fit(client_train, rounds=2, verbose=False)
    assert len(model.history) == 2
    assert "aligned_gradient_norm" in model.history[0]

    scores = model.decision_function(X_test)
    assert scores.shape == (len(X_test),)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)

    auc = roc_auc_score(y_test, scores)
    assert auc > 0.65, f"PCGrad-LUNAR AUC {auc:.4f} should be better than random"


def test_loc_nfst_bound(synthetic_multiclient_data):
    """Test Tier 3 LOC_NFST_Bound closed-form null-space analytical baseline."""
    client_train, X_test, y_test = synthetic_multiclient_data

    model = LOC_NFST_Bound(n_clusters=3, L_min=3, seed=42)

    with pytest.raises(RuntimeError):
        model.decision_function(X_test)

    # Fit closed-form baseline (T=1 analytical solve)
    model.fit(client_train, verbose=False)

    assert model.W is not None
    assert model.P_N is not None
    assert model.L >= 3
    D = client_train[0].shape[1]
    assert model.W.shape == (D, model.L)
    assert model.P_N.shape == (D, D)

    # Check that columns of W are orthonormal: W^T W = I_L
    WtW = model.W.T @ model.W
    np.testing.assert_allclose(WtW, np.eye(model.L), atol=1e-4)

    scores = model.decision_function(X_test)
    assert scores.shape == (len(X_test),)
    assert np.all(scores >= 0.0), "Null-space projection squared norms must be non-negative"

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)

    # Analytical upper bound should yield strong anomaly separation (high AUC-ROC)
    auc = roc_auc_score(y_test, scores)
    assert auc > 0.85, f"LOC-NFST upper bound AUC {auc:.4f} should be high"


def test_baselines_flexible_inputs():
    """Verify all baselines support dict, list, and single ndarray inputs."""
    np.random.seed(42)
    X1 = np.random.randn(30, 4).astype(np.float32)
    X2 = np.random.randn(30, 4).astype(np.float32)
    X_test = np.random.randn(10, 4).astype(np.float32)

    # 1. Dict input
    dict_data = {"client_A": X1, "client_B": X2}
    ae = FedAutoEncoder(code_size=2, local_epochs=1, device="cpu").fit(dict_data, rounds=1)
    assert ae.decision_function(X_test).shape == (10,)

    # 2. Single ndarray input
    single_data = np.vstack([X1, X2])
    loc = LOC_NFST_Bound(n_clusters=2, L_min=2).fit(single_data)
    assert loc.decision_function(X_test).shape == (10,)

    naive = NaiveFedLunar(k=3, local_epochs=1, device="cpu").fit(single_data, rounds=1)
    assert naive.decision_function(X_test).shape == (10,)
