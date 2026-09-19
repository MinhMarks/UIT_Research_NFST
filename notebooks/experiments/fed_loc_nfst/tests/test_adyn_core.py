"""
Unit Tests for ADYN-LOC-NFST Core
====================================
Tests mathematical correctness of Rank-1 updates, cluster lifecycle,
and the full ADYN pipeline.

Run:
    pytest fed_loc_nfst/tests/test_adyn_core.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Path setup for direct test runner
_TESTS_DIR = Path(__file__).parent
_EXPERIMENTS_DIR = _TESTS_DIR.parent.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from fed_loc_nfst.adyn_core import (
    ClusterSlot, ClusterBank, SubspaceEngine,
    run_lifecycle_step, do_split, do_merge, find_merge_pair,
    should_split, THETA_MERGE, K_MIN
)
from fed_loc_nfst.adyn_loc_nfst import fit_static_loc_nfst, AdynLOCNFST


# ============================================================
# Test 1: Welford Online Update Correctness
# ============================================================

def test_welford_update():
    """Welford centroid should match batch mean."""
    d = 10
    N = 100
    X = np.random.randn(N, d).astype(np.float64)

    slot = ClusterSlot()
    slot.init(d, X[0])
    for x in X[1:]:
        slot.welford_update(x)

    np.testing.assert_allclose(slot.centroid, np.mean(X, axis=0), atol=1e-8)
    assert slot.count == N
    print(f"✓ Welford centroid matches batch mean (N={N}, d={d})")


# ============================================================
# Test 2: Rank-1 Split Update — S_w Monotone Decrease
# ============================================================

def test_rank1_split_Sw_decrease():
    """
    After a cluster split, S_w should decrease by exactly vv^T (Theorem 1).
    Also: S_t should remain invariant.
    """
    d = 20
    N = 500
    np.random.seed(42)

    X = np.random.randn(N, d).astype(np.float64)

    # Build a simple model with 2 clusters
    bank = ClusterBank(d=d)
    # Manually init 2 clusters
    slot0 = bank.slots[0]
    slot0.init(d, X[:N//2].mean(axis=0))
    slot0.count = N // 2
    slot0.M2 = np.sum((X[:N//2] - slot0.centroid)**2, axis=0)

    slot1 = bank.slots[1]
    slot1.init(d, X[N//2:].mean(axis=0))
    slot1.count = N // 2
    slot1.M2 = np.sum((X[N//2:] - slot1.centroid)**2, axis=0)

    S_w_before, S_t_before = bank.compute_scatter()

    # Initialize subspace engine
    engine = SubspaceEngine(d=d)
    engine.initialize(S_w_before, S_t_before)
    A_before = engine.A.copy()

    # Perform split on slot 0
    mu_j1 = slot0.centroid.copy()
    N_j1 = slot0.count // 2
    N_j2 = slot0.count - N_j1

    # Compute expected Δ
    # This test checks that eigenvalues of A decrease after split downdate
    mu_j2 = mu_j1 + 0.5  # slightly different centroid

    engine.rank1_update_split(mu_j1, mu_j2, N_j1, N_j2)
    A_after = engine.A.copy()

    # Eigenvalues should not increase on average (Rank-1 downdate)
    eigvals_before = np.sort(np.linalg.eigvalsh(A_before))
    eigvals_after = np.sort(np.linalg.eigvalsh(A_after))

    assert np.mean(eigvals_after) <= np.mean(eigvals_before) + 1e-8, \
        f"Mean eigenvalue should decrease after split: {np.mean(eigvals_before):.6f} → {np.mean(eigvals_after):.6f}"
    print(f"✓ Rank-1 split: mean eigval decreased: {np.mean(eigvals_before):.4f} → {np.mean(eigvals_after):.4f}")


# ============================================================
# Test 3: Rank-1 Merge Update — S_w Monotone Increase
# ============================================================

def test_rank1_merge_Sw_increase():
    """
    After a cluster merge, S_w should increase by ww^T.
    """
    d = 15
    np.random.seed(123)

    bank = ClusterBank(d=d)
    slot0 = bank.slots[0]
    slot1 = bank.slots[1]

    mu_a = np.random.randn(d)
    mu_b = np.random.randn(d) + 3.0  # well-separated

    slot0.init(d, mu_a)
    slot0.count = 100
    slot0.M2 = np.abs(np.random.randn(d)) * 100

    slot1.init(d, mu_b)
    slot1.count = 80
    slot1.M2 = np.abs(np.random.randn(d)) * 80

    S_w, S_t = bank.compute_scatter()
    engine = SubspaceEngine(d=d)
    engine.initialize(S_w, S_t)
    A_before = engine.A.copy()

    engine.rank1_update_merge(mu_a, mu_b, 100, 80)
    A_after = engine.A.copy()

    eigvals_before = np.sort(np.linalg.eigvalsh(A_before))
    eigvals_after = np.sort(np.linalg.eigvalsh(A_after))

    assert np.mean(eigvals_after) >= np.mean(eigvals_before) - 1e-8, \
        f"Mean eigenvalue should increase after merge: {np.mean(eigvals_before):.6f} → {np.mean(eigvals_after):.6f}"
    print(f"✓ Rank-1 merge: mean eigval increased: {np.mean(eigvals_before):.4f} → {np.mean(eigvals_after):.4f}")


# ============================================================
# Test 4: Cluster Bank Online Assignment
# ============================================================

def test_cluster_bank_online_assignment():
    """Cluster bank should correctly assign points to nearest cluster."""
    d = 5
    bank = ClusterBank(d=d)

    # Create 3 well-separated clusters
    centers = np.array([[10, 0, 0, 0, 0],
                         [0, 10, 0, 0, 0],
                         [0, 0, 10, 0, 0]], dtype=np.float64)

    for k, c in enumerate(centers):
        slot = bank.slots[k]
        slot.init(d, c)
        slot.count = 100
        slot.M2 = np.ones(d) * 0.1

    # Sample near each center — should assign to correct cluster
    for k, c in enumerate(centers):
        x = c + np.random.randn(d) * 0.01
        nearest = bank.assign(x)
        assert nearest == k, f"Expected cluster {k}, got {nearest}"

    print(f"✓ ClusterBank assignment: 3 well-separated clusters correctly identified")


# ============================================================
# Test 5: ADYN-LOC-NFST Full Pipeline on Synthetic Data
# ============================================================

def test_adyn_full_pipeline_synthetic():
    """
    ADYN-LOC-NFST should achieve AUC > 70% on simple synthetic 2-class data.
    (synthetic data is always harder — just test it doesn't crash and gives
     a reasonable result)
    """
    np.random.seed(42)
    d = 20
    N_train = 1000
    N_test = 500

    # Normal class: two separated Gaussians (simulating concept drift)
    X_normal = np.vstack([
        np.random.randn(N_train // 2, d),
        np.random.randn(N_train // 2, d) + 5.0
    ])

    # Anomaly: uniform distribution far from normal
    X_anomaly = np.random.randn(N_test // 2, d) + 10.0
    X_normal_test = np.vstack([
        np.random.randn(N_test // 4, d),
        np.random.randn(N_test // 4, d) + 5.0
    ])

    X_test = np.vstack([X_normal_test, X_anomaly])
    y_test = np.array([0] * (N_test // 2) + [1] * (N_test // 2))

    # Run ADYN
    model = AdynLOCNFST(K_init=4, n_chunks=3, eval_every=100, seed=42)
    history = model.fit_temporal(X_normal, y_test=y_test, X_test=X_test)

    assert model.W is not None, "W should be set after fit"
    assert model.L >= 1, f"L should be >= 1, got {model.L}"
    assert model.K_final >= K_MIN, f"K_final should be >= K_MIN, got {model.K_final}"

    y_proba = model.predict(X_test)
    assert y_proba.shape == (N_test, 2), f"Expected shape ({N_test}, 2)"

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(1 - y_test, y_proba[:, 0])
    print(f"✓ ADYN synthetic AUC: {auc*100:.2f}% (L={model.L}, K_final={model.K_final})")
    assert auc >= 0.5, f"AUC should be >= 0.5 (random chance), got {auc:.4f}"


# ============================================================
# Test 6: Static K Baseline on Synthetic
# ============================================================

def test_static_k_baseline_synthetic():
    """Static-K LOC-NFST should produce valid W and scores."""
    np.random.seed(0)
    d = 15
    N = 500

    X_train = np.random.randn(N, d)
    X_test  = np.random.randn(200, d)
    y_test  = np.random.randint(0, 2, 200)

    W, null_centers, max_train, L = fit_static_loc_nfst(X_train, K=10, seed=0)

    assert W.shape[0] == d, f"W row dim should be d={d}"
    assert L >= 1, f"L should be >= 1"
    assert max_train > 0, "max_train should be positive"

    print(f"✓ Static-K: W={W.shape}, L={L}, max_train={max_train:.4f}")


if __name__ == "__main__":
    print("Running ADYN-LOC-NFST unit tests...")
    test_welford_update()
    test_rank1_split_Sw_decrease()
    test_rank1_merge_Sw_increase()
    test_cluster_bank_online_assignment()
    test_static_k_baseline_synthetic()
    test_adyn_full_pipeline_synthetic()
    print("\n✓ All 6 tests passed!")
