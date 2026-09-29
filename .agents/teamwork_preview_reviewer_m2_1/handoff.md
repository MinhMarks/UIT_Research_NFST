# Milestone M2 Review & Adversarial Audit Report

**Reviewer:** Reviewer 1 (`teamwork_preview_reviewer_m2_1`)  
**Roles:** reviewer, critic  
**Target Milestone:** M2 (3-Tier Baseline Hierarchy & M1 Upstream Fixes)  
**Parent Agent:** `parent` (`37c8034b-fcb6-4906-bcf8-1f986e523ea0`)  
**Date:** 2026-09-23T08:08:00Z  
**Verdict:** **`REQUEST_CHANGES`**

---

## 1. Observation

### 1.1. Verification of Worker M2 Claims & Unit Tests
1. **Baseline Unit Test Suite Execution:**
   Command:
   ```powershell
   pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v
   ```
   Direct observation: All 24 tests passed in 20.27s:
   - `test_naive_fed_lunar` PASSED
   - `test_fed_autoencoder` PASSED
   - `test_fedprox_lunar` PASSED
   - `test_pcgrad_fed_lunar` PASSED
   - `test_loc_nfst_bound` PASSED
   - `test_baselines_flexible_inputs` PASSED
   - All 18 upstream tests for LUNAR MLP, CMNP, and DROGA PASSED.

2. **Upstream Mathematical Verification Suite:**
   Command:
   ```powershell
   pytest tests/test_m2_math_verification.py -v
   ```
   Direct observation: All 10 mathematical tests passed in 13.99s:
   - FedProx autograd exactness: $\nabla_\theta \mathcal{L}_{\text{prox}} = \nabla_\theta \mathcal{L}_{\text{ranking}} + \mu (\theta - \theta_t)$ verified to floating-point precision ($10^{-7}$).
   - FedProx parameter drift suppression verified under varying $\mu$.
   - PCGrad pairwise orthogonality verified ($\langle g_i^{\text{proj}}, g_j \rangle \ge -10^{-12}$).
   - SimpleAutoEncoder true MSE reconstruction loss and gradients verified.

3. **DROGA Scale-Disparity Resolution:**
   Executing `python tests/stress_droga_harness.py`:
   - Edge Case 3 ($\|g_1\| = 10^4 \|g_2\|$ antagonistic):
     - DR-PCGrad inner products: `[18751600.0, 1875.119]` (strictly positive).
     - DR-CAGrad inner products: `[15001141.0, 1500.081]` (strictly positive).
   - $M=3$ randomized stress trials:
     - DR-CAGrad violation rate reduced from 63.6% (pre-fix) down to 3.70%.
     - DR-CAGrad with adaptive $c$ violation rate reduced from 57.4% down to 2.70%.

---

### 1.2. Defect Observations: Master 4-Tier E2E Test Suite Failures

Executing the project's master test runner:
```powershell
python tests/run_e2e_tests.py
```
Direct observation: **10 tests failed out of 126 tests across all 4 tiers** (116 passed, 10 failed, exit code 1):

```
================================================================================
TEST SUITE EXECUTION SUMMARY TABLE
================================================================================
Tier                                | Total  | Passed | Failed | Time (s) | Status
--------------------------------------------------------------------------------
Tier 1: Feature Coverage (F1 - F14) | 70     | 66     | 4      | 19.43    | FAIL
Tier 2: Boundary & Corner Cases     | 40     | 39     | 1      | 3.57     | FAIL
Tier 3: Cross-Feature Interactions  | 11     | 7      | 4      | 1.36     | FAIL
Tier 4: Real-World Applications     | 5      | 4      | 1      | 1.92     | FAIL
--------------------------------------------------------------------------------
TOTAL / AGGREGATE                   | 126    | 116    | 10     | 26.28    | FAILED
================================================================================
```

Every single one of the 10 failures stems from `LOC_NFST_Bound` in `fed_lunar/baselines/loc_nfst_bound.py`:

1. **Failure Group 1: Missing Property `null_basis`:**
   - File: `tests/e2e/test_tier1_features.py:673`:
     ```python
     loc_nfst = get_loc_nfst_bound()
     loc_nfst.fit(X_norm)
     assert loc_nfst.null_basis is not None
     ```
   - Verbatim error:
     ```
     AttributeError: 'LOC_NFST_Bound' object has no attribute 'null_basis'
     ```
   - In `loc_nfst_bound.py`, the projection matrix is stored as `self.W` instead of exposing `self.null_basis` (or a `@property def null_basis(self): return self.W`).

2. **Failure Group 2: Unexpected Keyword Argument `tol` in `fit()`:**
   - Files:
     - `tests/e2e/test_tier1_features.py:699`
     - `tests/e2e/test_tier1_features.py:712`
     - `tests/e2e/test_tier2_boundaries.py:426`
     - `tests/e2e/test_tier3_pairwise.py:148` (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT)
   - Verbatim error:
     ```
     TypeError: LOC_NFST_Bound.fit() got an unexpected keyword argument 'tol'
     ```
   - In `loc_nfst_bound.py`, line 84:
     ```python
     def fit(
         self,
         client_train_data: Union[List[np.ndarray], Dict[Any, np.ndarray], np.ndarray],
         y_clusters: Optional[np.ndarray] = None,
         verbose: bool = False,
     ) -> "LOC_NFST_Bound":
     ```
     The method rejects `tol`, `rounds`, and generic keyword arguments (`**kwargs`).

3. **Failure Group 3: Missing Method `score(X)`:**
   - Files:
     - `tests/e2e/test_tier1_features.py:731`
     - `tests/e2e/test_tier4_applications.py:207`
   - Verbatim error:
     ```
     AttributeError: 'LOC_NFST_Bound' object has no attribute 'score'
     ```
   - In `loc_nfst_bound.py`, the method is named `decision_function(self, X)` but does not provide `score` as an alias.

4. **Failure Group 4: Missing Property `threshold`:**
   - In `tests/e2e/test_tier1_features.py:674`:
     ```python
     assert loc_nfst.threshold is not None
     ```
   - `LOC_NFST_Bound` stores the decision threshold as `self.max_train_score` and has no `threshold` property or alias.

---

### 1.3. Defect Observation: Low-Rank Manifold Mathematical Truncation
In `fed_lunar/baselines/loc_nfst_bound.py`, lines 106–167:
```python
U, s_t, _ = np.linalg.svd(P_t, full_matrices=False)
rank_Pt = int(np.sum(s_t > self.epsilon_svd))
rank_Pt = max(1, min(rank_Pt, D))
Q = U[:, :rank_Pt]  # (D, rank_Pt)
...
A = Q.T @ S_w @ Q
B = scipy.linalg.null_space(A, rcond=self.epsilon_svd)
...
self.W = np.ascontiguousarray(Q @ B, dtype=np.float64)  # (D, L)
```

Direct empirical evaluation on low-rank manifold data ($D=10, \text{intrinsic rank}=3, N=50$):
```python
basis = np.random.randn(10, 3)
coords = np.random.randn(50, 3)
X_norm = coords @ basis.T  # Exact rank 3 subspace in R^10
m = LOC_NFST_Bound().fit(X_norm)
scores = m.decision_function(X_norm)
```
- Direct observation:
  `Max score on normal samples: 67.66281 | Mean score: 16.148169`
- In contrast, the true null space of the rank-3 manifold is spanned by $U_{:, rank\_Pt:}$ (the 7 orthogonal dimensions where normal data has zero variance):
  `True null space projection score: 4.978e-30`
- Because `LOC_NFST_Bound` set `full_matrices=False` and discarded $U_{:, rank\_Pt:}$, it searched for null directions exclusively inside the range space $Q$ where the data has high variance, causing normal points to receive massive anomaly scores (67.66 vs $10^{-30}$).

---

### 1.4. Defect Observation: Input Type Fragility in `_format_client_data`
In `fed_lunar/baselines/naive_lunar.py` (lines 80–94), `fed_lunar/baselines/fed_ae.py` (lines 74–89), and `fed_lunar/baselines/fedprox_lunar.py` (lines 84–99):
```python
if isinstance(client_train_data, np.ndarray):
    return [client_train_data.astype(np.float32)]
elif isinstance(client_train_data, dict):
    return [np.asarray(data, dtype=np.float32) for data in client_train_data.values()]
elif isinstance(client_train_data, list):
    return [np.asarray(data, dtype=np.float32) for data in client_train_data]
else:
    raise TypeError(f"Unsupported client_train_data type: {type(client_train_data)}")
```
If a user or benchmark script passes a `tuple` of client datasets `(client_1, client_2, client_3)`:
- `isinstance(client_train_data, list)` is `False`.
- Crashes with `TypeError: Unsupported client_train_data type: <class 'tuple'>`.
- In contrast, `loc_nfst_bound.py` line 72 correctly uses `isinstance(client_train_data, (list, tuple))`.

---

## 2. Logic Chain

1. **Step 1 (Interface Divergence Causes System-Level Regression):**  
   - Observations in Section 1.2 demonstrate that `LOC_NFST_Bound` does not satisfy the contract stubs required by the project's existing testing infrastructure (`tests/run_e2e_tests.py`).  
   - Because `null_basis`, `threshold`, `score(X)`, and `fit(..., tol=..., rounds=...)` are missing, 10 distinct integration tests fail across all 4 tiers (Tiers 1, 2, 3, and 4).  
   - Downstream Milestones M3 and M4 depend directly on these interfaces for automated benchmarking. Approving M2 in this state would immediately block M3.

2. **Step 2 (Self-Certification Without Regression Testing):**  
   - Worker M2 authored `tests/test_baselines.py` and demonstrated that all 5 baselines pass when called according to Worker M2's new method names.  
   - However, Worker M2 did not execute `tests/run_e2e_tests.py` or verify compatibility against `contract_stubs.py`, resulting in self-certifying work that broke existing tests.

3. **Step 3 (Mathematical Flaw in Null-Space Manifold Projection):**  
   - Observation in Section 1.3 shows that for any dataset whose intrinsic dimensionality is lower than the ambient space ($d < D$), the true null space of the manifold is the orthogonal complement of the data ($U_{:, rank\_Pt:}$).  
   - By computing $W = Q @ B$ exclusively within the range space $Q$, `LOC_NFST_Bound` projects samples onto directions of high intra-cluster spread when $A$ is full-rank, producing anomaly scores of 67.66 on perfectly normal samples.  
   - Incorporating the manifold orthogonal complement $U_{:, rank\_Pt:}$ when $rank(P_t) < D$ restores the true null space property (scores $< 10^{-29}$).

4. **Step 4 (Absence of Malicious Integrity Violations):**  
   - Source code analysis confirms that Worker M2 did not hardcode results, did not create dummy facades, and implemented genuine PyTorch and NumPy optimization algorithms.  
   - The failures are technical and mathematical contract discrepancies, not ethical violations.

---

## 3. Caveats

1. **Synthetic vs. Real Dataset Rank:**  
   In benchmark datasets such as BoTIoT (35 features) or EdgeIIoTset (42 features) with $N > 10,000$ continuous features, $P_t$ is typically full rank ($rank(P_t) = D$). In that regime, $U_{:, rank\_Pt:}$ is empty, and $W = Q @ B$ via near-null relaxation operates as intended. However, whenever feature correlation or zero-variance columns induce rank deficiency, the orthogonal complement is critical.
2. **Reviewer Role Boundary:**  
   In strict accordance with the Reviewer role constraints, no changes have been committed directly to the codebase by this agent. All necessary corrections are detailed below for Worker M2 to implement.

---

## 4. Conclusion & Required Changes

**Verdict: `REQUEST_CHANGES`**

The baseline implementation is high quality, mathematically rigorous, and resolves earlier DROGA scale issues, but it cannot be approved until the interface contract breaks in `LOC_NFST_Bound` and input type handling are fixed.

### Mandatory Fixes for Worker M2:

1. **Update `LOC_NFST_Bound` in `fed_lunar/baselines/loc_nfst_bound.py`:**
   - **Expose properties:**
     ```python
     @property
     def null_basis(self) -> Optional[np.ndarray]:
         return self.W

     @property
     def threshold(self) -> float:
         return self.max_train_score
     ```
   - **Expose method alias:**
     ```python
     score = decision_function
     ```
   - **Update `fit()` signature to accept `tol`, `rounds`, and `**kwargs`:**
     ```python
     def fit(
         self,
         client_train_data: Union[List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray], np.ndarray],
         y_clusters: Optional[np.ndarray] = None,
         verbose: bool = False,
         tol: Optional[float] = None,
         rounds: Optional[int] = None,
         **kwargs: Any,
     ) -> "LOC_NFST_Bound":
         if tol is not None:
             self.epsilon_svd = float(tol)
             self.epsilon_near_null = float(tol)
     ```
   - **Incorporate Manifold Orthogonal Complement when $rank(P_t) < D$:**
     When SVD reveals rank deficiency (`rank_Pt < D`):
     Compute SVD with `full_matrices=True`. The columns $U_{:, rank\_Pt:}$ represent the exact null space where normal manifold variance is zero. Set $W = U_{:, rank\_Pt:}$ (or concatenate with $Q @ B$). This ensures normal samples on low-rank manifolds project to zero ($< 10^{-12}$).

2. **Normalize Input Handling Across All Baselines:**
   - In `NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, and `PCGradFedLunar`:
     Update `_format_client_data` to accept `(list, tuple)`:
     ```python
     elif isinstance(client_train_data, (list, tuple)):
         return [np.asarray(data, dtype=np.float32) for data in client_train_data]
     ```
   - Add `score = decision_function` alias across all baseline classes for uniform API consistency.

3. **Verify Master Test Suite:**
   Run `python tests/run_e2e_tests.py` and ensure all 126 tests across all 4 tiers pass with 100% success rate.

---

## 5. Verification Method

To independently verify the resolution of these findings:

1. **Execute the Master 4-Tier E2E Test Suite:**
   ```powershell
   python tests/run_e2e_tests.py
   ```
   *Expected outcome upon fix:* `126 passed in ~25s`, Exit code 0, Status `SUCCESS` across all 4 tiers.

2. **Execute the Milestone M2 Baseline Test Suite:**
   ```powershell
   pytest tests/test_baselines.py tests/test_m2_math_verification.py -v
   ```
   *Expected outcome:* All 34 tests pass.

3. **Execute Low-Rank Manifold Check:**
   ```powershell
   python -c "
   import numpy as np
   from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound
   basis = np.random.randn(10, 3)
   X_norm = (np.random.randn(50, 3) @ basis.T).astype(np.float32)
   loc = LOC_NFST_Bound().fit(X_norm, tol=1e-3)
   scores = loc.score(X_norm)
   assert np.all(scores < 1e-2), f'Normal score {np.max(scores)} exceeds 1e-2'
   assert loc.null_basis is not None
   assert loc.threshold is not None
   print('LOC-NFST Low-Rank Verification: PASS')
   "
   ```
   *Expected outcome:* Prints `LOC-NFST Low-Rank Verification: PASS`.
