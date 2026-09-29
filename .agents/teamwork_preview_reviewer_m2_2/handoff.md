# Milestone M2 Review & Adversarial Challenge Report: Baseline Math & Numerical Stability

**Reviewer:** Reviewer 2 (`teamwork_preview_reviewer_m2_2`)  
**Roles:** reviewer, critic  
**Target Milestone:** M2 (3-Tier Baseline Hierarchy & M1 Fixes)  
**Parent Agent:** `parent` (`37c8034b-fcb6-4906-bcf8-1f986e523ea0`)  
**Verdict:** **REQUEST_CHANGES**  
**Date:** 2026-09-23T01:10:00Z  

---

## 1. Observation

### 1.1. Scope and Implementation Inspected
The review inspected the following core files and baseline implementations:
1. `fed_lunar/baselines/fedprox_lunar.py`:
   - `FedProxLunar`: Proximal regularization penalty $\frac{\mu}{2} \|\theta - \theta_t\|^2$ added to local ranking loss (lines 170–210).
   - `PCGradFedLunar`: Federated LUNAR with standard PCGrad aggregation at server level without CMNP or FSDS sketches (lines 420–495).
2. `fed_lunar/baselines/fed_ae.py` & `fed_lunar/models/autoencoder.py`:
   - `SimpleAutoEncoder`: Symmetric encoder-decoder architecture with LeakyReLU activations and latent bottleneck (lines 58–84).
   - `FedAutoEncoder`: True MSE reconstruction loss `F.mse_loss(recon, x_batch)` (lines 161–166), FedAvg parameter aggregation (lines 173–180), and sample-wise squared Euclidean anomaly scoring $\|x - \hat{x}\|_2^2$ calibrated to the 99th percentile of normal training envelope (lines 193–202, 233–235, 250–256).
3. `fed_lunar/baselines/loc_nfst_bound.py`:
   - `LOC_NFST_Bound`: Closed-form Null-Space analytical baseline ($T=1$), total scatter deviation matrix $P_t$, SVD decomposition $P_t = U S V^T$ with rank truncation $Q = U[:, :rank\_Pt]$ (lines 106–114), within-class scatter $S_w = \frac{1}{N} \sum_k \sum_{x \in C_k} (x - m_k)(x - m_k)^T$ (lines 130–137), subspace projection $A = Q^T S_w Q$ (line 140), null-space solve with near-null spectral relaxation fallback (lines 142–165), projection matrix $W = Q B$, projector $P_N = W W^T$ (lines 166–170), and nearest-centroid distance scoring $\|W^T (x - m^*)\|_2^2$ (lines 200–214).
4. `fed_lunar/federated/strategy.py`:
   - Unit-norm gradient scaling $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ prior to Gram matrix computation and pairwise projections in `dr_pcgrad` (lines 166–173) and `dr_cagrad` (lines 254–261).
   - Rescaling aligned unit direction by average norm $\bar{g}_{\text{norm}} = \sum_i w_i \|g_i\|$ to preserve physical optimization step size (lines 193, 316, 350).
5. `tests/test_baselines.py`:
   - Unit tests across all 5 baseline models on synthetic multi-client non-IID datasets.
6. `tests/run_e2e_tests.py` and `tests/e2e/contract_stubs.py`:
   - Master 4-tier E2E contract test suite established in Milestone M1.

### 1.2. Verification Command Results

1. **Independent Mathematical Verification Suite (`tests/test_m2_math_verification.py`):**
   10/10 tests passed (21.91s), verifying FedProx autograd exactness, drift suppression, PCGrad orthogonality, unit-norm scale disparity invariance, zero-gradient stability, autoencoder MSE loss, and LOC-NFST SVD / near-null spectral decomposition.

2. **Baseline Unit Test Suite (`pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`):**
   24/24 tests passed (43.29s).

3. **Adversarial Stress Harness (`python tests/stress_droga_harness.py`):**
   Completed 4,000 randomized trials and extreme edge cases. Confirmed that unit-norm gradient scaling eliminates scale disparity distortion on Case 3 ($\|g_1\| = 10^4 \|g_2\|$) with strictly positive inner products on both gradients ($1.5 \times 10^7$ and $1500.08$).

4. **Master E2E Test Suite (`python tests/run_e2e_tests.py`):**
   **FAILED** with 10 failures out of 126 tests (exit code 1):
   - Tier 1: 4 failed out of 70.
   - Tier 2: 1 failed out of 40.
   - Tier 3: 4 failed out of 11.
   - Tier 4: 1 failed out of 5.
   - Exact failure trace in all 10 tests:
     - `TypeError: LOC_NFST_Bound.fit() got an unexpected keyword argument 'tol'`
     - `AttributeError: 'LOC_NFST_Bound' object has no attribute 'score'`
     - `AttributeError: 'LOC_NFST_Bound' object has no attribute 'null_basis'`
     - `AttributeError: 'LOC_NFST_Bound' object has no attribute 'threshold'`

---

## 2. Findings

### [Major] Finding 1: Interface Contract Regression in `LOC_NFST_Bound` (10 E2E Test Failures)
- **What:** In `tests/e2e/contract_stubs.py` and `tests/e2e/test_tier1_features.py`, `test_tier2_boundaries.py`, `test_tier3_pairwise.py`, `test_tier4_applications.py`, the established contract for `LOC_NFST_Bound` requires:
  1. `fit(X, tol=...)`: parameter `tol` for singular matrix tolerance.
  2. `score(X)`: anomaly score evaluation.
  3. `null_basis`: property exposing projection matrix $W$.
  4. `threshold`: property exposing the decision threshold.
  In `fed_lunar/baselines/loc_nfst_bound.py`, `fit()` does not accept `tol` (or `**kwargs`), `score()` was named `decision_function()`, `null_basis` was named `W`, and `threshold` was named `max_train_score`.
- **Where:** `fed_lunar/baselines/loc_nfst_bound.py`, lines 82–180, 182–215.
- **Why:** `tests/run_e2e_tests.py` executes `get_loc_nfst_bound()` from `contract_stubs.py`. Because `LOC_NFST_Bound` is imported, it replaces the reference stub, but crashes 10 existing E2E tests across all 4 tiers due to missing compatibility aliases.
- **Suggestion:** In `fed_lunar/baselines/loc_nfst_bound.py`:
  1. Update `fit()` signature to accept `tol: Optional[float] = None, **kwargs`:
     ```python
     def fit(self, client_train_data, y_clusters=None, verbose=False, tol: Optional[float] = None, **kwargs):
         if tol is not None:
             self.epsilon_svd = tol
         ...
     ```
  2. Add compatibility method and properties:
     ```python
     def score(self, X: Union[np.ndarray, Any]) -> np.ndarray:
         """Alias for decision_function to conform to E2E contract."""
         return self.decision_function(X)

     @property
     def null_basis(self) -> Optional[np.ndarray]:
         """Alias for projection matrix W."""
         return self.W

     @property
     def threshold(self) -> float:
         """Alias for decision threshold."""
         return self.max_train_score
     ```

### [Minor] Finding 2: Subspace Null Space vs Intra-Cluster Null Space in LOC-NFST
- **What:** In `test_f8_loc_nfst_null_basis_orthogonal_to_normal` (`test_tier1_features.py:690`), normal data is generated from a low-rank linear manifold (rank 3 in 10D). The test asserts that normal samples in the null space project to $< 10^{-2}$.
- **Where:** `fed_lunar/baselines/loc_nfst_bound.py`, lines 110–145.
- **Why:** `LOC_NFST_Bound` follows the NFST formulation from `notebooks/experiments/fed_loc_nfst/adyn_loc_nfst.py`, which defines $Q$ as the range of $S_t$ (the 3D manifold), and then computes within-class scatter $S_w$ across $K=3$ KMeans clusters. In a 3D subspace, 3 cluster centers span the 3D space, so $S_w$ is full-rank inside $Q$, forcing near-null relaxation to pick eigenvectors inside the normal subspace rather than the orthogonal complement.
- **Suggestion:** When `n_clusters == 1` or when intrinsic rank is lower than ambient dimension, ensure that if the null space of $S_w$ inside $Q$ is empty, the orthogonal complement $U[:, rank\_Pt:]$ (the true null space of the normal data manifold) is utilized for null-space projection.

---

## 3. Logic Chain

1. **Step 1 (Math & Numerical Verification):**
   - FedProx proximal regularization: $\mathcal{L}_{\text{ranking}} + \frac{\mu}{2}\|\theta - \theta_t\|^2$ verified. PyTorch autograd gradient matches the exact analytical gradient down to $10^{-6}$ precision (`TestFedProxMathAndGradients`).
   - PCGrad sequential orthogonalization: `dr_pcgrad` sequentially projects conflicting updates against the original peer unit vectors with random peer shuffling.
   - Unit-Norm Gradient Scaling: $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ maps gradients into the angular domain prior to Gram matrix computation and projection, successfully resolving scale disparity ($\|g_1\| = 10^4 \|g_2\|$).
   - FedAutoEncoder: True MSE reconstruction loss `F.mse_loss` and sample-wise squared error anomaly scoring $\|x - \hat{x}\|_2^2$ calibrated to the 99th percentile envelope verified.

2. **Step 2 (Integrity Audit):**
   - Verified zero hardcoded outputs, zero facade implementations, zero shortcuts.
   - All models execute genuine optimization, autograd differentiation, SVD decomposition, and null-space linear algebra.

3. **Step 3 (E2E Regression Discovery):**
   - While Worker M2's targeted test suite (`tests/test_baselines.py`) passes 100%, executing the project master E2E test runner (`python tests/run_e2e_tests.py`) revealed 10 test failures across Tiers 1, 2, 3, and 4 due to missing method aliases in `LOC_NFST_Bound` (`tol`, `score`, `null_basis`, `threshold`).
   - In accordance with the Teamwork Quality Review standards and Reviewer constraints (never self-modify source code, report failures as findings), this regression requires changes before Milestone M2 can be approved.

---

## 4. Caveats

1. **Sequential PCGrad Greedy Re-Conflict:**
   In `stress_droga_harness.py`, for $M \ge 3$ on an antagonistic simplex, sequential pairwise PCGrad can re-introduce conflict with prior peers (well-known theoretical property of pairwise PCGrad). DR-CAGrad is recommended when strictly non-negative inner products are required.
2. **Edge Case 4 in DR-CAGrad:**
   When gradients are exactly collinear opposing ($180^\circ$) with extreme scale disparity ($10^4$ vs $10^{-4}$), their normalized sum is zero ($\tilde{g}_0 = 0$), causing fallback to the unnormalized sum which conflicts with the tiny gradient. In high-dimensional IoT neural networks ($P \gg 10^3$), gradients across non-IID clients never form exact 1D collinear opposition.

---

## 5. Conclusion

**Verdict: REQUEST_CHANGES**

- **Mathematical Correctness & Numerical Stability:** VERIFIED AND SOUND. FedProx proximal regularization, PCGrad orthogonalization, FedAutoEncoder MSE loss, and unit-norm gradient scaling are mathematically correct and robust.
- **Blocker:** Major finding in `fed_lunar/baselines/loc_nfst_bound.py`: Interface contract mismatch causes 10 test failures in `tests/run_e2e_tests.py` (`fit(..., tol=...)`, `score()`, `null_basis`, `threshold`).
- **Required Action:** Worker M2 must add the backward-compatible aliases to `LOC_NFST_Bound` so that `python tests/run_e2e_tests.py` passes with 0 failures (126/126 tests passing).

---

## 6. Verification Method

To independently verify the issue and validate the fix:

1. **Reproduce the 10 E2E Failures:**
   ```powershell
   python tests/run_e2e_tests.py
   ```
   *Current result:* `FAILED: 10 test(s) failed out of 126`.

2. **Verify Mathematical Exactness:**
   ```powershell
   pytest tests/test_m2_math_verification.py -v
   ```
   *Expected result:* 10/10 passed.

3. **Verify Expected Resolution:**
   Add `score`, `null_basis`, `threshold`, and `tol` parameter to `LOC_NFST_Bound` and re-run:
   ```powershell
   python tests/run_e2e_tests.py
   ```
   *Target result:* `[SUCCESS] All 126 tests in selected tier(s) passed successfully with exit code 0`.
