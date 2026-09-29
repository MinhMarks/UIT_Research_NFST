# Milestone M2 Handoff Report: 3-Tier Baseline Hierarchy & M1 Fixes

**Author:** Worker 1 (`teamwork_preview_worker_m2_1`)  
**Milestone:** M2 (3-Tier Baseline Hierarchy & Bug Fixes)  
**Parent Agent:** `parent` (`37c8034b-fcb6-4906-bcf8-1f986e523ea0`)  
**Date:** 2026-09-22T23:12:00Z  
**Status:** COMPLETE (100% Tests Passing)

---

## 1. Observation

### 1.1. Upstream Defect Observations (M1 Verification Issues)
1. **Flaky Fallback Test (`test_subspace_negative_generator_fallback`):**
   - File: `fed_lunar/models/negative_gen.py`, lines 280–283.
   - Original code:
     ```python
     ambient_dim = X_norm.shape[1]
     indices = self.rng.choice(X_norm.shape[0], size=count, replace=True)
     anchors = X_norm[indices]
     ```
   - In `tests/test_cmnp_purging.py`, line 195:
     ```python
     distances = np.linalg.norm(negatives - X_A, axis=1)
     assert np.all(distances >= 0.15)
     ```
   - When `count == X_norm.shape[0]`, `indices` with random replacement generated anchor points out-of-order with respect to `X_A`, causing `negatives[i] - X_A[i]` to compute distances between random pairs rather than the anchor and its own perturbation.

2. **Scale-Disparity Distortion in DROGA:**
   - File: `fed_lunar/federated/strategy.py`, lines 244–245 and 175–178.
   - Identified in Challenger 2 report (`teamwork_preview_challenger_m1_2/handoff.md`, lines 77–78):
     - Edge Case 3: Scale disparity $\|g_1\| = 10^4 \|g_2\|$ (antagonistic).
     - Original result: `CAGrad IPs: [6.00e7, -2999.38]` (Critical Failure: large negative inner product on the smaller gradient).

### 1.2. Implemented Codebase Artifacts
The 3-Tier Baseline Hierarchy was implemented from scratch across 5 files:
1. `fed_lunar/baselines/__init__.py`: Exports `NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`.
2. `fed_lunar/baselines/naive_lunar.py`: Feature F5 (Tier 1 Baseline) `NaiveFedLunar` (Standard FedAvg on LUNAR MLP weights with uncoordinated local subspace perturbation, no CMNP filter, no DROGA projection).
3. `fed_lunar/baselines/fed_ae.py`: Feature F6 (Tier 2 Baseline) `FedAutoEncoder` (Deep AutoEncoder using `SimpleAutoEncoder` with MSE reconstruction loss and FedAvg aggregation; anomaly score $\|x - \hat{x}\|_2^2$).
4. `fed_lunar/baselines/fedprox_lunar.py`: Feature F7 (Tier 2 Baselines):
   - `FedProxLunar`: Federated LUNAR with proximal regularization $\frac{\mu}{2}\|\theta - \theta_t\|^2$ added to local loss to counter client drift.
   - `PCGradFedLunar`: Federated LUNAR with standard PCGrad aggregation (Yu et al., NeurIPS 2020) without FSDS sketches or CMNP.
5. `fed_lunar/baselines/loc_nfst_bound.py`: Feature F8 (Tier 3 Analytical Upper Bound) `LOC_NFST_Bound` (Closed-form Null-Space analytical baseline $T=1$, integrating SVD total scatter decomposition, incremental within-class scatter, null-space spectral solve with near-null relaxation fallback, scoring as $\|P_N(x - m^*)\|_2^2$).
6. `tests/test_baselines.py`: Comprehensive test suite testing all 5 baseline classes, verifying fit, scoring, probability calibration, input flexibility (lists, dicts, arrays), and exception handling.

### 1.3. Verification Command Output
Executed:
`pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`
Verbatim output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.6, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: D:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST
plugins: anyio-4.12.0, hypothesis-6.152.1, cov-7.1.0
collecting ... collected 24 items

tests/test_baselines.py::test_naive_fed_lunar PASSED                     [  4%]
tests/test_baselines.py::test_fed_autoencoder PASSED                     [  8%]
tests/test_baselines.py::test_fedprox_lunar PASSED                       [ 12%]
tests/test_baselines.py::test_pcgrad_fed_lunar PASSED                    [ 16%]
tests/test_baselines.py::test_loc_nfst_bound PASSED                      [ 20%]
tests/test_baselines.py::test_baselines_flexible_inputs PASSED           [ 25%]
tests/test_lunar_model.py::test_lunar_mlp_init_and_shapes PASSED         [ 29%]
tests/test_lunar_model.py::test_lunar_mlp_invalid_k_and_dimensions PASSED [ 33%]
tests/test_lunar_model.py::test_knn_distance_extractor_accuracy PASSED   [ 37%]
tests/test_lunar_model.py::test_knn_distance_extractor_self_exclusion PASSED [ 41%]
tests/test_lunar_model.py::test_lunar_distance_ranking_loss PASSED       [ 45%]
tests/test_lunar_model.py::test_lunar_mlp_training_convergence_toy PASSED [ 50%]
tests/test_lunar_model.py::test_simple_autoencoder PASSED                [ 54%]
tests/test_cmnp_purging.py::test_fsds_sketch_computation_and_properties PASSED [ 58%]
tests/test_cmnp_purging.py::test_null_space_and_subspace_distances PASSED [ 62%]
tests/test_cmnp_purging.py::test_cmnp_hard_purging_disjoint_manifolds PASSED [ 66%]
tests/test_cmnp_purging.py::test_cmnp_continuous_soft_weighting PASSED   [ 70%]
tests/test_cmnp_purging.py::test_subspace_negative_generator_fallback PASSED [ 75%]
tests/test_cmnp_purging.py::test_theorem_2_cmnp_eliminates_antagonism PASSED [ 79%]
tests/test_droga_alignment.py::test_compute_gradient_conflict_metrics PASSED [ 83%]
tests/test_droga_alignment.py::test_dr_pcgrad_projection PASSED          [ 87%]
tests/test_droga_alignment.py::test_dr_cagrad_dual_simplex_qp PASSED     [ 91%]
tests/test_droga_alignment.py::test_droga_strategy_server_aggregation PASSED [ 95%]
tests/test_droga_alignment.py::test_client_training_round_and_aggregation PASSED [100%]

============================= 24 passed in 14.95s =============================
```

In addition, executing `python tests/stress_droga_harness.py` for Case 3 scale disparity yielded:
- `Case 3 CAGrad IPs: [15001141.0, 1500.081]` (strictly non-negative).
- `Case 3 PCGrad IPs: [18751600.0, 1875.119]` (strictly non-negative).

---

## 2. Logic Chain

1. **Step 1 (Fixing Flaky Fallback Index Mismatch):**  
   - Based on Section 1.1, `_fallback_boundary_noise` sampled `indices` via `rng.choice(replace=True)`.  
   - Replacing this with 1-to-1 anchor mapping `anchors = X_norm` when `count == n_norm` (and slicing/tiling when `count != n_norm`) guarantees that `negatives[i] = X_norm[i] + unit_dirs[i] * radii[i]`.  
   - Therefore, `norm(negatives[i] - X_norm[i]) == radii[i] >= 2.0 * sigma_pert = 0.20 >= 0.15`, permanently eliminating test flakiness.

2. **Step 2 (Unit-Norm Gradient Scaling in DROGA):**  
   - Based on Section 1.1, when client gradient magnitudes differ by orders of magnitude (e.g. $\|g_1\| = 10^4$, $\|g_2\| = 1$), the Gram matrix entries are dominated by $\|g_1\|^2 = 10^8$. The unscaled QP solver penalizes combining with $g_2$, producing an aligned direction that conflicts with $g_2$ ($\langle g_{\text{aligned}}, g_2 \rangle = -2999.38$).  
   - By scaling client gradients to unit vectors $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ prior to Gram matrix computation and pairwise projections, the optimization is performed purely in the angular domain where each client has equal geometric voice.  
   - Multiplying the normalized aligned direction $\tilde{g}_{\text{aligned}}$ by the mean client gradient norm $\bar{g}_{\text{norm}}$ preserves the physical gradient scale while guaranteeing strictly positive inner products ($\langle g_{\text{aligned}}, g_1 \rangle = 1.50 \times 10^7 > 0$, $\langle g_{\text{aligned}}, g_2 \rangle = 1500.08 > 0$).

3. **Step 3 (Standardized 3-Tier Baseline Architecture):**  
   - All 5 baseline models (`NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`) conform to a uniform scikit-learn compatible API:
     - `fit(client_train_data, rounds=10, ...)`
     - `decision_function(X)` $\to$ `np.ndarray` of shape `(N,)`
     - `predict_proba(X)` $\to$ `np.ndarray` of shape `(N, 2)`
     - `predict(X, threshold=...)` $\to$ `np.ndarray` of shape `(N,)`
   - Tier 1 (`NaiveFedLunar`): Implements uncoordinated local subspace perturbation without CMNP, and standard FedAvg parameter averaging without DROGA.
   - Tier 2 (`FedAutoEncoder`): Uses `SimpleAutoEncoder` with MSE reconstruction loss, FedAvg weight averaging, and anomaly scoring by $\|x - \hat{x}\|_2^2$.
   - Tier 2 (`FedProxLunar`): Extends local LUNAR loss with the proximal penalty $\frac{\mu}{2}\|\theta - \theta_t\|^2$.
   - Tier 2 (`PCGradFedLunar`): Uses standard PCGrad gradient projection without FSDS sketches or CMNP.
   - Tier 3 (`LOC_NFST_Bound`): Computes closed-form null-space projection matrix $W$ ($T=1$) via SVD on total scatter and null-space solve on within-class scatter, scoring test samples by squared projection norm $\|P_N(x - m^*)\|_2^2$.

4. **Step 4 (Test Suite Verification):**  
   - Tests in `tests/test_baselines.py` verify that all baseline models execute without error on multi-client non-IID datasets, generate properly formatted predictions, and separate normal data from anomalous intrusions with high AUC-ROC (>0.65 for iterative baselines, >0.85 for analytical upper bound).

---

## 3. Caveats

1. **Perturbation Scale Sensitivity in LUNAR Baselines:**  
   In LUNAR distance-ranking baselines (`NaiveFedLunar`, `FedProxLunar`, `PCGradFedLunar`), the perturbation scale `sigma_pert` must match or exceed the intra-cluster spread of local data (e.g. `sigma_pert >= 0.2` for synthetic Gaussian clusters with $\sigma = 0.2$). If `sigma_pert` is substantially smaller than the cluster radius, pseudo-negatives fall inside the dense cluster center, confusing normal and anomalous neighborhood distance profiles.
2. **Spectral Rank Cutoff in LOC-NFST:**  
   When training data has small sample sizes or near-zero variance features, `LOC_NFST_Bound` utilizes near-null spectral relaxation to select the smallest eigenvectors of $A$, ensuring that $L \ge L_{\text{min}}$ and projection matrix $W$ never collapses to zero dimension.

---

## 4. Conclusion

Milestone M2 is **COMPLETED**.
- All 4 required baseline files and 5 baseline classes (`NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`) are fully implemented in `fed_lunar/baselines/`.
- Both upstream defects (flaky fallback boundary noise indexing and scale-disparity distortion in DROGA) are verified and resolved.
- Full test suite passes with 100% success rate (24/24 tests passing).

---

## 5. Verification Method

To independently verify this implementation:

1. **Run the Full Unit Test Suite:**
   ```powershell
   pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v
   ```
   *Expected outcome:* All 24 tests pass cleanly.

2. **Verify Scale-Disparity Mitigation:**
   ```powershell
   python -c "
   import torch
   from fed_lunar.federated.strategy import dr_cagrad, dr_pcgrad
   v1 = torch.tensor([1.0, 0.0, 0.0])
   v3 = torch.tensor([0.0, 0.0, 1.0])
   g_large = v1 * 1e4
   g_small = (v1 * (-0.5) + v3 * 0.866) * 1.0
   grads = [g_large, g_small]
   g_ca = dr_cagrad(grads, c_param=0.4)
   g_pc = dr_pcgrad(grads, seed=42)
   print('CAGrad IPs:', [torch.dot(g_ca, g).item() for g in grads])
   print('PCGrad IPs:', [torch.dot(g_pc, g).item() for g in grads])
   "
   ```
   *Expected outcome:* All inner products are positive (> 1000.0).

3. **Verify Baselines Package Imports:**
   ```powershell
   python -c "from fed_lunar.baselines import NaiveFedLunar, FedAutoEncoder, FedProxLunar, PCGradFedLunar, LOC_NFST_Bound; print('Imports successful!')"
   ```
   *Expected outcome:* Prints `Imports successful!`.
