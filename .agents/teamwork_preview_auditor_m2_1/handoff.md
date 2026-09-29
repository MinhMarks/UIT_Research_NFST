# Milestone M2 Forensic Integrity Audit Report

**Auditor:** Forensic Integrity Auditor (`teamwork_preview_auditor_m2_1`)  
**Parent Agent:** `parent` (`37c8034b-fcb6-4906-bcf8-1f986e523ea0`)  
**Scope:** Milestone M2 Baseline Models (`fed_lunar/baselines/`), Baseline Tests (`tests/test_baselines.py`), and Upstream Bug Fixes (`negative_gen.py`, `strategy.py`)  
**Integrity Mode:** Development Mode (as specified in `ORIGINAL_REQUEST.md`)  
**Verdict:** **CLEAN** (Zero Integrity Violations)

---

## 1. Observation

### 1.1. Codebase Scope and Static Forensics
The following files were inspected line-by-line for static anti-patterns:
- `fed_lunar/baselines/__init__.py` (23 lines): Exports 5 baseline classes cleanly (`NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`).
- `fed_lunar/baselines/naive_lunar.py` (300 lines):
  - Lines 171–207: Genuine PyTorch local training loop using `torch.optim.Adam` and `LunarDistanceRankingLoss`.
  - Lines 212–218: Genuine parameter aggregation $\sum_i w_i \theta_i$ (standard FedAvg).
  - Lines 236–269: Authentic test inference querying `KNNDistanceExtractor` and `LUNAR_MLP.predict_proba`.
  - Zero hardcoded output arrays, dummy returns, or mock objects.
- `fed_lunar/baselines/fed_ae.py` (271 lines):
  - Lines 146–169: Authentic local autoencoder training with `F.mse_loss(recon, x_batch)` and Adam optimization.
  - Lines 174–180: Standard FedAvg model aggregation.
  - Lines 205–238: Authentic anomaly scoring via sample-wise squared reconstruction error $\|x - \hat{x}\|_2^2 = \sum_d (x_d - \hat{x}_d)^2$.
  - Lines 193–201: Dynamic threshold calibration from 99th percentile of normal data.
- `fed_lunar/baselines/fedprox_lunar.py` (553 lines):
  - Lines 171: Initial parameters cloned: `init_params = [p.detach().clone() for p in client_model.parameters()]`.
  - Lines 201–206: Proximal penalty calculation: `prox_term = torch.sum((p - p_init) ** 2)`, `total_loss = ranking_loss + (0.5 * self.mu) * prox_term`.
  - Lines 470–491: PCGrad client effective gradient computation $g_c = \theta_{\text{init}} - \theta_{\text{final}}$ and orthogonal aggregation via `dr_pcgrad`.
- `fed_lunar/baselines/loc_nfst_bound.py` (241 lines):
  - Lines 110: Total scatter basis computed via `np.linalg.svd(P_t, full_matrices=False)`.
  - Lines 130–137: Incremental within-class scatter matrix $S_w = \frac{1}{N} \sum_k \sum_{x \in C_k} (x - m_k)(x - m_k)^T$.
  - Lines 140–163: Null-space decomposition $A = Q^T S_w Q$ using `scipy.linalg.null_space` with `scipy.linalg.eigh` near-null relaxation fallback.
  - Lines 211–215: Vectorized null-space projection scoring: $z = (x - m^*) W$, $\text{score} = \|z\|_2^2$.
- `tests/test_baselines.py` (253 lines):
  - Grep for `assert True` or trivial tautologies returned 0 occurrences.
  - All test assertions check non-trivial properties: shapes, probabilities summing to 1.0, non-negative scores, proper exception raising before `fit()`, and genuine separation criteria (`roc_auc_score(y_test, scores) > 0.65`, `> 0.70`, `> 0.85`).
- `fed_lunar/models/negative_gen.py` (lines 280–296):
  - Corrected `_fallback_boundary_noise` to align anchor indexing directly with $X_{\text{norm}}$ when count matches.
- `fed_lunar/federated/strategy.py` (lines 166–173, 254–261):
  - Implemented unit-norm gradient scaling $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ to resolve scale-disparity distortion.

### 1.2. Independent Test Execution
Executed command:
`pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`

Verbatim Output:
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

============================= 24 passed in 29.16s =============================
```

### 1.3. Empirical Tracing & Optimization Probes

#### Probe 1: FedProx Drift Penalty Verification
- Measured client parameter drift $\|\theta_{\text{final}} - \theta_{\text{init}}\|$ across identical initializations:
  - When $\mu = 0.0$: Parameter drift = `3.3564`
  - When $\mu = 10.0$: Parameter drift = `0.3140`
  - **Result:** Proximal penalty reduced drift by **10.69x**, confirming authentic regularization in gradient backprop.

#### Probe 2: Loss Descent Across Federated Rounds
- Monitored mean training loss per round on non-IID 2-client dataset:
  - `NaiveFedLunar`: Round 1: `1.3966` $\to$ Round 2: `1.3878` $\to$ Round 3: `1.3852` $\to$ Round 4: `1.3753` (strictly descending).
  - `PCGradFedLunar`: Round 1: `1.3966` $\to$ Round 2: `1.3878` $\to$ Round 3: `1.3854` $\to$ Round 4: `1.3750` (strictly descending).
  - **Result:** Authentic continuous optimization and parameter convergence.

#### Probe 3: Fed-AE Reconstruction Error Separation
- Evaluated reconstruction errors $\|x - \hat{x}\|_2^2$:
  - Normal sample mean reconstruction error: `6.9116`
  - Anomaly sample mean reconstruction error: `227.5974`
  - **Result:** Separation ratio of **32.93x**, demonstrating genuine autoencoder representation learning.

#### Probe 4: LOC-NFST Bound Mathematical Authenticity
- Evaluated exact null-space projection properties:
  - Projection matrix orthonormality: $\|W^T W - I_L\|_\infty = 1.220767 \times 10^{-15}$ (exact machine precision).
  - Normal sample residual score $\|W^T(x - m^*)\|_2^2$: `0.0915`
  - Anomaly sample residual score $\|W^T(x - m^*)\|_2^2$: `110.3676`
  - **Result:** Separation ratio of **1206.2x**, confirming genuine spectral null-space projection.

#### Probe 5: Upstream Bug Fixes Empirical Validation
- Boundary noise radius bounds (`_fallback_boundary_noise` with $\sigma_{\text{pert}} = 0.1$):
  - Count = 50: min distance = `0.2008`, max distance = `0.3470` (strictly $\in [2.0\sigma, 3.5\sigma]$).
  - Count = 25: min distance = `0.2027`, max distance = `0.3496` (strictly $\in [2.0\sigma, 3.5\sigma]$).
- Scale disparity handling in DROGA ($\|g_1\| = 10^4 \|g_2\|$ and $\cos \angle(g_1, g_2) < 0$):
  - CAGrad inner products: `[15001141.0, 1500.081]` (strictly $> 0$).
  - PCGrad inner products: `[18751600.0, 1875.119]` (strictly $> 0$).

#### Probe 6: Edge Cases and Flexible Input Handling
- Single query input shape `(1, D)`: Evaluated across all 5 baselines; returned `decision_function` shape `(1,)`, `predict_proba` shape `(1, 2)`, and `predict` shape `(1,)`.
- PyTorch tensor inputs `torch.randn(5, D)`: Evaluated across all 5 baselines without device mismatch or type error.

---

## 2. Logic Chain

1. **Step 1 (Integrity Mode Context):**  
   - `ORIGINAL_REQUEST.md` specifies `Integrity mode: development`. Under development mode, the primary prohibitions are hardcoded test results, facade implementations, and fabricated verification outputs.
2. **Step 2 (Absence of Prohibited Anti-Patterns):**  
   - Static analysis across `naive_lunar.py`, `fed_ae.py`, `fedprox_lunar.py`, `loc_nfst_bound.py`, and `test_baselines.py` revealed zero instances of hardcoded return values, dummy implementations, or trivial tautological assertions.
3. **Step 3 (Empirical Verification of Core Algorithmic Claims):**  
   - Runtime tracing verified that FedProx actively penalizes client model drift ($\Delta \theta$ dropped from 3.356 to 0.314), that Fed-AE reconstructs normal inputs with 32x lower error than anomalies, and that LOC-NFST achieves machine-precision orthonormality ($1.22 \times 10^{-15}$) and 1200x anomaly separation.
4. **Step 4 (Test Execution & Fix Validation):**  
   - All 24 unit and regression tests pass natively in 29.16 seconds without failure or flakiness.
   - The upstream defects (fallback boundary noise distance mismatch and scale-disparity gradient conflict) were verified to be genuinely resolved.
5. **Step 5 (Verdict Synthesis):**  
   - Since all forensic checks passed empirically and no integrity violations were detected under any integrity mode, the work product is certified CLEAN.

---

## 3. Caveats

- **Scalability to Exascale Networks:** All tests and empirical probes were executed in single-machine simulated environments ($M \in [1, 5]$ clients, $D \in [4, 8]$ features, batch sizes up to 128). Full evaluation on massive IoT datasets (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT) is scheduled for Milestone M3.
- **Spectral Rank Sensitivity:** Under extreme ill-conditioning, `LOC_NFST_Bound` relies on near-null spectral relaxation ($L_{\text{min}}=3$), which produces approximate null-space directions rather than pure algebraic nullity. This behavior is documented and mathematically expected.

---

## 4. Conclusion

**Verdict: CLEAN**

Milestone M2 work products (`fed_lunar/baselines/naive_lunar.py`, `fed_lunar/baselines/fed_ae.py`, `fed_lunar/baselines/fedprox_lunar.py`, `fed_lunar/baselines/loc_nfst_bound.py`, `fed_lunar/baselines/__init__.py`, `tests/test_baselines.py`, and bug fixes in `negative_gen.py` and `strategy.py`) are fully compliant with all architectural contracts, contain genuine computational logic, and exhibit zero integrity violations.

---

## 5. Verification Method

To independently verify the forensic findings:

1. **Execute Complete Test Suite:**
   ```powershell
   pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v
   ```
   *Expected result:* 24 passed in ~30s.

2. **Empirical Optimization Dynamics Probe:**
   ```powershell
   python -c "
   import numpy as np, torch
   from fed_lunar.baselines import FedProxLunar, FedAutoEncoder, LOC_NFST_Bound
   from fed_lunar.models.lunar_mlp import LUNAR_MLP
   c1 = np.random.randn(50, 6).astype(np.float32) * 0.2 + 1.0
   c2 = np.random.randn(50, 6).astype(np.float32) * 0.2 - 1.0
   # FedProx
   m0 = FedProxLunar(k=5, mu=0.0, device='cpu', seed=42).fit([c1], rounds=1)
   m10 = FedProxLunar(k=5, mu=10.0, device='cpu', seed=42).fit([c1], rounds=1)
   init = {k: v.cpu() for k, v in LUNAR_MLP(k=5).state_dict().items()}
   d0 = sum(torch.norm(m0.global_model.state_dict()[k].cpu() - init[k]).item() for k in init)
   d10 = sum(torch.norm(m10.global_model.state_dict()[k].cpu() - init[k]).item() for k in init)
   print(f'Drift mu=0: {d0:.4f}, Drift mu=10: {d10:.4f}')
   assert d10 < d0
   # LOC-NFST
   loc = LOC_NFST_Bound(n_clusters=2).fit([c1, c2])
   print('Ortho error:', np.max(np.abs(loc.W.T @ loc.W - np.eye(loc.L))))
   assert np.max(np.abs(loc.W.T @ loc.W - np.eye(loc.L))) < 1e-4
   print('PROBE PASS!')
   "
   ```
   *Expected result:* Prints `PROBE PASS!`.
