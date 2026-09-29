# Handoff Report: Local Codebase & Architecture Survey

**Agent:** Explorer 1 (Local Codebase & Architecture Surveyor)  
**Working Directory:** `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_1`  
**Date:** 2026-09-22T15:55:00Z  
**Parent Conversation ID:** `37c8034b-fcb6-4906-bcf8-1f986e523ea0`  
**Handoff Type:** Hard Handoff (Task Complete)

---

## 1. Observation

1. **Git Repository Status:**
   - Command: `git status; git branch -a; git remote -v`
   - Active branch: `feature/federated-loc-nfst` (synchronized with `origin/feature/federated-loc-nfst`).
   - Remote: `https://github.com/MinhMarks/UIT_Research_NFST`.
   - Head commit: `c536801 fix: use AdynLOCNFST fit_temporal and predict in run_drift_injection`.
   - The required branch `feature/federated-lunar-novel` does not yet exist.
2. **Existing LUNAR Implementations:**
   - LUNAR is imported as an external baseline in `notebooks/baselines/fast_baselines.py` (line 42: `from pyod.models.lunar import LUNAR`), `notebooks/baselines/tune_baselines.py` (line 210: `model_dict = {..., "LUNAR": LUNAR}`), and `notebooks/baselines/Anomaly_Type_tune_baselines.py` (line 42).
   - In `notebooks/baselines/tune_baselines.py` (line 190), tuning parameters are specified as: `"LUNAR": [{"n_endpoints": 5}, {"n_endpoints": 10}, {"n_endpoints": 20}]`.
   - No custom PyTorch implementation of LUNAR (k-NN distance ranking + MLP + pseudo-negative generator) exists in the repository.
3. **Existing LOC-NFST Codebase (Tier 3 Upper Bound):**
   - Centralized memory-optimized LOC-NFST is implemented in `notebooks/experiments/OC_NFST_memory_optimized.py`:
     - `calculate_NPD_optimized` (lines 192–253): Computes $S_w$ incrementally, SVD on total deviation matrix $P_t \to Q$, solves $A = Q^T S_w Q$, computes null space $B = \text{null\_space}(A)$, and returns projection matrix $W = QB$.
   - Centralized baseline runner matching FL spectral solve is in `notebooks/experiments/fed_loc_nfst/run_centralized.py`:
     - `compute_centralized_NPD` (lines 55–130): Implements symmetric eigendecomposition (`eigh`) with near-null fallback ($\epsilon_{near\_null} = 10^{-4}$) to prevent $L=0$ collapse.
   - Federated LOC-NFST is implemented in `notebooks/experiments/fed_loc_nfst/`:
     - `strategy.py` (`FedLOCStrategy`, lines 308–450): Performs one-shot scatter aggregation via `aggregate_scatter_matrices` (lines 50–133) and adaptive spectral solve via `adaptive_spectral_solve` (lines 139–241).
     - `client.py` (`FedLOCClient`, lines 148–285): Performs Welford scatter accumulation via `compute_local_scatter` (lines 81–141).
4. **Unit Test Execution on FL-LOC-NFST:**
   - Ran `pytest notebooks/experiments/fed_loc_nfst/tests/test_scatter_aggregation.py`:
     - Result: `4 passed, 1 failed in 8.24s`.
     - Verbatim failure: `AssertionError: F1 deviation 65.3200% exceeds 5% tolerance on synthetic data. Centralized: {'AUCROC': 74.25, ..., 'F1 Score': 0.6532, ...} FL: {'AUCROC': 50.0, ..., 'F1 Score': 0.0, ..., 'Threshold': inf}`.
     - Root cause: In `strategy.py` line 270, `compute_null_centers` calculates `min_dist_to_centers(null_anchors, null_anchors)` across cluster anchors rather than training points, causing `max_train` distortion when computing anomaly probabilities.
5. **Existing Dataset Loaders:**
   - Located in `DataProcessing/`:
     - `BoTIoT.py` (line 26: 32 flow features).
     - `EdgeIIoTset.py` (line 30: 42 telemetry & flow features).
     - `CICIoT2023.py` (line 28: 46 statistical features).
     - `N_BaIoT.py` (line 33: 115 packet statistic features across time windows).
   - `DataProcessing/generate_oc_datasets.py` (lines 151–204) standardizes 30,000 normal train samples, 15,000 test normal samples, and 15,000 test anomaly samples, scaled via `StandardScaler`, `MinMaxScaler`, `Normalizer`, `QuantileTransformer`, `RobustScaler`.
   - `notebooks/experiments/fed_loc_nfst/data_utils.py` (lines 118–177) implements `partition_dirichlet`: partitions normal-only data into Non-IID clients using K-Means pseudo-clusters ($K = 3M$) and Dirichlet distribution sampling $\text{Dir}(\alpha)$.
   - `Datascaled/Official_OC_Data/` is currently an empty directory in the local repository; datasets reside on remote server `postmaster.iec` at `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`.
6. **Existing Baselines & Neural Architectures:**
   - `baseline_model/dasvdd_wrapper.py` (lines 24–57) implements `SimpleAutoEncoder(nn.Module)` with 3-layer encoder and symmetric decoder for tabular data.
   - `baseline_model/algorithms/net_torch.py` (lines 71–140) implements `MLPnet(torch.nn.Module)` with configurable activations and skip connections.
   - No implementations exist for FedAvg, FedProx, Fed-AE, PCGrad, or CAGrad.

---

## 2. Logic Chain

1. **R1 Feasibility:**
   - *Observation 2* shows LUNAR is only available via PyOD, which lacks PyTorch gradient exposure.
   - Therefore, a modular PyTorch implementation of LUNAR (`LUNAR_MLP` + k-NN distance feature extractor + pseudo-negative generator) must be developed from first principles.
   - Under Non-IID Dirichlet partitioning (*Observation 5*), disjoint client manifolds cause synthetic pseudo-negatives generated on Client A to intrude into Client B's normal manifold, causing $\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$.
   - Mitigating this requires Cross-Manifold Negative Purging (filtering candidate negatives against shared anchor bounding spheres) and Orthogonal Gradient Alignment (PCGrad projection at aggregation time).
2. **R2 Baseline Hierarchy Execution:**
   - *Tier 1 (Naive Fed-LUNAR):* Can be built by connecting the PyTorch LUNAR model to Flower simulation with standard FedAvg.
   - *Tier 2 (Fed-AE & PCGrad/FedProx LUNAR):* `SimpleAutoEncoder` (*Observation 6*) is already implemented and directly reusable for Fed-AE with reconstruction MSE loss. PCGrad can be implemented as an optimizer/strategy wrapper.
   - *Tier 3 (LOC-NFST Bound):* Centralized and federated LOC-NFST (*Observation 3*) are already fully operational and serve as the analytical upper bound.
3. **R3 Empirical Evaluation:**
   - Preprocessing scripts for all 4 canonical datasets exist in `DataProcessing/` (*Observation 5*).
   - Partitioning logic `partition_dirichlet` in `data_utils.py` directly handles $M \ge 3$ Non-IID clients.
4. **R4 Automated Verification:**
   - `evaluate.py` (*Observation 5*) provides Youden's J thresholding, AUC-ROC, and F1 calculations, which need to be supplemented with False Alarm Rate (FAR) and gradient conflict ratio logging.

---

## 3. Caveats

1. **Local Dataset Files Absent:** `Datascaled/Official_OC_Data/` is empty locally. Full benchmarking must execute on `postmaster.iec` where pre-scaled CSVs reside, or a synthetic mini-subset should be used for local unit tests.
2. **Python Environment on Local System:** System python environments lack `pyod`, `torch`, and `flwr` in global paths; tests and runners must use virtual environments with `requirements_fl.txt` installed.
3. **FL-LOC-NFST Threshold Inconsistency:** As observed in unit test failure (*Observation 4*), `compute_null_centers` in `strategy.py` uses cluster-to-cluster distance for `max_train` rather than sample-to-cluster distance. While centralized LOC-NFST remains the accurate upper bound, any comparative benchmark using FL-LOC-NFST should account for this normalization detail.

---

## 4. Conclusion

The repository provides solid infrastructure for dataset ingestion (`DataProcessing/`), Non-IID Dirichlet partitioning (`data_utils.py`), threshold evaluation (`evaluate.py`), tabular autoencoding (`SimpleAutoEncoder`), and closed-form null-space bounding (LOC-NFST).

The required development for R1–R4 is cleanly scoped:
1. Build a pure PyTorch LUNAR package (`fed_lunar/`) containing:
   - k-NN distance ranking MLP.
   - Pseudo-negative generator with Cross-Manifold Negative Purging (CMNP).
   - Orthogonal Gradient Alignment (PCGrad / CAGrad).
2. Wire up Flower clients and server strategies for the 3-tier baseline set:
   - Tier 1: Naive Fed-LUNAR (FedAvg).
   - Tier 2: Fed-AE (`SimpleAutoEncoder`) & PCGrad-adapted LUNAR.
   - Tier 3: LOC-NFST.
3. Execute benchmarks across BoTIoT, EdgeIIoTset, CICIoT2023, and N_BaIoT on branch `feature/federated-lunar-novel` logging metrics to `outputs/lunar_results/`.

---

## 5. Verification Method

1. **Survey Report Verification:**
   - Inspect `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_1\report.md` to confirm complete code inventory, line references, and architectural recommendations.
2. **LOC-NFST Baseline Unit Tests:**
   - Run: `pytest notebooks/experiments/fed_loc_nfst/tests/test_scatter_aggregation.py`
   - Verifies scatter shift correction, $S_t \equiv S_w + S_b$ decomposition, and near-null fallback.
3. **Git Status Verification:**
   - Run: `git status` to ensure working tree remains clean and unmodified (read-only compliance).
