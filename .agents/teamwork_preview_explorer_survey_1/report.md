# Local Codebase & Architecture Survey Report: Federated LUNAR & Baseline Integration

**Date:** 2026-09-22  
**Author:** Explorer 1 (Local Codebase & Architecture Surveyor)  
**Target Repository:** `D:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST`  
**Primary Objective:** Map existing LUNAR references, LOC-NFST null-space baseline, dataset loaders, and FL harnesses to support the design and benchmarking of the novel Federated LUNAR algorithm under Non-IID distributions (R1–R4).

---

## 1. Executive Summary

This survey systematically inspects the local codebase (`D:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST`) to evaluate its preparedness for executing the research plan specified in `ORIGINAL_REQUEST.md`. 

### Key Discoveries:
1. **Existing LUNAR Footprint:** LUNAR is referenced throughout the paper drafts (`main.tex`, `REPORT_NATURE_OF_DIVERSE_ANOMALIES_AND_CLUSTERING.md`) and benchmark grids (`notebooks/baselines/tune_baselines.py`, `fast_baselines.py`) as a high-performing baseline via `pyod.models.lunar.LUNAR`. However, no standalone PyTorch implementation of LUNAR or any federated variant currently exists in the repository.
2. **LOC-NFST Baseline (Tier 3 Upper Bound):** The repository features a mature and highly optimized implementation of centralized LOC-NFST (`notebooks/experiments/OC_NFST_memory_optimized.py`) and a Flower-based One-Shot Federated LOC-NFST harness (`notebooks/experiments/fed_loc_nfst/`). This provides the exact mathematical upper bound required by Tier 3 in R2.
3. **Data Loading & Preprocessing:** Production-grade loaders exist in `DataProcessing/` for all 4 canonical IoT datasets: `BoTIoT.py` (32 features), `EdgeIIoTset.py` (42 features), `CICIoT2023.py` (46 features), and `N_BaIoT.py` (115 features). The scaling and One-Class dataset generation pipeline is standardized in `generate_oc_datasets.py`.
4. **Federated Learning Infrastructure:** The repository possesses a working Flower (`flwr`) simulation harness in `fed_loc_nfst/run_fl_simulation.py` with Dirichlet-based Non-IID partitioning in `fed_loc_nfst/data_utils.py` and threshold-independent Youden's J evaluation in `fed_loc_nfst/evaluate.py`.
5. **Architectural Gaps:** The repository currently lacks:
   - Standalone, modular PyTorch implementation of LUNAR (k-NN distance ranking + MLP scoring + pseudo-negative generator).
   - Iterative FL engine (FedAvg / FedProx) for neural network weights.
   - Fed-AE (Federated AutoEncoder) baseline (though a compatible `SimpleAutoEncoder` architecture is available in `baseline_model/dasvdd_wrapper.py`).
   - Cross-Manifold Negative Purging and Orthogonal Gradient Alignment (PCGrad / CAGrad) mechanisms (R1).
   - Dedicated branch `feature/federated-lunar-novel` (currently on `feature/federated-loc-nfst`).

---

## 2. Git Status & Repository Configuration

### 2.1 Branch & Commit Inventory
- **Current Active Branch:** `feature/federated-loc-nfst` (synchronized with `origin/feature/federated-loc-nfst`).
- **Remote URL:** `https://github.com/MinhMarks/UIT_Research_NFST`
- **Latest Commit:** `c536801 fix: use AdynLOCNFST fit_temporal and predict in run_drift_injection`
- **Existing Remote Branches:**
  - `origin/main`
  - `origin/feature/federated-loc-nfst`
  - `origin/main1`
  - `origin/main_convertY`
  - `origin/main_tuned_baselines`
  - `origin/standard_folder_structure`
  - `origin/tune_metric_stuff`
- **Target Branch Status:** `feature/federated-lunar-novel` **does not yet exist** locally or remotely. It must be branched off `main` or `feature/federated-loc-nfst`.

### 2.2 Untracked & Worktree Files
The repository root contains several newly generated markdown reports and the `.agents/` metadata directory:
- `.agents/` (Agent metadata, plan, dispatch, and reports)
- `DE_CUONG_KHOA_LUAN_TOT_NGHIEP_LOC_NFST.md`
- `FORM_DANG_KY_DE_CUONG_KLTN.md`
- `GIAI_DAP_PHAN_BIEN_CHUYEN_SAU_LOC_NFST.md`
- `GIAI_MA_IEEE_VA_CASE_STUDY_TARGET_2013.md`
- `ORIGINAL_REQUEST.md`
- `REPORT_NATURE_OF_DIVERSE_ANOMALIES_AND_CLUSTERING.md`

---

## 3. Code Inventory

### 3.1 LOC-NFST Implementations (Theoretical Upper Bound)

| Component | File Path | Key Classes / Functions | Mathematical Functionality |
| :--- | :--- | :--- | :--- |
| **Centralized Optimized LOC-NFST** | `notebooks/experiments/OC_NFST_memory_optimized.py` | `calculate_NPD_optimized` (lines 192–253)<br>`project_to_null` (line 258)<br>`evaluate_scores` | Computes $S_w$ incrementally, SVD on $P_t \to Q$, solves $A = Q^T S_w Q$, null space $B = \text{null\_space}(A)$, $W = QB$. Scores via min Euclidean distance to normal samples in null space. |
| **Centralized Baseline Runner** | `notebooks/experiments/fed_loc_nfst/run_centralized.py` | `compute_centralized_NPD` (lines 55–130) | Adapts centralized LOC-NFST to use `eigh` with near-null fallback ($\epsilon_{near\_null} = 10^{-4}$) to prevent $L=0$ collapse on full-rank scatter matrices. |
| **Federated LOC-NFST Strategy** | `notebooks/experiments/fed_loc_nfst/strategy.py` | `FedLOCStrategy` (lines 308–450)<br>`aggregate_scatter_matrices` (lines 50–133)<br>`adaptive_spectral_solve` (lines 139–241) | One-Shot ($T=1$) server-side aggregation. Lossless scatter shift correction: $S_w^{global} = \sum S_w^m + \sum N_{mk}(\mu_{mk} - \mu_k)(\mu_{mk} - \mu_k)^T$, $S_t \equiv S_w + S_b$. |
| **Federated LOC-NFST Client** | `notebooks/experiments/fed_loc_nfst/client.py` | `FedLOCClient` (lines 148–285)<br>`compute_local_scatter` (lines 81–141) | Streaming Welford accumulation of local within-class scatter $S_w^m$ and cluster centroids $\mu_k^m$. Serialization into Flower `NDArrays`. |
| **FL Simulation Runner** | `notebooks/experiments/fed_loc_nfst/run_fl_simulation.py` | `run_fl_experiment` (lines 103–220)<br>`main` | Flower simulation launcher (`flwr.simulation.start_simulation`). Tracks client payload, inference latency, AUC-ROC, and F1. |
| **Dynamic Cluster Lifecycle (ADYN)** | `notebooks/experiments/fed_loc_nfst/adyn_core.py`<br>`notebooks/experiments/fed_loc_nfst/adyn_loc_nfst.py` | `SubspaceEngine`<br>`ClusterManager`<br>`QuarantineBuffer` | Online adaptation to concept drift: Rank-1 downdate (Split) and update (Merge) of matrix $A$, Brand's thin SVD update of $Q$ (Birth), sliding quarantine buffer. |

### 3.2 Existing Baselines & Neural Architectures

| Component | File Path | Key Classes / Functions | Details |
| :--- | :--- | :--- | :--- |
| **Deep AutoEncoder Architecture** | `baseline_model/dasvdd_wrapper.py` | `SimpleAutoEncoder` (lines 24–57) | PyTorch `nn.Module` with 3-layer encoder (input $\to \text{hidden}_1 \to \text{hidden}_2 \to \text{code}$) and symmetric decoder. LeakyReLU activations. Ideal substrate for Fed-AE! |
| **Deep Support Vector Data Description** | `baseline_model/dasvdd_wrapper.py` | `DASVDD` (lines 59–253) | PyOD-compatible wrapper for Deep SVDD with AutoEncoder pretraining. |
| **Deep Isolation Forest** | `baseline_model/dif_wrapper.py`<br>`baseline_model/algorithms/dif.py` | `DIF`, `DIForest` | Neural representation-based Isolation Forest. |
| **Neural Transformation AD** | `baseline_model/neutralad_wrapper.py` | `NeuTraLAD` | Self-supervised transformation prediction network. |
| **PyTorch Neural Backbones** | `baseline_model/algorithms/net_torch.py` | `MLPnet` (lines 71–140)<br>`GRUNet`, `LSTMNet`, `GinEncoderGraph` | Modular PyTorch neural network building blocks with configurable layers, activations, skip connections, and dropout. |
| **Baseline Grid Search & Tuning** | `notebooks/baselines/tune_baselines.py` | `get_model` (line 195)<br>`run_experiment` (line 240) | Tuning harness testing 20+ detectors (LUNAR, KNN, LOF, AutoEncoder, IForest, VAE, etc.) under noise contamination ($0\%$ to $10\%$). |
| **Multiple Kernel Fisher Null-Space** | `PMKFN.py` | `PMKFN` (lines 8–246) | Alternating optimization for $p$-norm Multiple Kernel One-Class Fisher Null-Space (Rahimzadeh Arashloo, 2020). |
| **Decomposition Representation Learning** | `DRLAD.py` | `DRLAD` (lines 47–248) | PyTorch implementation of DRL-AD with random orthogonal projection. |

### 3.3 Dataset Preprocessing & Loaders

| Dataset | Loader Script | Features | Characteristics & Normal Traffic Ratio |
| :--- | :--- | :--- | :--- |
| **BoTIoT** | `DataProcessing/BoTIoT.py` | 32 network flow features | DoS, DDoS, Service Scan, Keylogging, Data Exfiltration. Benign count: 9,542 samples. |
| **EdgeIIoTset** | `DataProcessing/EdgeIIoTset.py` | 42 telemetry & flow features | Multi-protocol IIoT attacks (Modbus, MQTT, HTTP, DNS). Benign count: 100,000 samples. |
| **CICIoT2023** | `DataProcessing/CICIoT2023.py` | 46 statistical features | High-throughput flood attacks (DDoS-UDP, ICMP, SYN). Benign count: 1,098,195 samples. |
| **N_BaIoT** | `DataProcessing/N_BaIoT.py` | 115 packet statistic features | Real hardware botnet across 9 devices over 5 time windows (100ms to 1min). High dimensional. |
| **Dataset Generator** | `DataProcessing/generate_oc_datasets.py` | Configurable | Splits raw traffic into 30,000 Train Normal (strictly benign), 15,000 Test Normal, 15,000 Test Anomaly. Scalers: StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, Normalizer. Output format: `Train_<Scaler>_data_<Dataset>.csv` and `Test_<Scaler>_data_<Dataset>.csv`. |
| **Partitioning Utilities** | `notebooks/experiments/fed_loc_nfst/data_utils.py` | `partition_dirichlet` (lines 118–177)<br>`load_dataset` (lines 32–74) | Implements Dirichlet-based Non-IID splitting of One-Class training data by clustering normal samples via K-Means into $K = 3M$ pseudo-subclasses, then assigning samples proportionally via $\text{Dir}(\alpha)$. |

---

## 4. Architectural Analysis: Strengths & Gaps (R1–R4)

### 4.1 Requirement R1: Mathematical Formulation & Novel Algorithmic Design
- **Requirement:** Formulate gradient conflict dynamics $\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$ caused by uncoordinated pseudo-negative generation across disjoint client manifolds. Implement Cross-Manifold Negative Purging with Orthogonal Gradient Alignment (PCGrad/CAGrad and Debiased Contrastive Learning).
- **Codebase Strengths:**
  - The repository demonstrates deep mathematical rigor in manifold and subspace analysis (as evidenced in `FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md` and `RESEARCH_DYNAMIC_K_LOC_NFST.md`).
  - The Dirichlet partitioning logic in `data_utils.py` precisely models the geometric fragmentation of normal manifolds across clients.
- **Identified Gaps:**
  - **No LUNAR GNN/MLP in PyTorch:** Currently, LUNAR is only called as a black-box PyOD model (`from pyod.models.lunar import LUNAR`). PyOD executes LUNAR in a monolithic scikit-learn style fit-predict loop without exposing internal PyTorch tensors, gradient hooks, or optimizer steps necessary for federated coordination.
  - **Absence of Gradient Conflict Tracking:** There is no existing code tracking pairwise gradient cosines $\cos \angle(g_i, g_j) = \frac{\langle g_i, g_j \rangle}{\|g_i\| \|g_j\|}$ during federated rounds.
  - **Absence of Orthogonal Projection Solvers:** Neither PCGrad (projecting conflicting gradients onto each other's normal hyperplanes) nor CAGrad (conflict-averse steepest descent) are implemented.
  - **Absence of Negative Purging:** LUNAR's standard pseudo-negative generator samples points uniformly within the feature bounding box or adds Gaussian noise to normal samples. In FL, when Client A generates pseudo-negatives, points that land on Client B's normal manifold are not currently identified or filtered.

### 4.2 Requirement R2: Baseline Hierarchy Implementation
- **Requirement:** 3-tier hierarchy:
  1. *Tier 1 (Base/Naive FL):* Naive Federated LUNAR (Standard FedAvg on LUNAR MLP weights with local subspace perturbation).
  2. *Tier 2 (SOTA Representation / Conflict-Aware):* FedAvg with Deep Autoencoder (Fed-AE) and PCGrad/FedProx-adapted LUNAR.
  3. *Tier 3 (Analytical Bound):* LOC-NFST (Null-Space closed-form baseline as the theoretical upper bound).
- **Codebase Strengths:**
  - **Tier 3 is fully implemented:** Centralized LOC-NFST (`OC_NFST_memory_optimized.py`) and Federated One-Shot LOC-NFST (`fed_loc_nfst/strategy.py`) are already complete, highly optimized, and benchmarked.
  - **AutoEncoder architecture is ready:** `SimpleAutoEncoder` in `baseline_model/dasvdd_wrapper.py` is written in pure PyTorch and fits tabular IoT data with arbitrary feature dimensions $d$.
- **Identified Gaps:**
  - **Tier 1 (Naive Fed-LUNAR) is missing:** Needs a PyTorch LUNAR client that performs local training with subspace perturbation and uploads raw model weights to a FedAvg aggregator.
  - **Tier 2 (Fed-AE & FedProx/PCGrad LUNAR) is missing:** While `SimpleAutoEncoder` exists, there is no federated harness training it across clients with MSE reconstruction loss, nor is there a FedProx/PCGrad baseline.
  - **Unit Test Flaw Identified in FL-LOC-NFST:** In `notebooks/experiments/fed_loc_nfst/tests/test_scatter_aggregation.py`, `TestEndToEndF1.test_fl_f1_within_tolerance` failed (`F1=0.0000`, `Threshold=inf`). Investigation reveals that `compute_null_centers` calculates distances between cluster anchors rather than between training points and anchors, causing $max\_train$ normalization distortion.

### 4.3 Requirement R3: Empirical Evaluation on 4 Canonical Datasets
- **Requirement:** Benchmark on `BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT` using pre-scaled datasets in `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/` on server `postmaster.iec`, with $M \ge 3$ Non-IID clients and $\le 5\%$ test contamination.
- **Codebase Strengths:**
  - Full data preprocessing scripts for all 4 datasets are co-located in `DataProcessing/`.
  - Feature dimensions and label mappings are rigorously defined (`CICIoT2023` = 46 fts, `N_BaIoT` = 115 fts, `EdgeIIoTset` = 42 fts, `BoTIoT` = 32 fts).
  - The remote directory structure (`/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`) mirrors the local `Datascaled/Official_OC_Data/`.
- **Identified Gaps:**
  - Pre-scaled CSV files are currently absent in the local `Datascaled/Official_OC_Data/` (directory is empty). They must be accessed on the remote server `postmaster.iec` or generated locally if needed for offline sanity checks.

### 4.4 Requirement R4: Automated Verification & Metrics Reporting
- **Requirement:** Output programmatic summary tables logging AUC-ROC, F1, False Alarm Rate (FAR), gradient conflict ratio ($\% \text{ rounds with } \cos \angle(g_i, g_j) < 0$), convergence round count, inference latency (ms), and peak memory (MB).
- **Codebase Strengths:**
  - `fed_loc_nfst/evaluate.py` already implements optimal threshold searching using Youden's J index ($J = \text{TPR} - \text{FPR}$), computing AUC-ROC, AUC-PR, F1, Precision, Recall, and Accuracy.
  - `tracemalloc` memory profiling and wall-clock timer patterns are established in `notebooks/baselines/tune_baselines.py`.
- **Identified Gaps:**
  - False Alarm Rate ($\text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) is not explicitly reported in `evaluate.py`.
  - Gradient conflict ratio logger needs to be created for the federated server strategy.
  - Structured output directory `outputs/lunar_results/` specified in R4 does not yet exist.

---

## 5. Architectural Deep-Dive: The Gradient Conflict Problem in Federated LUNAR

### 5.1 Mathematical Mechanism of LUNAR
In LUNAR (Goodge et al., AAAI 2022), each sample $x \in \mathbb{R}^d$ is mapped to a nearest-neighbor distance profile $D(x) = [\|x - \text{NN}_1(x)\|, \dots, \|x - \text{NN}_k(x)\|]^T \in \mathbb{R}^k$. An MLP $f_\theta: \mathbb{R}^k \to [0, 1]$ parameterizes the anomaly score.
The training objective uses pseudo-anomalies $\tilde{x} \sim \mathcal{Q}(X)$:
$$\mathcal{L}(\theta) = \frac{1}{|X|} \sum_{x \in X} \ell(f_\theta(D(x)), 0) + \frac{1}{|\tilde{X}|} \sum_{\tilde{x} \in \tilde{X}} \ell(f_\theta(D(\tilde{x})), 1)$$
where $\ell$ is Mean Squared Error (MSE) or Binary Cross-Entropy (BCE).

### 5.2 Breakdown under Non-IID Federated Learning
Consider two clients $A$ and $B$ with distinct normal sub-manifolds $\mathcal{M}_A$ and $\mathcal{M}_B$ ($\mathcal{M}_A \cap \mathcal{M}_B = \emptyset$):
1. **Cross-Manifold Intrusion:** Client $A$, having observed only $\mathcal{M}_A$, samples pseudo-negatives $\tilde{x}_A$ uniformly in the ambient subspace. A substantial fraction of $\tilde{x}_A$ inevitably lands on $\mathcal{M}_B$.
2. **Conflicting Supervised Signals:**
   - Client $A$ assigns label $y=1$ to samples falling in $\mathcal{M}_B$, producing gradient $\nabla_\theta \mathcal{L}_A$ that increases $f_\theta(D(x))$ for distance patterns typical of $\mathcal{M}_B$.
   - Client $B$ assigns label $y=0$ to its legitimate normal samples in $\mathcal{M}_B$, producing gradient $\nabla_\theta \mathcal{L}_B$ that decreases $f_\theta(D(x))$ for those exact same distance patterns.
3. **Adversarial Negative Gradient Cancellation:**
   $$\langle \nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B \rangle < 0$$
   Under standard FedAvg ($g_{avg} = \frac{1}{2}(g_A + g_B)$), the gradients directly cancel out along the subspace separating $\mathcal{M}_A$ and $\mathcal{M}_B$. The global model suffers from severe gradient stagnation, oscillating loss, and failure to learn boundary distinctions.

### 5.3 Algorithmic Solution Architecture
To resolve this, the novel Federated LUNAR architecture requires two synchronized components:
1. **Cross-Manifold Negative Purging (CMNP):**
   - Clients compute compact privacy-preserving centroid representations of their local normal manifolds (e.g., $K$ centroids with radius $R_k$, matching the low-payload approach in `fed_loc_nfst/client.py`).
   - Server broadcasts the global anchor set $\mathcal{C}_{global}$.
   - During local pseudo-negative synthesis on client $m$, any synthetic point $\tilde{x}$ that falls within the acceptance sphere of any other client's anchor is purged or debiased ($\tilde{x}$ is discarded or down-weighted via debiased contrastive weighting).
2. **Orthogonal Gradient Alignment (OGA):**
   - At each communication round, the server collects client gradient updates $g_1, \dots, g_M$.
   - The server inspects pairwise inner products $\langle g_i, g_j \rangle$.
   - If $\langle g_i, g_j \rangle < 0$, the server applies PCGrad projection:
     $$g_i \leftarrow g_i - \frac{\langle g_i, g_j \rangle}{\|g_j\|^2} g_j$$
   - The conflict ratio is logged programmatically, and the aligned gradient drives the server update.

---

## 6. Recommended Integration Points & File Architecture

To integrate the novel Federated LUNAR without disturbing the existing LOC-NFST code, we recommend adding a dedicated module under `notebooks/experiments/fed_lunar/` (or package `fed_lunar/`).

### Proposed File Layout

```
notebooks/experiments/fed_lunar/
├── __init__.py
├── config.py                 # Hyperparameters (k-NN size, learning rate, PCGrad flags, paths)
├── models/
│   ├── __init__.py
│   ├── lunar_mlp.py          # Pure PyTorch LUNAR Distance-Ranking MLP
│   └── autoencoder.py        # Fed-AE PyTorch Model (adapted from dasvdd_wrapper.py)
├── sampling/
│   ├── __init__.py
│   ├── negative_generator.py # Naive Subspace Perturbation generator
│   └── manifold_purger.py    # Cross-Manifold Negative Purging (CMNP) using global anchors
├── optim/
│   ├── __init__.py
│   ├── pcgrad.py             # Projecting Conflicting Gradients (PCGrad) algorithm
│   └── cagrad.py             # Conflict-Averse Gradient Descent (CAGrad) algorithm
├── client.py                 # Flower NumPyClient for LUNAR & Fed-AE
├── strategy.py               # Custom Flower Strategy (logging gradient conflict ratio & PCGrad)
├── run_lunar_fl.py           # End-to-end benchmark runner for Tiers 1, 2, 3 across the 4 datasets
└── tests/
    ├── __init__.py
    ├── test_lunar_model.py   # Unit test for LUNAR distance computation and forward pass
    ├── test_pcgrad.py        # Unit test verifying orthogonal projection when <g_i, g_j> < 0
    └── test_purging.py       # Unit test verifying cross-manifold negative purging
```

### Reusable Codebase Components

| New Module Requirement | Existing Codebase Asset to Reuse / Adapt | Location |
| :--- | :--- | :--- |
| **Dataset Loading & Non-IID Partition** | `load_dataset`, `partition_dirichlet`, `partition_iid` | `notebooks/experiments/fed_loc_nfst/data_utils.py` |
| **Scoring & Threshold Evaluation** | `evaluate_scores` (Youden's J ROC thresholding) | `notebooks/experiments/fed_loc_nfst/evaluate.py` |
| **Tier 3 Baseline (Analytical Bound)** | `FedLOCStrategy`, `compute_centralized_NPD` | `notebooks/experiments/fed_loc_nfst/strategy.py`<br>`notebooks/experiments/fed_loc_nfst/run_centralized.py` |
| **Deep AutoEncoder (Fed-AE)** | `SimpleAutoEncoder` | `baseline_model/dasvdd_wrapper.py` (lines 24–57) |
| **FL Simulation Runner Pattern** | Flower simulation configuration and client spawning | `notebooks/experiments/fed_loc_nfst/run_fl_simulation.py` |
| **Dataset Definitions & Preprocessing** | Raw dataset classes (`BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT`) | `DataProcessing/` |

---

## 7. Immediate Next Steps for Implementation

1. **Create Branch:** Create and check out git branch `feature/federated-lunar-novel` as required by acceptance criteria.
2. **Build PyTorch LUNAR Core:** Implement `lunar_mlp.py` (k-NN distance profile extractor + MLP) and verify standalone forward/backward pass.
3. **Build Gradient Alignment Engine:** Implement `pcgrad.py` with gradient projection and pairwise cosine conflict tracking ($\% \cos \angle < 0$).
4. **Implement Negative Purger:** Build `manifold_purger.py` utilizing anchor spheres shared across clients to eliminate cross-manifold intrusion.
5. **Implement Baselines:**
   - Tier 1: Naive Fed-LUNAR (FedAvg with unpurged negative sampling).
   - Tier 2: Fed-AE (FedAvg on `SimpleAutoEncoder`) and FedProx-LUNAR.
   - Tier 3: Link to existing LOC-NFST upper bound.
6. **Benchmark & Report:** Execute end-to-end simulation across BoTIoT, EdgeIIoTset, CICIoT2023, and N_BaIoT, outputting CSVs to `outputs/lunar_results/` and drafting `WALKTHROUGH_FEDERATED_LUNAR.md`.
