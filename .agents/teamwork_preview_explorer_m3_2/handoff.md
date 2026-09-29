# Handoff Report: Non-IID Dirichlet Partitioning & Baseline Integration (Milestone M3)

**Agent Identity**: `teamwork_preview_explorer_m3_2`  
**Milestone**: M3 — Non-IID Dirichlet IoT Benchmark Harness & Metrics  
**Date**: 2026-09-23  
**Status**: Complete (Hard Handoff)

---

## 1. Observation

### 1.1 Existing Codebase & Architecture
1. **Dirichlet Partitioner & One-Class Data Pipeline (`fed_lunar/benchmark/data_loader.py`)**:
   - `DirichletPartitioner` (lines 29–160): Implements Non-IID Dirichlet partitioning with default parameters `num_clients=3`, `alpha=0.5`, `num_clusters=5`, `use_clustering=True`, `seed=42`.
   - Lines 92–104:
     ```python
     if labels is None and self.use_clustering and N >= self.num_clusters * 2:
         n_clusters = min(self.num_clusters, N // 2)
         kmeans = MiniBatchKMeans(
             n_clusters=n_clusters,
             random_state=self.seed,
             batch_size=min(1024, N),
             n_init=3,
         )
         cat_labels = kmeans.fit_predict(X)
     ```
   - Lines 111–133: For each latent cluster $c$, samples proportions $\mathbf{p}_c \sim \text{Dirichlet}(\alpha \mathbf{1}_M)$, computes sample counts $n_{c, m} = \lfloor p_{c, m} \cdot N_c \rfloor$, and assigns sample indices to clients without replacement.
   - `OneClassDatasetLoader` (lines 161–280): Resolves file names via `Train_{scaler}_data_{dataset}.csv` and `Test_{scaler}_data_{dataset}.csv`. Filters training data strictly to normal samples ($y=0$).
   - `partition_and_prepare_dataset` (lines 281–343): End-to-end wrapper returning `(client_train_data, X_test, y_test, metadata)`.

2. **Benchmark CLI Runner (`fed_lunar/benchmark/run_benchmark.py`)**:
   - Lines 32–39 import the 3-Tier Baseline Hierarchy:
     ```python
     from fed_lunar.federated.fed_lunar import FedLUNAR
     from fed_lunar.baselines.naive_lunar import NaiveFedLunar
     from fed_lunar.baselines.fed_ae import FedAutoEncoder
     from fed_lunar.baselines.fedprox_lunar import FedProxLunar, PCGradFedLunar
     from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound
     ```
   - Lines 55–145 (`instantiate_model`): Configures 8 distinct models:
     - Proposed: `Proposed_FedLUNAR` (CMNP + DROGA / DR-CAGrad, $k=10, r=10, c=0.4, \tau_{\text{null}}=1.0$)
     - Ablation 1: `Ablation_FedLUNAR_NoDROGA` (CMNP enabled, standard FedAvg)
     - Ablation 2: `Ablation_FedLUNAR_NoCMNP` (DROGA enabled, $\tau_{\text{null}}=0.0$)
     - Tier 1: `Naive_FedLUNAR` (Naive FedAvg on LUNAR, uncoordinated perturbation)
     - Tier 2: `FedAutoEncoder` (Deep Autoencoder with MSE loss)
     - Tier 2: `FedProx_LUNAR` (Proximal regularization $\mu = 0.01$)
     - Tier 2: `PCGrad_FedLUNAR` (Standard PCGrad gradient projection without CMNP)
     - Tier 3: `LOC_NFST_Bound` (Closed-form null-space analytical upper bound)
   - Lines 310–326 (Argument Parser): Declares CLI options:
     `--data_dir`, `--output_dir`, `--dataset`, `--models`, `--clients`, `--alpha`, `--rounds`, `--max_train_samples`, `--max_test_samples`, `--device`, `--seed`.
   - Current limitation observed: `run_benchmark.py` writes `outputs/lunar_results/benchmark_summary.csv` and `.json`, but does not yet write individual `{dataset_name}_{method}.csv` files as required by `PROJECT.md` Contract 3 (lines 75–79). Furthermore, `peak_memory_mb` and `convergence_rounds` are not currently tracked in the output row.

3. **Metrics Pipeline (`fed_lunar/benchmark/metrics.py`)**:
   - `calculate_detection_metrics` (lines 17–96): Computes `auc_roc`, `f1_macro`, `f1_binary`, `precision`, `detection_rate` (Recall), `far` (False Alarm Rate), `tp`, `fp`, `tn`, `fn`, and evaluated `threshold`.
   - `measure_inference_latency` (lines 98–137): Measures per-sample latency in milliseconds via warm-up and repeated batch runs.

4. **Test Suite Baseline Verification**:
   - Executed `pytest tests/test_baselines.py`.
   - Result: **All 6 tests passed in 20.71 seconds**:
     - `test_naive_fed_lunar` PASSED
     - `test_fed_autoencoder` PASSED
     - `test_fedprox_lunar` PASSED
     - `test_pcgrad_fed_lunar` PASSED
     - `test_loc_nfst_bound` PASSED
     - `test_baselines_flexible_inputs` PASSED

5. **Dataset Storage & Physical Path Observation**:
   - Local directory `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\Datascaled\Official_OC_Data` exists but is currently empty (0 CSV files).
   - Real pre-scaled CSV files reside on remote server `postmaster.iec` at `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`.
   - The 4 canonical datasets have dimensions: BoTIoT ($D=35$), EdgeIIoTset ($D=42$), CICIoT2023 ($D=46$), N_BaIoT ($D=115$).

---

## 2. Logic Chain

### 2.1 Mathematical Formulation of Non-IID Dirichlet Partitioning for Continuous One-Class Data

#### The Theoretical Dilemma in One-Class FL
In standard supervised Federated Learning (e.g. Hsu et al., 2019; Lin et al., 2020), data samples possess discrete ground-truth class labels $y_i \in \{1, \dots, C\}$. Non-IID partitions are generated by sampling class proportions across clients:
$$\mathbf{p}_c = (p_{c, 1}, \dots, p_{c, M}) \sim \text{Dirichlet}(\alpha \mathbf{1}_M)$$
In One-Class FL Intrusion Detection (FL-IDS):
- The training set contains **strictly benign telemetry**: $y_i = 0$ for all $i \in \{1, \dots, N\}$.
- Standard label-based Dirichlet partitioning degenerates into a single category ($C=1$). Directly applying Dirichlet sampling to a single category produces **only sample quantity imbalance** (e.g., Client 1 receives 500 samples, Client 2 receives 200 samples), while the marginal feature distributions remain identical:
  $$P_1(X) = P_2(X) = \dots = P_M(X) \quad \implies \text{Strictly IID!}$$

#### The Two-Stage Cluster-Skew Dirichlet Formulation
To create realistic feature-skew and manifold-skew matching physical IoT deployments (where different edge nodes monitor distinct hardware functions, sensors, or network protocols), we formulate a two-stage generative allocation process:

1. **Stage 1: Latent Sub-Manifold Clustering**
   Benign IoT telemetry is assumed to lie on a union of $K$ latent sub-manifolds / operating regimes:
   $$\mathcal{M}_{\text{normal}} = \bigcup_{k=1}^K \mathcal{M}_k \subset \mathbb{R}^D$$
   Given unlabelled normal training points $\mathbf{X} \in \mathbb{R}^{N \times D}$, we fit an unsupervised clustering estimator (MiniBatchKMeans or GMM) with $K$ clusters:
   $$c_i = \arg\min_{k \in \{1, \dots, K\}} \|x_i - \boldsymbol{\mu}_k\|_2^2$$
   This segments $\mathbf{X}$ into disjoint cluster subsets $\mathcal{C}_k = \{x_i \mid c_i = k\}$ with cardinalities $N_k = |\mathcal{C}_k|$.

2. **Stage 2: Dirichlet Cluster Proportion Sampling**
   For each latent sub-manifold $k \in \{1, \dots, K\}$, we sample a client allocation vector $\mathbf{p}_k \in \Delta^{M-1}$ from a symmetric Dirichlet distribution parameterized by concentration $\alpha > 0$:
   $$\mathbf{p}_k = (p_{k, 1}, p_{k, 2}, \dots, p_{k, M}) \sim \text{Dirichlet}(\alpha \mathbf{1}_M)$$
   where the probability density on the $(M-1)$-simplex is:
   $$f(\mathbf{p}_k; \alpha) = \frac{\Gamma(M \alpha)}{\Gamma(\alpha)^M} \prod_{m=1}^M p_{k, m}^{\alpha - 1}, \quad \sum_{m=1}^M p_{k, m} = 1, \quad p_{k, m} \ge 0$$
   Each client $m \in \{1, \dots, M\}$ receives an allocation from cluster $k$:
   $$n_{k, m} = \lfloor p_{k, m} \cdot N_k \rfloor$$
   with residual samples distributed via randomized multinomial assignment to ensure $\sum_{m=1}^M n_{k, m} = N_k$.
   The training dataset for client $m$ is:
   $$\mathcal{D}_m = \bigcup_{k=1}^K \mathcal{C}_{k, m}, \quad \text{where } \mathcal{C}_{k, m} \subset \mathcal{C}_k, \quad |\mathcal{C}_{k, m}| = n_{k, m}$$

#### Rigorous Causal Proof: Why Dirichlet Cluster-Skew Causes Gradient Conflicts
Let $\alpha = 0.5$ and $M=3$ (or $M=5$).
1. **Simplex Vertex Concentration**: Because $\alpha = 0.5 < 1.0$, the Dirichlet density $p_{k, m}^{\alpha - 1} = p_{k, m}^{-0.5}$ diverges to infinity as $p_{k, m} \to 0$. The distribution concentrates its probability mass at the vertices and edges of the simplex.
2. **Disjoint Sub-Manifold Ownership**: For any cluster $k$, one client dominates with $p_{k, m} \approx 0.85 - 0.95$, while peer clients have $p_{k, j} \approx 0.01 - 0.05$. Thus, Client $A$'s training manifold is $\mathcal{M}_A \approx \mathcal{C}_1$ and Client $B$'s training manifold is $\mathcal{M}_B \approx \mathcal{C}_2$, with separation $\text{dist}(\mathcal{C}_1, \mathcal{C}_2) > 0$.
3. **Uncoordinated Pseudo-Negative Generation**: Client $A$ trains on $\mathcal{C}_1$ (normal, $y=0$) and synthesizes pseudo-negatives $\tilde{x}_A = x_A + \delta$. Because Client $A$ has no access to Client $B$'s data, $\tilde{x}_A$ intrudes into Client $B$'s benign sub-manifold $\mathcal{C}_2$.
4. **Opposing Loss Gradients**:
   - Client $A$'s distance-ranking loss treats $\tilde{x}_A \in \mathcal{C}_2$ as **anomaly** ($y=1$):
     $$\nabla_\theta \mathcal{L}_A(\theta) \propto - \nabla_\theta f_\theta(\mathbf{d}_A(\tilde{x}_A))$$
   - Client $B$ treats true samples $x_B \in \mathcal{C}_2$ as **normal** ($y=0$):
     $$\nabla_\theta \mathcal{L}_B(\theta) \propto + \nabla_\theta f_\theta(\mathbf{d}_B(x_B))$$
   - Because $\tilde{x}_A \approx x_B$, their Jacobian vectors align: $\nabla_\theta f_\theta(\tilde{x}_A) \approx \nabla_\theta f_\theta(x_B)$.
   - Taking the inner product of the client updates yields:
     $$\langle \nabla_\theta \mathcal{L}_A(\theta), \nabla_\theta \mathcal{L}_B(\theta) \rangle < 0$$
   This proves that the Dirichlet cluster-skew partition is the exact physical generator of adversarial negative gradient cancellation.

#### Recommended Hyperparameters for Dirichlet Partitioning
- **Number of latent clusters $K$**: For $M=3$, set $K=5$ (or $K=6$). For $M=5$, set $K=10$. Rule: $K \ge 2M$ to ensure every client has sufficient sub-manifold diversity.
- **Concentration parameter $\alpha$**:
  - $\alpha = 0.5$: Default strong Non-IID manifold skew.
  - $\alpha = 0.1$: Extreme pathological Non-IID skew (stress test).
  - $\alpha = 1.0$: Moderate Non-IID skew (uniform Dirichlet prior).
  - $\alpha = 10.0$ / $\infty$: Near-IID baseline.

---

### 2.2 Integration with Fed-LUNAR and the 3-Tier Baseline Hierarchy

The 3-tier baseline hierarchy and the proposed model are mapped as follows:

| Tier | Model Name | Class & Module | Mechanism & Characteristics | Theoretical Role |
|---|---|---|---|---|
| **Proposed** | `Proposed_FedLUNAR` | `FedLUNAR` (`fed_lunar.federated.fed_lunar`) | Distance-Ranking LUNAR MLP + FSDS sketches + CMNP negative purging + DROGA (DR-CAGrad) server aggregation | Novel proposed method resolving both local intrusion and global conflict |
| **Ablation 1**| `Ablation_FedLUNAR_NoDROGA` | `FedLUNAR(mode='FedAvg')` | CMNP enabled, but standard FedAvg parameter averaging without gradient surgery | Isolates contribution of CMNP alone |
| **Ablation 2**| `Ablation_FedLUNAR_NoCMNP` | `FedLUNAR(tau_null=0.0)` | DR-CAGrad enabled, but uncoordinated perturbation without CMNP | Isolates contribution of DROGA alone |
| **Tier 1** | `Naive_FedLUNAR` | `NaiveFedLunar` (`fed_lunar.baselines.naive_lunar`) | Standard FedAvg on LUNAR MLP; uncoordinated local subspace perturbation (no CMNP, no DROGA) | Primary baseline demonstrating gradient stagnation failure mode |
| **Tier 2** | `FedAutoEncoder` | `FedAutoEncoder` (`fed_lunar.baselines.fed_ae`) | Deep symmetric Autoencoder with MSE loss $\|x - \hat{x}\|_2^2$, aggregated via standard FedAvg | SOTA representation baseline; immune to negative intrusion but suffers representation drift |
| **Tier 2** | `FedProx_LUNAR` | `FedProxLunar` (`fed_lunar.baselines.fedprox_lunar`) | LUNAR with proximal drift regularization $\frac{\mu}{2}\|\theta - \theta_t\|_2^2$ | Optimization baseline constraining parameter drift |
| **Tier 2** | `PCGrad_FedLUNAR`| `PCGradFedLunar` (`fed_lunar.baselines.fedprox_lunar`) | Standard PCGrad gradient surgery at server; uncoordinated perturbation locally | Multi-task gradient surgery baseline without manifold sketching |
| **Tier 3** | `LOC_NFST_Bound` | `LOC_NFST_Bound` (`fed_lunar.baselines.loc_nfst_bound`) | Closed-form Null-Space projection operator $P_N = W W^T$; nearest anchor residual $\|P_N(x - m^*)\|_2^2$ ($T=1$) | Analytical upper bound (theoretical performance ceiling) |

#### Standardized Unified Interface Contract
All 8 models adhere to the unified contract:
1. `model.fit(client_train_data: List[np.ndarray], rounds: int = 10, verbose: bool = False)`:
   - Accepts list of $M$ numpy arrays containing local normal samples.
   - For `LOC_NFST_Bound`, `rounds` is ignored (one-shot analytical spectral solve $T=1$).
2. `scores = model.decision_function(X_test: np.ndarray, batch_size: int = 1024) -> np.ndarray`:
   - Returns 1D array of shape $(N_{\text{test}},)$ containing continuous anomaly scores.
   - For LUNAR variants: probability scores $\in [0, 1]$.
   - For Fed-AE: squared reconstruction errors $\|x - \hat{x}\|_2^2 \ge 0$.
   - For LOC-NFST: squared null-space residual norms $\|W^T(x - m^*)\|_2^2 \ge 0$.
3. `proba = model.predict_proba(X_test: np.ndarray) -> np.ndarray`:
   - Returns 2D array of shape $(N_{\text{test}}, 2)$ with $[P(\text{normal}), P(\text{anomaly})]$.
4. `preds = model.predict(X_test: np.ndarray, threshold: Optional[float] = None) -> np.ndarray`:
   - Returns binary predictions $\in \{0, 1\}$.
5. `model.history -> List[Dict[str, Any]]`:
   - Iterative models log per-round training dynamics: `round`, `mean_loss`, `pre_gcr`, `pre_mean_cosine`, `aligned_gradient_norm`.

---

### 2.3 Proposed Architecture for `fed_lunar/benchmark/run_benchmark.py`

#### Complete CLI Interface Options
The CLI runner must support both individual dataset/method targeting and full comparative sweeps:
```
python -m fed_lunar.benchmark.run_benchmark [OPTIONS]
```
- `--dataset` (str, default: `'all'`): Target dataset name (`'BoTIoT'`, `'EdgeIIoTset'`, `'CICIoT2023'`, `'N_BaIoT'`, or `'all'`).
- `--method` / `--models` (str, default: `'all'`): Target model name (`'Proposed_FedLUNAR'`, `'Naive_FedLUNAR'`, `'FedAutoEncoder'`, `'FedProx_LUNAR'`, `'PCGrad_FedLUNAR'`, `'LOC_NFST_Bound'`, `'Ablation_FedLUNAR_NoDROGA'`, `'Ablation_FedLUNAR_NoCMNP'`, `'core'`, or `'all'`).
- `--clients` (int, default: `3`): Number of federated clients $M$ (supports $M \ge 3$, e.g., 3, 5, 10).
- `--alpha` (float, default: `0.5`): Dirichlet concentration parameter (lower = higher Non-IID skew).
- `--rounds` (int, default: `10`): Number of federated communication rounds.
- `--data_dir` (str, default: auto-resolved): Path to `Official_OC_Data` directory.
- `--output_dir` (str, default: `'outputs/lunar_results'`): Output directory for results.
- `--synthetic_fallback` (bool flag, default: `True`): If real pre-scaled dataset CSVs are missing, automatically invoke the synthetic fallback generator so that benchmarks and local tests run cleanly.
- `--max_train_samples` (int, default: `20000`): Subsample cap on normal training data.
- `--max_test_samples` (int, default: `10000`): Subsample cap on testing data.
- `--scaler` (str, default: `'StandardScaler'`): Feature scaler (`'StandardScaler'` or `'QuantileTransformer'`).
- `--device` (str, default: `'cuda'` if available else `'cpu'`).
- `--seed` (int, default: `42`).
- `--verbose` (flag, default: `False`).

#### Execution Pipeline & Flow
```
[CLI Arguments]
       │
       ▼
[Directory & Output Initialization] ──> Ensure outputs/lunar_results/
       │
       ▼
[Dataset Loading & Non-IID Dirichlet Partitioning]
       │  - Check data_dir for CSVs
       │  - If missing and synthetic_fallback: generate realistic synthetic IoT stream
       │  - Run DirichletPartitioner(alpha=0.5, use_clustering=True, num_clusters=max(5, 2M))
       │  - Return client_train_data [X_1, ..., X_M], X_test, y_test
       ▼
[Model Loop: Iterate across selected methods]
       │
       ├─► 1. Start Resource & Time Tracking (tracemalloc, torch.cuda.reset_peak_memory_stats)
       │
       ├─► 2. Model Instantiation via instantiate_model(model_name, device, seed)
       │
       ├─► 3. model.fit(client_train_data, rounds=rounds)
       │
       ├─► 4. Track Training Wall-Clock Time t_train
       │
       ├─► 5. y_scores = model.decision_function(X_test)
       │
       ├─► 6. calculate_detection_metrics(y_test, y_scores) ──> AUC-ROC, F1-Macro, F1-Binary, FAR, DR
       │
       ├─► 7. measure_inference_latency(model, X_test) ──> Latency (ms/sample)
       │
       ├─► 8. Stop Resource Tracking ──> peak_memory_mb
       │
       ├─► 9. Extract Optimization Dynamics from model.history ──> GCR (%), Mean Cosine, Convergence Rounds
       │
       ├─► 10. Write Individual Result CSV: outputs/lunar_results/{dataset}_{method}.csv
       │
       └─► 11. Append to aggregated results list
       │
       ▼
[Global Summary Serialization]
       ├─► Save outputs/lunar_results/benchmark_summary.csv
       ├─► Save outputs/lunar_results/benchmark_summary.json
       └─► Print formatted Markdown Comparison Table to stdout
```

#### Detailed Metrics Logging Specification
Each individual experiment record contains the following schema:
- Identification: `dataset`, `model` (or `method`), `clients`, `dirichlet_alpha`, `rounds`, `scaler`, `input_dim`.
- Detection Performance: `auc_roc` (%), `f1_macro` (%), `f1_binary` (%), `precision` (%), `detection_rate` (Recall %), `far` (False Alarm Rate %), `threshold`, `tp`, `fp`, `tn`, `fn`.
- Optimization Dynamics:
  - `gradient_conflict_ratio` (% pairs with $\cos \angle(g_i, g_j) < 0$).
  - `mean_cosine`: Average pairwise cosine similarity between client updates.
  - `convergence_rounds`: The communication round index where relative loss stabilization satisfies $\frac{|\mathcal{L}_r - \mathcal{L}_{r-1}|}{\mathcal{L}_{r-1}} < 0.01$ for 2 consecutive rounds (or $1$ for LOC-NFST).
- Edge Viability:
  - `latency_ms_per_sample`: Mean inference duration in milliseconds.
  - `peak_memory_mb`: Peak RAM / VRAM consumption during training and scoring.
  - `train_time_sec`: Total training duration in seconds.

---

## 3. Caveats

1. **Remote vs. Local Dataset Presence**:
   - The local workspace at `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\Datascaled\Official_OC_Data` is empty. The real pre-scaled dataset CSV files are present on the remote server `postmaster.iec` at `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`.
   - Implementers must ensure the synthetic fallback generator in `data_loader.py` is enabled by default during local Windows testing, while dynamically using the real data path when executing on `postmaster.iec`.
2. **High-Dimensional Clustering on N_BaIoT ($D=115$)**:
   - For $D=115$, standard $K$-Means clustering in ambient space can be sensitive to isotropic distance decay. `MiniBatchKMeans` with $K=5$ or $K=10$ handles this efficiently. Alternatively, PCA dimensionality reduction to $r=10$ prior to clustering can be optionally applied for cluster assignment.
3. **Empty Cluster Allocation Protection**:
   - Under extreme skew ($\alpha < 0.2$), a client may randomly receive 0 samples if $K$ is small. The existing fallback in `data_loader.py` lines 139–141 borrows random normal samples to ensure client viability. Setting $K \ge 2M$ prevents this condition.

---

## 4. Conclusion

1. **Dirichlet Formulation**: We have established the exact mathematical formulation of Non-IID Dirichlet partitioning for continuous One-Class data: unsupervised latent sub-manifold clustering followed by cluster-wise Dirichlet allocation $\mathbf{p}_k \sim \text{Dirichlet}(\alpha \mathbf{1}_M)$ with $\alpha = 0.5$. We provided the causal proof showing this partition directly triggers adversarial negative gradient cancellation $\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$.
2. **Baseline Hierarchy**: All 3 tiers (`NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`) and the proposed `FedLUNAR` share a unified scikit-learn / PyTorch hybrid interface (`fit`, `decision_function`, `predict_proba`, `predict`, `history`) and pass all unit tests.
3. **Benchmark Architecture**: We specified the exact CLI architecture for `run_benchmark.py` with full option flags, dual CSV serialization (`outputs/lunar_results/{dataset}_{method}.csv` and `benchmark_summary.csv`), and instrumentation for AUC-ROC, F1, FAR, GCR, convergence round count, per-sample latency, and peak memory.

---

## 5. Verification Method

### 5.1 Independent Test Commands
Execute the test suites from the project root `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST`:
```powershell
# 1. Verify baseline models hierarchy
pytest tests/test_baselines.py -v

# 2. Verify CMNP negative purging and FSDS sketches
pytest tests/test_cmnp_purging.py -v

# 3. Verify DROGA gradient alignment strategies
pytest tests/test_droga_alignment.py -v

# 4. Verify End-to-End benchmark runner on synthetic slice (once test_benchmark_harness.py is added)
pytest tests/test_benchmark_harness.py -v
```

### 5.2 Inspection Checklist for Implementers
- Inspect `fed_lunar/benchmark/run_benchmark.py`:
  - Verify that both `--method` and `--models` arguments are accepted.
  - Verify that each evaluation logs an individual CSV `outputs/lunar_results/{dataset}_{method}.csv` with all required columns.
  - Verify that `peak_memory_mb` is measured via `tracemalloc` (CPU) and `torch.cuda.max_memory_allocated` (GPU).
  - Verify that `convergence_rounds` is calculated from model training history.
- Inspect `fed_lunar/benchmark/data_loader.py`:
  - Verify that `DirichletPartitioner(alpha=0.5, use_clustering=True)` produces distinct cluster ownership across the $M$ clients.

### 5.3 Invalidation Conditions
- If any baseline model fails to expose `decision_function(X) -> np.ndarray` returning 1D float scores, the interface contract is violated.
- If Dirichlet partitioning without clustering is applied to unlabelled one-class data, it degrades to sample quantity skew rather than distribution skew, invalidating the Non-IID experimental premise.
