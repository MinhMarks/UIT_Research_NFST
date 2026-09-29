# Project: Novel Federated LUNAR for Non-IID IoT Intrusion Detection

## Architecture & System Overview
Novel Federated Learning adaptation of LUNAR (Local Outlier Neural Network) designed to overcome Adversarial Negative Gradient Cancellation and Cross-Manifold Intrusion under Non-IID distributions in IoT Intrusion Detection.

The architecture comprises:
1. **Local LUNAR Neural Engine (`fed_lunar/models/`)**:
   - $k$-NN distance extractor: Extracts sorted Euclidean distance vectors $\mathbf{d}_c(z) \in \mathbb{R}^k$ from local benign manifold $\mathcal{M}_c$.
   - LUNAR Ranking MLP $f_\theta$: Multi-layer perceptron mapping distance vectors to anomaly likelihood scores $\hat{y} \in [0, 1]$.
   - Subspace Pseudo-Negative Generator with **Cross-Manifold Negative Purging (CMNP)**: Synthesizes pseudo-negatives $\tilde{x} = x + \delta$ while rejecting candidates that intrude into other clients' normal manifolds via Federated Subspace Density Sketches (FSDS: $\{\mu_c, \Lambda_c, U_c, r_c^{\max}\}$).
2. **Federated Optimization Engine (`fed_lunar/federated/`)**:
   - Client Local Trainer: Trains $f_\theta$ on local benign points ($y=0$) and purged pseudo-negatives ($y=1$), computing gradient updates $g_c = \nabla_\theta \mathcal{L}_c(\theta)$.
   - **Distance-Ranking Orthogonal Gradient Alignment (DROGA / DR-PCGrad & DR-CAGrad)**: Aggregates client updates while actively resolving antagonistic gradient directions ($\langle g_i, g_j \rangle < 0$) via orthogonal projection / minimax dual simplex QP.
3. **3-Tier Baseline Hierarchy (`fed_lunar/baselines/`)**:
   - Tier 1: Naive Federated LUNAR (Standard FedAvg on LUNAR MLP weights with uncoordinated local subspace perturbation).
   - Tier 2: FedAvg with Deep Autoencoder (Fed-AE using `SimpleAutoEncoder`) and PCGrad/FedProx-adapted LUNAR.
   - Tier 3: LOC-NFST (Null-Space closed-form baseline as the theoretical upper bound).
4. **Non-IID IoT Benchmark Harness (`fed_lunar/benchmark/`)**:
   - One-Class Non-IID Dirichlet Partitioner ($M \ge 3$ clients, contamination $\le 5\%$ test stream).
   - Benchmark runner across 4 canonical datasets: BoTIoT (35 fts), EdgeIIoTset (42 fts), CICIoT2023 (46 fts), N_BaIoT (115 fts).
   - Metrics Logger: AUC-ROC (%), F1-Score, False Alarm Rate (FAR), Gradient Conflict Ratio (% rounds with $\cos \angle(g_i, g_j) < 0$), Convergence Round Count, Per-sample Latency (ms), and Peak Memory (MB), outputting structured CSVs to `outputs/lunar_results/`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | PyTorch LUNAR Distance-Ranking MLP | Core modular neural network mapping $k$-NN distance vectors to anomaly probabilities | M1 | Survey (E1, E3) |
| F2 | Mathematical Formulation of Gradient Conflict | Analytical derivation & proof of $\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$ under Non-IID pseudo-negative intrusion | M1 | Survey (E3) |
| F3 | Cross-Manifold Negative Purging (CMNP) | Subspace density sketch filtering (FSDS) rejecting pseudo-negatives intruding on peer manifolds | M1 | Survey (E3) |
| F4 | Orthogonal Gradient Alignment (DROGA) | Server-side DR-PCGrad and DR-CAGrad gradient projection resolving conflicting client updates | M1 | Survey (E3) |
| F5 | Tier 1 Baseline: Naive Fed-LUNAR | Standard FedAvg aggregation on LUNAR with uncoordinated perturbation | M2 | Survey (E1, E3) |
| F6 | Tier 2 Baseline: Fed-AE | Federated Deep Autoencoder baseline with reconstruction loss | M2 | Survey (E1, E3) |
| F7 | Tier 2 Baseline: FedProx / PCGrad LUNAR | FedProx with proximal term $\frac{\mu}{2}\|\theta - \theta_t\|^2$ and standard PCGrad | M2 | Survey (E1, E3) |
| F8 | Tier 3 Baseline: LOC-NFST Bound | Closed-form Null-Space analytical upper bound ($T=1$) | M2 | Survey (E1) |
| F9 | Non-IID Dirichlet Dataset Partitioner | Standardized One-Class loader partitioning benign traffic across $M \ge 3$ clients ($\alpha=0.5$) | M3 | Survey (E1, E2) |
| F10 | Multi-Dataset Benchmarking Pipeline | Ingestion and evaluation across BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT | M3 | Survey (E1, E2) |
| F11 | Automated Metrics & CSV Logging | Logging AUC-ROC, F1, FAR, gradient conflict ratio, rounds, latency, memory to `outputs/lunar_results/` | M3 | Survey (E1, E2) |
| F12 | Remote Server Execution on postmaster.iec | Automated headless benchmarking on GPU RTX 5090 using `/opt/tljh/user/bin/python3` | M4 | Survey (E2) |
| F13 | Git Branch Management | Dedicated branch `feature/federated-lunar-novel` created, cleanly committed, and pushed | M4 | Survey (E1, E2) |
| F14 | Comprehensive Walkthrough Report | `WALKTHROUGH_FEDERATED_LUNAR.md` with theoretical analysis, ablation study, and peer-reviewed citations | M5 | Survey (E1, E3) |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Fed-LUNAR Engine & Algorithms | F1, F2, F3, F4: PyTorch LUNAR, CMNP negative purging, DROGA gradient alignment | none | DONE |
| M2 | 3-Tier Baseline Hierarchy | F5, F6, F7, F8: Naive Fed-LUNAR, Fed-AE, FedProx-LUNAR, LOC-NFST bound | M1 | DONE |
| M3 | Non-IID Benchmark Harness & Metrics | F9, F10, F11: Dirichlet partitioner, 4-dataset runner, CSV logging to `outputs/lunar_results/` | M1, M2 | IN_PROGRESS |
| M4 | Remote Execution & Git Publication | F12, F13: Branch `feature/federated-lunar-novel`, remote execution on postmaster.iec, CSV verification | M3 | PLANNED |
| M5 | Walkthrough Documentation & Audit | F14: `WALKTHROUGH_FEDERATED_LUNAR.md`, peer-reviewed citations, final verification audit | M4 | PLANNED |

---

## Interface Contracts

### 1. Local LUNAR Engine ↔ Client Trainer
- `LUNAR_MLP(k: int, hidden_dims: list[int] = [64, 32, 16], dropout: float = 0.1) -> nn.Module`:
  - Input: Tensor of sorted $k$-NN Euclidean distances `(batch_size, k)`
  - Output: Anomaly probabilities `(batch_size, 1)` in $[0, 1]$
- `NegativeGenerator(negative_ratio: float = 1.0, epsilon: float = 0.1, cmnp: Optional[CMNPFilter] = None)`:
  - Input: Normal samples `X_normal` `(N, D)`
  - Output: Synthetic negative samples `X_neg` `(N * negative_ratio, D)` where candidates violating peer FSDS sketches are purged.

### 2. Client Trainer ↔ Federated Server Aggregator
- `ClientUpdate`:
  - Input: Global model weights $\theta_t$, peer FSDS sketches $\{\mathcal{S}_j\}_{j \ne i}$
  - Output: Updated weights $\theta_{t+1}^{(i)}$, local gradient/delta $g_i = \theta_t - \theta_{t+1}^{(i)}$, number of samples $n_i$, local FSDS sketch $\mathcal{S}_i = \{\mu_i, \Lambda_i, U_i, r_i^{\max}\}$.
- `ServerAggregator (DROGA / DR-PCGrad / DR-CAGrad)`:
  - Input: Client gradients $\{g_1, \dots, g_M\}$, client sample weights $\{w_1, \dots, w_M\}$
  - Output: Aligned global update $g_{\text{aligned}}$ satisfying $\langle g_{\text{aligned}}, g_i \rangle \ge 0$ for all $i \in [M]$, and updated global weights $\theta_{t+1} = \theta_t - \eta g_{\text{aligned}}$.
  - Logs: Pairwise cosine similarity matrix $C_{ij} = \frac{\langle g_i, g_j \rangle}{\|g_i\| \|g_j\|}$ and gradient conflict ratio $\text{GCR} = \frac{\sum_{i < j} \mathbb{I}(C_{ij} < 0)}{\binom{M}{2}}$.

### 3. Benchmark Runner ↔ Metrics Output
- `run_benchmark(dataset_name: str, method: str, clients: int = 3, alpha: float = 0.5, rounds: int = 15, scaler: str = 'QuantileTransformer')`:
  - Output: Dictionary logged to `outputs/lunar_results/{dataset_name}_{method}.csv` containing:
    `dataset, method, clients, alpha, rounds, auc_roc, f1_score, far, gradient_conflict_ratio, convergence_rounds, latency_ms_per_sample, peak_memory_mb`.

---

## Code Layout
```
fed_lunar/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── lunar_mlp.py         # PyTorch LUNAR MLP architecture & distance-ranking loss
│   ├── negative_gen.py      # Subspace perturbation & Cross-Manifold Negative Purging (CMNP)
│   └── autoencoder.py       # SimpleAutoEncoder for Fed-AE baseline
├── federated/
│   ├── __init__.py
│   ├── client.py            # Local client trainer (LUNAR, Fed-AE, FedProx)
│   ├── strategy.py          # Server aggregation strategies (FedAvg, DR-PCGrad, DR-CAGrad)
│   └── sketches.py          # Federated Subspace Density Sketches (FSDS)
├── baselines/
│   ├── __init__.py
│   ├── naive_lunar.py       # Tier 1: Naive Fed-LUNAR
│   ├── fed_ae.py            # Tier 2: Fed-AE
│   ├── fedprox_lunar.py     # Tier 2: FedProx / PCGrad LUNAR
│   └── loc_nfst_bound.py    # Tier 3: LOC-NFST closed-form upper bound
├── benchmark/
│   ├── __init__.py
│   ├── data_loader.py       # One-Class loader & Dirichlet partitioner for 4 datasets
│   ├── metrics.py           # AUC-ROC, F1, FAR, gradient conflict ratio, latency, memory
│   └── run_benchmark.py     # Master benchmark CLI runner
tests/
├── test_lunar_model.py      # Unit tests for LUNAR MLP and distance ranking
├── test_cmnp_purging.py     # Unit tests for CMNP and FSDS sketch filtering
├── test_droga_alignment.py  # Unit tests for DR-PCGrad and DR-CAGrad gradient alignment
├── test_baselines.py        # Unit tests for 3-tier baselines
└── test_e2e_benchmark.py    # Opaque-box E2E pipeline tests
outputs/
└── lunar_results/           # CSV results across all 4 datasets and 3 tiers + novel method
```
