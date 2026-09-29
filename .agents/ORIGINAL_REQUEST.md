# Original User Request

## 2026-09-22T15:48:35Z

Design, implement, and benchmark a novel Federated Learning adaptation of LUNAR that resolves the universal FL-IDS challenge of **Adversarial Negative Gradient Cancellation & Cross-Manifold Intrusion under Non-IID distributions**, verified empirically across 4 canonical IoT datasets on server `postmaster.iec` on a dedicated git branch.

Working directory: `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst` (remote `postmaster.iec`) / `D:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST` (local)
Branch: `feature/federated-lunar-novel`
Integrity mode: development

## Requirements

### R1. Mathematical Formulation & Novel Algorithmic Design
Formulate the exact gradient conflict dynamics $\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$ caused by uncoordinated pseudo-negative generation across disjoint client manifolds. Propose and implement a scientifically sound extension of LUNAR—integrating **Cross-Manifold Negative Purging** with **Orthogonal Gradient Alignment** (adapting foundations from NeurIPS/ICLR literature like PCGrad/CAGrad and Debiased Contrastive Learning to the distance-ranking architecture of LUNAR).

### R2. Baseline Hierarchy Implementation
Implement and compare against a strict 3-tier baseline set:
1. **Tier 1 (Base/Naive FL):** Naive Federated LUNAR (Standard FedAvg on LUNAR MLP weights with local subspace perturbation).
2. **Tier 2 (SOTA Representation / Conflict-Aware):** FedAvg with Deep Autoencoder (Fed-AE) and PCGrad/FedProx-adapted LUNAR.
3. **Tier 3 (Analytical Bound):** LOC-NFST (Null-Space closed-form baseline as the theoretical upper bound).

### R3. Empirical Evaluation on 4 Canonical Datasets
Execute full comparative benchmarks using pre-scaled datasets in `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/` on the following 4 distinct IoT domains:
- `BoTIoT` (Botnet DDoS & Information Theft)
- `EdgeIIoTset` (Industrial IoT multi-protocol traffic)
- `CICIoT2023` (Large-scale high-throughput IoT flood attacks)
- `N_BaIoT` (High-dimensional 115-feature commercial IoT hardware botnet)
Evaluation uses standardized One-Class protocol (Normal-only training across $M \ge 3$ Non-IID clients, contamination $\le 5\%$ in test stream).

### R4. Automated Verification & Metrics Reporting
Output programmatic summary tables logging:
- Detection performance: AUC-ROC (%), F1-Score, False Alarm Rate (FAR).
- Optimization dynamics: Gradient conflict ratio ($\% \text{ rounds with } \cos \angle(g_i, g_j) < 0$), convergence round count.
- Edge viability: Per-sample inference latency (ms) and peak memory (MB).

## Acceptance Criteria

### Convergence & Performance Criteria
- [ ] The novel Fed-LUNAR method demonstrates active mitigation of gradient conflicts, showing a demonstrable reduction in negative gradient cosine angles during training.
- [ ] On Non-IID client partitions across the 4 datasets, the novel Fed-LUNAR achieves a measurable AUC-ROC improvement over Naive Fed-LUNAR, preventing the gradient stagnation failure mode.
- [ ] All mathematical claims and citations in the generated reports are grounded in genuine, peer-reviewed literature with verified DOIs/proceedings.

### Server & Git Artifact Criteria
- [ ] Dedicated git branch `feature/federated-lunar-novel` is created and pushed to GitHub with clean commit structure.
- [ ] End-to-end benchmark runner executes completely on server `postmaster.iec` using Python 3 (`/opt/tljh/user/bin/python3`), outputting structured CSV results in `outputs/lunar_results/`.
- [ ] Complete walkthrough report document (`WALKTHROUGH_FEDERATED_LUNAR.md`) generated, analyzing the findings, ablation study results, and theoretical implications.

## 2026-09-24T09:07:42Z

Author a complete, publication-grade A* Security conference paper package in LaTeX (target: IEEE S&P / ACM CCS / USENIX Security / NDSS) that formally reshapes the problem definition, threat model, and research gap of Federated Distance-Ranking Graph Outlier Detection (Fed-LUNAR) for IoT Edge Networks. The paper package must feature rigorous mathematical theorems, step-by-step proofs of Out-of-Distribution Distance-Ranking Inversion and Cross-Manifold Negative Gradient Cancellation, full multi-dataset benchmark tables from real server executions, and competitive positioning against SOTA baselines.

Working directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST/paper_latex
Branch: feature/federated-lunar-novel
Integrity mode: development

## Requirements

### R1. Target Venue Formatting & Package Structure
- Produce a clean, self-contained LaTeX paper project in `paper_latex/` containing:
  - `main.tex` (using standard double-column IEEEtran conference format, 10–13 pages + references).
  - `references.bib` (exhaustive, verified peer-reviewed bibliography with genuine DOIs, authors, and venues; zero hallucinations).
  - Clean modular section files (`sec_intro.tex`, `sec_threat_model.tex`, `sec_formulation.tex`, `sec_methodology.tex`, `sec_proofs.tex`, `sec_experiments.tex`, `sec_related.tex`, `sec_conclusion.tex`).
- Ensure valid LaTeX syntax that compiles without errors or missing references.

### R2. Scientific Problem Reshaping & Threat Model (Security A* Standards)
- Formulate the system model and threat model under real-world IoT environments:
  - **System Model**: Decentralized, multi-tenant IoT edge gateways under Non-IID traffic distributions ($\mathcal{M}_i \cap \mathcal{M}_j \approx \emptyset$).
  - **Threat Model**: Coordinated botnet reconnaissance, stealthy multi-stage attacks (e.g., Mirai, Gafgyt), and volumetric floods.
- Explicitly articulate the **Defensible Research Gap**:
  - Contrast Fed-LUNAR against Autoencoder/Kitsune (reconstruction-based) and LOC-NFST (closed-form spectral null space).
  - Explain why high AUC on simple volumetric attacks is insufficient: Autoencoders suffer reconstruction shortcutting on stealthy multi-point attacks; LOC-NFST suffers catastrophic null-space erosion under streaming Non-IID drift.
  - Establish why distance-ranking graph topological awareness ($k$-NN relational geometry) is uniquely suited for stealthy coordinated IoT threats.
  - Transparently acknowledge system trade-offs: higher training complexity (server QP solving) and hyperparameter $k$ selection vs. sub-millisecond streaming inference ($0.001$ ms).

### R3. Rigorous Mathematical Theorems & Step-by-Step Proofs
- **Theorem 1 (Out-of-Distribution Distance-Ranking Inversion & Monotonicity Breakdown)**:
  - Formally prove that under fixed-radius perturbation $\epsilon$, as attack distance $d(x_{\text{ood}}, \mathcal{N}_k) \to \infty$, the distance-ranking MLP extrapolates with inverted gradient $\nabla_d f_\theta(d) \le 0$, collapsing anomaly scores ($AUC \to 0.15\%$).
  - Formally prove that Multi-Scale Subspace Perturbation (MSSP) with geometric spacing spans the unbounded metric space, guaranteeing strictly positive rank monotonicity $\partial f_\theta / \partial d > 0$ $\forall d \in \mathbb{R}^+$.
- **Theorem 2 (Adversarial Negative Gradient Cancellation in Non-IID FL)**:
  - Formally prove that uncoordinated subspace perturbation across disjoint client manifolds generates conflicting gradients satisfying $\mathbb{E}[\langle \nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B \rangle] < 0$, leading to vanishing global updates in standard FedAvg ($\|\nabla_\theta \mathcal{L}_{\text{global}}\| \approx 0$).
  - **Lemma 2.1 (Purging Invariance)**: Prove that Federated Subspace Density Sketches (FSDS) and Cross-Manifold Negative Purging (CMNP) filter out invasive pseudo-negatives with probability $\ge 1 - \delta$.
  - **Lemma 2.2 (Pareto Directional Convergence)**: Prove that Distance-Ranking Orthogonal Gradient Alignment (DROGA) guarantees monotonic Pareto-objective descent without gradient cancellation.

### R4. Complete Empirical Results & Multi-Dataset Presentation
- Fully populate LaTeX tables using verified empirical data from `outputs/lunar_results/` on server `postmaster.iec` (NVIDIA RTX 5090):
  1. **Master Benchmark Table**: Comparative evaluation of Fed-LUNAR against Naive Fed-LUNAR, FedAutoEncoder, FedProx-LUNAR, PCGrad-FedLUNAR, and LOC-NFST across 4 datasets (`BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT`) on AUC-ROC, F1-Score, and False Alarm Rate (FAR).
  2. **Dirichlet Non-IID Sensitivity Sweep Table**: Performance across $\alpha \in \{0.1, 0.5, 1.0, 5.0\}$ (32 runs), showing stability at extreme heterogeneity ($\alpha=0.1$).
  3. **Ablation Studies Table**: Isolating contributions of MSSP, CMNP (`NoCMNP` drop: $-26.23\%$ F1), and DROGA (`NoDROGA` drop: $-1.36\%$ F1).
  4. **Edge Feasibility & Resource Profiling Table**: Per-sample inference latency ($0.001$ ms), memory footprint (RAM/VRAM), and communication bandwidth overhead ($<5$ KB FSDS sketches).

### R5. Comprehensive Related Works & Competitive Matrix
- Detail the state-of-the-art taxonomy across 5 paradigms:
  1. Signature/Rule-based NIDS (Snort, Zeek)
  2. Statistical & Reconstruction-based (Kitsune, FedAutoEncoder)
  3. Tree & Ensemble (Isolation Forest)
  4. Spectral / Analytical (LOC-NFST, KNFST)
  5. Graph & Contrastive (LUNAR, NeuTraL AD, FedGNN)
- Include a high-density LaTeX comparison matrix highlighting threat models, communication costs, Non-IID robustness, and mathematical failure modes.

## Acceptance Criteria

### Mathematical & Scientific Completeness
- [ ] Theorems 1 & 2 have complete, mathematically valid proofs with explicit definitions of manifolds, metrics, loss functions, and probability bounds.
- [ ] No ungrounded claims: competitive positioning accurately reflects real benchmark trade-offs with baselines (FedAutoEncoder, LOC-NFST).

### LaTeX Engineering & Publication Readiness
- [ ] The LaTeX package in `paper_latex/` contains all necessary files (`main.tex`, modular sections, `references.bib`).
- [ ] All table entries correspond strictly to verified empirical logs in `outputs/lunar_results/benchmark_summary.csv` and `outputs/lunar_results/sensitivity_sweep/`.
- [ ] `references.bib` contains at least 30 genuine peer-reviewed publications from IEEE S&P, ACM CCS, USENIX Security, NDSS, NeurIPS, ICML, ICLR, AAAI, and IEEE INFOCOM/IoT-J with verified DOIs.
- [ ] Invariant check: Document header notes the originating research prompt per workspace rule.
