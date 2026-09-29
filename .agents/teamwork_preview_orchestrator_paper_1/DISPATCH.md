## 2026-09-24T09:14:07Z

You are the Project Orchestrator for authoring the complete, publication-grade A* Security conference paper package in LaTeX (target: IEEE S&P / ACM CCS / USENIX Security / NDSS) for Federated Distance-Ranking Graph Outlier Detection (Fed-LUNAR) for IoT Edge Networks.

### Identity & Working Directory
- Identity: Project Orchestrator
- Working directory: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_paper_1`
- Target LaTeX project directory: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex`
- Authoritative user request: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md` (header `## 2026-09-24T09:07:42Z`)

### Requirements to Fulfill:
1. **R1. Target Venue Formatting & Package Structure**:
   - Produce a self-contained LaTeX paper project in `paper_latex/` containing: `main.tex` (standard double-column IEEEtran conference format, 10–13 pages + references), `references.bib` (exhaustive, verified bibliography >=30 entries with genuine DOIs, authors, venues, zero hallucinations), and clean modular section files (`sec_intro.tex`, `sec_threat_model.tex`, `sec_formulation.tex`, `sec_methodology.tex`, `sec_proofs.tex`, `sec_experiments.tex`, `sec_related.tex`, `sec_conclusion.tex`). Ensure valid LaTeX syntax that compiles without errors or missing references.
2. **R2. Scientific Problem Reshaping & Threat Model (Security A* Standards)**:
   - System Model: Decentralized, multi-tenant IoT edge gateways under Non-IID traffic distributions ($\mathcal{M}_i \cap \mathcal{M}_j \approx \emptyset$).
   - Threat Model: Coordinated botnet reconnaissance, stealthy multi-stage attacks (e.g. Mirai, Gafgyt), and volumetric floods.
   - Defensible Research Gap: Contrast Fed-LUNAR against Autoencoder/Kitsune (reconstruction-based) and LOC-NFST (closed-form spectral null space). Explain why high AUC on simple volumetric attacks is insufficient (Autoencoder reconstruction shortcutting on stealthy multi-point attacks; LOC-NFST catastrophic null-space erosion under streaming Non-IID drift). Establish why distance-ranking graph topological awareness ($k$-NN relational geometry) is uniquely suited. Transparently acknowledge system trade-offs: higher training complexity (server QP solving) and hyperparameter $k$ selection vs. sub-millisecond streaming inference ($0.001$ ms).
3. **R3. Rigorous Mathematical Theorems & Step-by-Step Proofs**:
   - Theorem 1 (Out-of-Distribution Distance-Ranking Inversion & Monotonicity Breakdown): Formally prove that under fixed-radius perturbation $\epsilon$, as attack distance $d(x_{\text{ood}}, \mathcal{N}_k) \to \infty$, the distance-ranking MLP extrapolates with inverted gradient $\nabla_d f_\theta(d) \le 0$, collapsing anomaly scores ($AUC \to 0.15\%$). Formally prove that Multi-Scale Subspace Perturbation (MSSP) with geometric spacing spans the unbounded metric space, guaranteeing strictly positive rank monotonicity $\partial f_\theta / \partial d > 0$ $\forall d \in \mathbb{R}^+$.
   - Theorem 2 (Adversarial Negative Gradient Cancellation in Non-IID FL): Formally prove that uncoordinated subspace perturbation across disjoint client manifolds generates conflicting gradients satisfying $\mathbb{E}[\langle \nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B \rangle] < 0$, leading to vanishing global updates in standard FedAvg ($\|\nabla_\theta \mathcal{L}_{\text{global}}\| \approx 0$).
   - Lemma 2.1 (Purging Invariance): Federated Subspace Density Sketches (FSDS) and Cross-Manifold Negative Purging (CMNP) filter out invasive pseudo-negatives with probability $\ge 1 - \delta$.
   - Lemma 2.2 (Pareto Directional Convergence): Distance-Ranking Orthogonal Gradient Alignment (DROGA) guarantees monotonic Pareto-objective descent without gradient cancellation.
4. **R4. Complete Empirical Results & Multi-Dataset Presentation**:
   - Use empirical data strictly from `outputs/lunar_results/`:
     - Master Benchmark Table: Fed-LUNAR vs Naive Fed-LUNAR, FedAutoEncoder, FedProx-LUNAR, PCGrad-FedLUNAR, LOC-NFST across BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT on AUC-ROC, F1-Score, FAR.
     - Dirichlet Non-IID Sensitivity Sweep Table: $\alpha \in \{0.1, 0.5, 1.0, 5.0\}$ (32 runs in `outputs/lunar_results/sensitivity_sweep/alpha_sensitivity_summary.csv`).
     - Ablation Studies Table: Isolating MSSP, CMNP (`NoCMNP` drop: $-26.23\%$ F1), DROGA (`NoDROGA` drop: $-1.36\%$ F1).
     - Edge Feasibility & Resource Profiling Table: Per-sample inference latency ($0.001$ ms), memory footprint (RAM/VRAM), bandwidth overhead ($<5$ KB FSDS sketches).
5. **R5. Comprehensive Related Works & Competitive Matrix**:
   - 5 paradigms taxonomy: Signature/Rule-based (Snort, Zeek), Statistical & Reconstruction-based (Kitsune, FedAutoEncoder), Tree & Ensemble (Isolation Forest), Spectral/Analytical (LOC-NFST, KNFST), Graph & Contrastive (LUNAR, NeuTraL AD, FedGNN). High-density comparison matrix table.

### Existing Materials in Workspace:
- `references_master.bib`, `related_work_master.tex`, `main.tex`
- `BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md`, `WALKTHROUGH_FEDERATED_LUNAR.md`, `FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md`
- `outputs/lunar_results/benchmark_summary.csv`, `outputs/lunar_results/sensitivity_sweep/alpha_sensitivity_summary.csv`
