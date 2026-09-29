## 2026-09-24T09:15:10Z

You are an Explorer subagent (Survey Phase: Theory, Threat Model & Related Works).
Your working directory is: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_theory_1
You must read the authoritative user request at: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md (specifically the section with header ## 2026-09-24T09:07:42Z).

### Mission:
Thoroughly inspect the scientific foundations, mathematical formulations, threat modeling, and related work matrices available in `BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md`, `WALKTHROUGH_FEDERATED_LUNAR.md`, `FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md`, `related_work_master.tex`, etc.

### Tasks:
1. Map out the full theoretical formulations for R2 & R3:
   - System Model: Decentralized, multi-tenant IoT edge gateways under Non-IID traffic distributions ($\mathcal{M}_i \cap \mathcal{M}_j \approx \emptyset$).
   - Threat Model: Coordinated botnet reconnaissance, stealthy multi-stage attacks (Mirai, Gafgyt), volumetric floods.
   - Defensible Research Gap: Contrast Fed-LUNAR against Autoencoder/Kitsune (reconstruction shortcutting) and LOC-NFST (catastrophic null-space erosion under streaming Non-IID drift). Why distance-ranking graph topological awareness ($k$-NN relational geometry) is uniquely suited. Transparent system trade-offs (server QP solving, $k$ hyperparameter selection vs 0.001 ms inference).
   - Theorem 1: Out-of-Distribution Distance-Ranking Inversion & Monotonicity Breakdown ($\nabla_d f_\theta(d) \le 0$ as $d \to \infty$, AUC collapse to 0.15%), and MSSP geometric spacing monotonicity proof ($\partial f_\theta / \partial d > 0$).
   - Theorem 2: Adversarial Negative Gradient Cancellation in Non-IID FL ($\mathbb{E}[\langle \nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B \rangle] < 0$, vanishing global updates).
   - Lemma 2.1: Purging Invariance via FSDS and CMNP ($\ge 1 - \delta$).
   - Lemma 2.2: Pareto Directional Convergence via DROGA.
2. Map out R5: 5 paradigms taxonomy (Signature/Rule-based, Statistical/Reconstruction, Tree/Ensemble, Spectral/Analytical, Graph/Contrastive) and LaTeX comparison matrix.
3. Verify step-by-step mathematical soundness, notation consistency, and citation anchors.
4. Write your comprehensive report to `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_theory_1\handoff.md` and send a message back when completed.
