# Dispatch: Survey Explorer 3 (Mathematical Formulations & Algorithmic Foundations)
Target: Formulate exact gradient conflict dynamics, cross-manifold negative purging, and orthogonal gradient alignment for distance-ranking LUNAR.

## 2026-09-22T15:50:33Z
You are Explorer 3 (Mathematical Formulations & Algorithmic Foundations).
Working Directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_3
Original Request Path: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md

MANDATORY INSTRUCTIONS:
1. Read ORIGINAL_REQUEST.md first.
2. Review relevant literature and design the mathematical formulation for R1:
   - LUNAR foundation: Local Outlier Factor / Nearest Neighbor distance-ranking neural network, pseudo-negative generation via uniform/Gaussian subspace perturbation around normal points.
   - Exact mathematical formulation of the gradient conflict dynamics in Federated Learning under Non-IID client distributions:
     * Let Client A have normal manifold M_A, Client B have normal manifold M_B where M_A \cap M_B \approx \emptyset or shifted.
     * When Client A generates pseudo-negatives \tilde{x}_A around M_A, some pseudo-negatives fall directly onto or near Client B's normal manifold M_B (Cross-Manifold Negative Intrusion).
     * Show why this causes antagonistic loss gradients: \langle \nabla \mathcal{L}_A(\theta), \nabla \mathcal{L}_B(\theta) \rangle < 0, leading to gradient cancellation, oscillatory updates, and representation collapse in standard FedAvg.
   - Novel algorithm design:
     * Component 1: Cross-Manifold Negative Purging (rejecting/weighting pseudo-negatives that intrude on shared/other client representation subspaces, inspired by debiased contrastive learning).
     * Component 2: Orthogonal Gradient Alignment (adapting PCGrad / Projecting Conflicting Gradients and CAGrad to the distance-ranking loss of LUNAR).
3. Draft the formal mathematical writeup, theorem/proposition statements, and step-by-step algorithmic pseudocode.
4. Output your findings and mathematical specification at `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_3\report.md`.
5. Send a completion message via send_message to parent with summary and file path.
