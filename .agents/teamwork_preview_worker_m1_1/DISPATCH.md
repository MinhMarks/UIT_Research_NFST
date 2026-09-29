# Dispatch: Worker 1 (Milestone M1: Core Fed-LUNAR Engine & Algorithms)
Target: Implement PyTorch LUNAR Distance-Ranking MLP, Cross-Manifold Negative Purging (CMNP) with Federated Subspace Density Sketches (FSDS), and Distance-Ranking Orthogonal Gradient Alignment (DROGA with DR-PCGrad and DR-CAGrad).

## 2026-09-22T22:43:11Z
You are Worker 1 for Milestone M1 (Core Fed-LUNAR Engine & Algorithms).
Working Directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m1_1
Original Request Path: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md
Scope Document: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_1\PROJECT.md
Explorer 3 Report (Mathematical Specifications & Algorithms): d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_3\report.md
Explorer 1 Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_1\report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INSTRUCTIONS:
1. Read ORIGINAL_REQUEST.md and the Explorer 3 Report first.
2. File Ownership: You exclusively own:
   - fed_lunar/__init__.py
   - fed_lunar/models/__init__.py
   - fed_lunar/models/lunar_mlp.py
   - fed_lunar/models/negative_gen.py
   - fed_lunar/models/autoencoder.py
   - fed_lunar/federated/__init__.py
   - fed_lunar/federated/client.py
   - fed_lunar/federated/strategy.py
   - fed_lunar/federated/sketches.py
   - tests/test_lunar_model.py
   - tests/test_cmnp_purging.py
   - tests/test_droga_alignment.py
3. Implementation Tasks:
   a. Implement PyTorch LUNAR Distance-Ranking MLP (`LUNAR_MLP`) in `fed_lunar/models/lunar_mlp.py` with k-NN distance ranking feature extractor and binary cross-entropy distance ranking loss.
   b. Implement Subspace Pseudo-Negative Generator with Cross-Manifold Negative Purging (CMNP) in `fed_lunar/models/negative_gen.py` and Federated Subspace Density Sketches (FSDS: $\mu_c, \Lambda_c, U_c, r_c^{\max}$) in `fed_lunar/federated/sketches.py` per Explorer 3 Report Algorithm 1 and Theorem 2.
   c. Implement Distance-Ranking Orthogonal Gradient Alignment (DROGA) in `fed_lunar/federated/strategy.py` with:
      - DR-PCGrad: pairwise gradient conflict projection ($\langle g_i, g_j \rangle < 0 \implies g_i = g_i - \frac{\langle g_i, g_j \rangle}{\|g_j\|^2} g_j$)
      - DR-CAGrad: conflict-averse gradient descent solving the dual simplex QP ($\min_\alpha \frac{1}{2} \|g_0 + \sum_i \alpha_i g_i\|^2$ s.t. $\alpha \ge 0, \sum \alpha_i = c \frac{\|g_0\|}{\max_i \|g_i\|}$)
      - Exact computation and logging of gradient conflict ratio (% conflicting gradient pairs with cosine < 0).
   d. Implement `SimpleAutoEncoder` in `fed_lunar/models/autoencoder.py` for downstream Tier 2 baseline.
   e. Implement client training logic in `fed_lunar/federated/client.py` coordinating local updates, negative generation, FSDS sketch export, and gradient computation.
   f. Write comprehensive unit tests in `tests/test_lunar_model.py`, `tests/test_cmnp_purging.py`, `tests/test_droga_alignment.py`.
4. Verification:
   Run the tests using python / pytest on the local environment. Ensure all tests pass.
5. Deliverables:
   Write a comprehensive report at `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m1_1\report.md` and handoff at `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m1_1\handoff.md`.
   Report completion to parent via send_message.
