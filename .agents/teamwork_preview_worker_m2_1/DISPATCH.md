## 2026-09-22T23:02:58Z

You are Worker 1 for Milestone M2 (3-Tier Baseline Hierarchy).
Working Directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m2_1
Original Request Path: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md
Scope Document: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_1\PROJECT.md
Explorer 1 Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_1\report.md
Explorer 3 Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_3\report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INSTRUCTIONS:
1. Read ORIGINAL_REQUEST.md and PROJECT.md first.
2. Implement the standardized 3-Tier Baseline Hierarchy in `fed_lunar/baselines/`:
   - Feature F5 (Tier 1): `fed_lunar/baselines/naive_lunar.py`
     * Class `NaiveFedLunar`: Standard FedAvg on LUNAR MLP weights with uncoordinated local subspace perturbation (NO CMNP filter, NO DROGA gradient projection).
     * Methods: `fit(client_train_data, rounds, ...)`, `predict_proba(X)`, `decision_function(X)`.
   - Feature F6 (Tier 2): `fed_lunar/baselines/fed_ae.py`
     * Class `FedAutoEncoder`: Federated Deep Autoencoder baseline using `SimpleAutoEncoder` (`fed_lunar/models/autoencoder.py`).
     * MSE reconstruction loss on normal client data, FedAvg aggregation of encoder/decoder weights.
     * Anomaly score: reconstruction error $\|x - \hat{x}\|_2^2$.
   - Feature F7 (Tier 2): `fed_lunar/baselines/fedprox_lunar.py`
     * Class `FedProxLunar`: Federated LUNAR with proximal regularization $\frac{\mu}{2}\|\theta - \theta_t\|^2$ added to client distance-ranking loss to counter client drift.
     * Class `PCGradFedLunar`: Federated LUNAR using standard PCGrad aggregation (Yu et al., NeurIPS 2020) without FSDS sketches or CMNP.
   - Feature F8 (Tier 3): `fed_lunar/baselines/loc_nfst_bound.py`
     * Class `LOC_NFST_Bound`: Closed-form Null-Space analytical baseline ($T=1$) serving as theoretical upper bound, integrating / referencing the exact Null-Space projection method from `OC_NFST_memory_optimized.py`.
     * Computes null-space projection matrix $P_N$ on benign traffic and computes anomaly scores as $\|P_N x\|_2^2$.
3. Fix issues identified during Milestone M1 verification:
   - In `fed_lunar/models/negative_gen.py` line 281: Fix `_fallback_boundary_noise` so that anchor indices maintain 1-to-1 correspondence without index mismatch, fixing the flaky `test_subspace_negative_generator_fallback`.
   - In `fed_lunar/federated/strategy.py`: Add unit-norm gradient scaling $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ prior to Gram matrix computation in `dr_cagrad` (and optionally `dr_pcgrad`) to resolve scale-disparity distortion (as proven by Challenger 2).
4. Unit Tests:
   - Create `tests/test_baselines.py` covering all 4 baseline classes (`NaiveFedLunar`, `FedAutoEncoder`, `FedProxLunar`, `PCGradFedLunar`, `LOC_NFST_Bound`).
   - Run `pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`.
   - Ensure all existing and new unit tests pass with 100% success rate.
5. Write your handoff report to `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m2_1\handoff.md` with:
   - Observation: List of implemented files, baseline classes, bug fixes.
   - Logic Chain: Design choices, loss functions, aggregation methods.
   - Caveats: Any edge cases or hyperparameter sensitivities.
   - Conclusion: Verification summary.
   - Verification Method: Exact commands and outputs.
6. Send completion message back to parent via send_message.
