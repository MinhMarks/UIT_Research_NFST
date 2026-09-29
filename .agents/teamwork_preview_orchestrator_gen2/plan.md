# Execution Plan: Project Orchestrator Gen 2

## Objective
Resume and complete the Federated LUNAR project across Milestones M2, M3, M4, and M5, ensuring all theoretical, empirical, git, and documentation acceptance criteria are met.

## Milestone Roadmap

### Milestone M2: 3-Tier Baseline Hierarchy (Remediation & Gate Clearance)
- **Task**: Remediate interface contract regressions in `fed_lunar/baselines/loc_nfst_bound.py` and tuple support in baseline formatters.
- **Worker Dispatch**:
  - `null_basis` and `threshold` properties in `LOC_NFST_Bound`.
  - `score = decision_function` alias across all baseline classes (`NaiveFederatedLUNAR`, `FedAvgAutoEncoder`, `FedProxLUNAR`, `PCGradLUNAR`, `LOC_NFST_Bound`).
  - `fit()` signature kwargs and tolerance handling.
  - `full_matrices=True` in SVD when rank < D for orthogonal complement null projection.
  - Tuple input support in `_format_client_data`.
  - Verification: Run `pytest tests/test_baselines.py tests/test_m2_math_verification.py -v` (34/34 passing) and `python tests/run_e2e_tests.py` (126/126 passing).
- **Review & Audit**:
  - 2 independent Reviewers (`teamwork_preview_reviewer`).
  - 1 Forensic Auditor (`teamwork_preview_auditor`).
- **Gate Check**: Require 100% APPROVE and CLEAN.

### Milestone M3: Non-IID IoT Benchmark Harness & Metrics Pipeline
- **Task**: Implement robust benchmark infrastructure for 4 canonical datasets (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT) under Dirichlet Non-IID ($\alpha=0.5$) with test contamination $\le 5\%$.
- **Components**:
  - `fed_lunar/benchmark/data_loader.py`: Dataset loader supporting local/remote data paths, Dirichlet splitting, and One-Class contamination control.
  - `fed_lunar/benchmark/metrics.py`: Calculation of AUC-ROC, F1, FAR, gradient conflict ratio, convergence rounds, per-sample inference latency, and peak memory.
  - `fed_lunar/benchmark/run_benchmark.py`: Unified CLI runner outputting structured CSVs to `outputs/lunar_results/{dataset}_{method}.csv`.
- **Review & Audit**: Reviewers + Auditor verification.

### Milestone M4: Remote Server Execution & Git Branch Publication
- **Task**: Git branch management and remote server benchmark execution on `postmaster.iec`.
- **Components**:
  - Create and push branch `feature/federated-lunar-novel` with clean commits.
  - Execute benchmark runs on `postmaster.iec` using `/opt/tljh/user/bin/python3` on datasets in `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`.
  - Pull or verify CSV results in `outputs/lunar_results/`.
- **Review & Audit**: Reviewers verify remote logs, CSV metrics, and git status.

### Milestone M5: Walkthrough Report & Final Victory Audit
- **Task**: Synthesize comprehensive documentation and conduct final audit.
- **Components**:
  - Generate `WALKTHROUGH_FEDERATED_LUNAR.md` with problem formulation, theoretical derivations, empirical comparison tables, ablation study, and peer-reviewed citations with verified DOIs.
  - Final comprehensive Forensic Audit (`teamwork_preview_auditor`) verifying zero hardcoding, real execution, and full fulfillment of requirements R1-R4.
  - Final Gate clearance and report victory to parent.
