# Plan: Orchestrator Gen 3 Execution

## Objectives
1. Milestone M2 Gate Certification: COMPLETE (Recorded in GATE_STATUS.md & PROJECT.md).
2. Milestone M3: Implement Non-IID Dirichlet IoT Benchmark Harness (`fed_lunar/benchmark/`) across 4 canonical IoT datasets (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT) with comprehensive metrics logging.
3. Milestone M4: Remote execution on server `postmaster.iec` via `/opt/tljh/user/bin/python3`, create/push git branch `feature/federated-lunar-novel`, output CSV results to `outputs/lunar_results/`.
4. Milestone M5: Deliver publication-grade walkthrough report `WALKTHROUGH_FEDERATED_LUNAR.md` with ablation study, theoretical implications, and genuine peer-reviewed citations with verified DOIs.
5. Notify Sentinel upon meeting all acceptance criteria for final Victory Audit.

## Detailed Phase Breakdown

### Phase 1: Milestone M2 Gate Certification [COMPLETED]
- [x] Verified 37 unit tests and 126/126 E2E tests passing.
- [x] Recorded Gate PASS in `GATE_STATUS.md`.
- [x] Updated `PROJECT.md` milestone status to DONE for M2 and IN_PROGRESS for M3.

### Phase 2: Milestone M3 Execution (Non-IID Benchmark Harness & Metrics)
- [ ] Step 2.1: Dispatch Explorer (`teamwork_preview_explorer`) to inspect `fed_lunar/benchmark/`, analyze dataset structures, Dirichlet partitioner specifications ($\alpha=0.5$, contamination $\le 5\%$), metrics logging (AUC-ROC, F1, FAR, gradient conflict ratio, rounds, latency ms, peak memory MB), and design unit test harness.
- [ ] Step 2.2: Dispatch Worker (`teamwork_preview_worker`) with Explorer findings to implement:
  - `fed_lunar/benchmark/data_loader.py`
  - `fed_lunar/benchmark/metrics.py`
  - `fed_lunar/benchmark/run_benchmark.py`
  - `tests/test_benchmark_harness.py`
- [ ] Step 2.3: Dispatch 2 Reviewers (`teamwork_preview_reviewer`) to evaluate correctness, robustness, and interface conformance.
- [ ] Step 2.4: Dispatch 2 Challengers (`teamwork_preview_challenger`) to stress-test the partitioner and metrics calculations.
- [ ] Step 2.5: Dispatch Forensic Auditor (`teamwork_preview_auditor`) to ensure zero mock/hardcoding.
- [ ] Step 2.6: Gate check for Milestone M3.

### Phase 3: Milestone M4 Execution (Remote Server Execution & Git Branch)
- [ ] Step 3.1: Create and push git branch `feature/federated-lunar-novel`.
- [ ] Step 3.2: Dispatch Worker to run remote benchmark on server `postmaster.iec` using `/opt/tljh/user/bin/python3` across the 4 datasets from `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`.
- [ ] Step 3.3: Collect and verify output CSVs in `outputs/lunar_results/`.
- [ ] Step 3.4: Review and audit M4 outputs.

### Phase 4: Milestone M5 Execution (Walkthrough Documentation & Final Verification)
- [ ] Step 4.1: Dispatch Explorer / Worker to compile `WALKTHROUGH_FEDERATED_LUNAR.md` with:
  - Algorithmic foundations and mathematical formulations (Prop 1 & Thm 2).
  - Empirical benchmark results across 4 datasets and 3-tier baselines.
  - Ablation studies (CMNP vs DROGA vs baseline).
  - Theoretical implications for Non-IID FL-IDS.
  - Verified peer-reviewed citations with DOIs.
- [ ] Step 4.2: Final Review and Forensic Audit.
- [ ] Step 4.3: Final handoff and completion message to Sentinel.
