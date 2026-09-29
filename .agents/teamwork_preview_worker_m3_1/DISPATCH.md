## 2026-09-23T02:32:34Z
You are Worker 1 for Milestone M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics).
Your Identity: teamwork_preview_worker_m3_1
Working Directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m3_1

MANDATORY INPUTS:
- User Request: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md (READ THIS FIRST)
- Project Scope: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_1\PROJECT.md
- Synthesis Plan: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_gen3\analysis.md
- Explorer 1 Report (Data Loader): d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_m3_1\handoff.md
- Explorer 2 Report (Dirichlet & Models): d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_m3_2\handoff.md
- Explorer 3 Report (Metrics & Test Harness): d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_m3_3\handoff.md

EXCLUSIVE FILE OWNERSHIP:
You own and have exclusive write permission to:
- fed_lunar/benchmark/data_loader.py
- fed_lunar/benchmark/metrics.py
- fed_lunar/benchmark/run_benchmark.py
- tests/test_benchmark_harness.py

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SPECIFIC IMPLEMENTATION TASKS:
1. Implement `fed_lunar/benchmark/data_loader.py`:
   - Full support for 4 IoT datasets: BoTIoT (D=35), EdgeIIoTset (D=42), CICIoT2023 (D=46), N_BaIoT (D=115).
   - Implement `SyntheticIoTGenerator` with deterministic multimodal normal manifolds and separated anomaly samples with test contamination <= 5%.
   - Implement `DirichletPartitioner` with 2-stage latent clustering (MiniBatchKMeans, K >= 2M, alpha=0.5) ensuring exact sample conservation (sum(n_m) == N) and non-empty client guarantees.
   - Implement `OneClassDatasetLoader` checking real CSV paths and seamlessly activating `SyntheticIoTGenerator` when files are absent. Enforce strict normal-only training data (y=0) and test contamination <= 5%.
   - Implement `partition_and_prepare_dataset(...)` matching the interface expected by `run_benchmark.py` and test harnesses.

2. Implement `fed_lunar/benchmark/metrics.py`:
   - Implement `calculate_optimal_f1_threshold(y_true, y_scores)` via Precision-Recall curve sweep.
   - Implement `calculate_detection_metrics(y_true, y_scores, threshold=None, threshold_percentile=95.0)` returning auc_roc, f1_score (=f1_optimal), f1_optimal, optimal_threshold, f1_calibrated, calibrated_threshold, far, detection_rate, precision, confusion counts (tp, fp, tn, fn). Handle NaN/inf gracefully.
   - Implement `calculate_optimization_dynamics(history, total_rounds, loss_tolerance=0.01, patience=2)` returning round_conflict_ratio, gradient_conflict_ratio, mean_cosine, convergence_rounds.
   - Implement `MemoryTracker` context manager tracking peak host RAM (tracemalloc) and GPU VRAM (torch.cuda.max_memory_allocated).
   - Implement `measure_inference_latency(model, X_test, n_runs=3, batch_size=1024, measure_single=True)` returning latency_ms_per_sample and latency_single_ms.

3. Implement `fed_lunar/benchmark/run_benchmark.py`:
   - Accept CLI options: `--dataset`, `--method` / `--models`, `--clients`, `--alpha`, `--rounds`, `--data_dir`, `--output_dir`, `--synthetic_fallback`, `--max_train_samples`, `--max_test_samples`, `--scaler`, `--device`, `--seed`, `--verbose`.
   - Support all 8 model variants: Proposed_FedLUNAR, Ablation_FedLUNAR_NoDROGA, Ablation_FedLUNAR_NoCMNP, Naive_FedLUNAR, FedAutoEncoder, FedProx_LUNAR, PCGrad_FedLUNAR, LOC_NFST_Bound.
   - Implement dual CSV logging:
     a) Per-run trajectory: `outputs/lunar_results/{dataset_name}_{method}.csv`
     b) Master summary: `outputs/lunar_results/benchmark_summary.csv` with exact 20-column schema matching PROJECT.md lines 77-79 and User Request R4:
        `dataset,method,clients,alpha,rounds,auc_roc,f1_score,f1_optimal,f1_calibrated,far,detection_rate,precision,gradient_conflict_ratio,round_conflict_ratio,convergence_rounds,latency_ms_per_sample,latency_single_ms,peak_memory_mb,train_time_sec,input_dim`
   - Print formatted Markdown summary table to stdout.

4. Implement `tests/test_benchmark_harness.py`:
   - Implement 16 test cases across 3 classes (`TestDirichletPartitioningProperties`, `TestMetricsPipelineAccuracy`, `TestEndToEndBenchmarkExecution`) as designed in Explorer 3's handoff.

5. VERIFICATION COMMANDS (Mandatory: run these in PowerShell and document exact results):
   - `pytest tests/test_benchmark_harness.py -v`
   - `pytest tests/test_baselines.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`
   - `python tests/run_e2e_tests.py`
   - `python -m fed_lunar.benchmark.run_benchmark --dataset BoTIoT --method LOC_NFST_Bound --clients 3 --alpha 0.5 --rounds 1`

OUTPUT REQUIREMENTS:
- Write your complete handoff report to `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_worker_m3_1\handoff.md`.
- Update `progress.md` in your working directory.
- Send a completion message via `send_message` to your parent once finished.
