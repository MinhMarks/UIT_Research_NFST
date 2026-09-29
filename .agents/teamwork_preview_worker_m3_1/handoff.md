# Hard Handoff Report: Milestone M3 Benchmark Harness & Metrics

**Agent**: `teamwork_preview_worker_m3_1`  
**Milestone**: M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics)  
**Parent**: `6d043925-dcfa-4d4b-9fed-52cd89b43248`  
**Date**: 2026-09-23T03:15:00Z  

---

## 1. Observation

### Code Implementations
The following 4 files were implemented under exclusive file ownership without modifying any external files:
1. `fed_lunar/benchmark/data_loader.py` (338 lines)
   - Defines configurations for 4 IoT datasets: `BoTIoT` ($D=35$), `EdgeIIoTset` ($D=42$), `CICIoT2023` ($D=46$), `N_BaIoT` ($D=115$).
   - Implements `SyntheticIoTGenerator` generating multi-modal normal clusters and out-of-distribution attacks when physical files are absent.
   - Implements `DirichletPartitioner` guaranteeing exact sample conservation ($\sum_{m=1}^M n_m = N$), non-empty partitions ($n_m \ge n_{\min}$), and strict contamination constraints ($\text{attack\_ratio} \le 5\%$).
   - Implements `OneClassDatasetLoader` with `StandardScaler` / `MinMaxScaler` fitting on normal training data only.
   - Implements helper `partition_and_prepare_dataset(dataset_name, data_dir, n_clients, alpha, ...)`.

2. `fed_lunar/benchmark/metrics.py` (356 lines)
   - Implements `calculate_optimal_f1_threshold(y_true, scores)` sweeping Precision-Recall curve thresholds to maximize F1.
   - Implements `calculate_detection_metrics(y_true, scores, threshold=None)` computing AUC-ROC, optimal F1, calibrated F1 (95th percentile of normal scores), False Alarm Rate (FAR), Detection Rate (DR), and Precision.
   - Implements `calculate_optimization_dynamics(pairwise_cosine_similarities)` computing Gradient Conflict Ratio (GCR) and Round Conflict Ratio (RCR).
   - Implements `MemoryTracker` tracking peak RAM (via `psutil`) and GPU VRAM (via `torch.cuda`).
   - Implements `measure_inference_latency(model, sample_batch, n_runs=50)` measuring both batched throughput (ms/sample) and single-sample streaming latency with CUDA synchronization.
   - Implements `MetricsLogger` maintaining backwards compatibility with end-to-end contract stubs.

3. `fed_lunar/benchmark/run_benchmark.py` (427 lines)
   - CLI runner supporting arguments `--dataset`, `--method`, `--clients`, `--alpha`, `--rounds`, `--scaler`, `--output_dir`, `--seed`, `--device`.
   - Supports 8 model variants:
     - `Proposed_FedLUNAR`: DROGA alignment + CMNP purging.
     - `Ablation_FedLUNAR_NoDROGA`: Pure FedAvg + CMNP purging.
     - `Ablation_FedLUNAR_NoCMNP`: DROGA alignment without CMNP purging.
     - `Naive_FedLUNAR`: Standard FedAvg on LUNAR client models.
     - `FedAutoEncoder`: Federated reconstruction error baseline.
     - `FedProx_LUNAR`: FedProx with proximal regularization term ($\mu=0.01$).
     - `PCGrad_FedLUNAR`: Gradient projection resolving pairwise negative cosine conflicts.
     - `LOC_NFST_Bound`: Centralized / localized theoretical upper-bound baseline.
   - Dual CSV persistence:
     - Master summary CSV `outputs/lunar_results/benchmark_summary.csv` matching the exact 20-column schema from `PROJECT.md`.
     - Per-run round-by-round trajectory CSV `outputs/lunar_results/{dataset}_{method}.csv`.
   - Rich Markdown summary table printed to console via `tabulate`.

4. `tests/test_benchmark_harness.py` (321 lines)
   - 16 test cases across 3 test classes:
     - `TestDirichletPartitioningProperties`: Tests sample conservation, partition non-emptiness, Dirichlet skew variance across alpha (0.1 vs 100.0), normal-only training guarantees ($y=0$), attack test contamination $\le 5\%$, and multi-dataset dimension compliance.
     - `TestMetricsPipelineAccuracy`: Tests perfect vs inverted detection scores, edge cases (all normal, zero attack), optimal vs calibrated F1 behavior, GCR & RCR calculation, and MemoryTracker & Latency measuring.
     - `TestEndToEndBenchmarkExecution`: Tests fast end-to-end run of `Proposed_FedLUNAR`, baseline `FedAutoEncoder`, `LOC_NFST_Bound`, 20-column CSV schema conformity, and CLI entry point invocation via subprocess.

### Verbatim Tool Command Results

1. `pytest tests/test_benchmark_harness.py -v`:
   ```
   ============================= test session starts =============================
   platform win32 -- Python 3.11.6, pytest-9.0.3, pluggy-1.6.0
   rootdir: D:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST
   collected 16 items

   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_sample_conservation_across_clients PASSED [  6%]
   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_all_clients_receive_non_empty_partitions PASSED [ 12%]
   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_dirichlet_skew_variance PASSED [ 18%]
   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_train_set_is_strictly_normal PASSED [ 25%]
   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_test_set_contamination_ratio PASSED [ 31%]
   tests\test_benchmark_harness.py::TestDirichletPartitioningProperties::test_multi_dataset_dimension_compliance PASSED [ 37%]
   tests\test_benchmark_harness.py::TestMetricsPipelineAccuracy::test_perfect_and_inverted_detection_metrics PASSED [ 43%]
   tests\test_benchmark_harness.py::TestMetricsPipelineAccuracy::test_edge_case_all_normal_zero_division PASSED [ 50%]
   tests\test_benchmark_harness.py::TestMetricsPipelineAccuracy::test_optimal_vs_calibrated_f1 PASSED [ 56%]
   tests\test_benchmark_harness.py::TestMetricsPipelineAccuracy::test_optimization_dynamics_gcr_rcr PASSED [ 62%]
   tests\test_benchmark_harness.py::TestMetricsPipelineAccuracy::test_memory_tracker_and_latency PASSED [ 68%]
   tests\test_benchmark_harness.py::TestEndToEndBenchmarkExecution::test_fast_e2e_proposed_fedlunar PASSED [ 75%]
   tests\test_benchmark_harness.py::TestEndToEndBenchmarkExecution::test_fast_e2e_baseline_fedautoencoder PASSED [ 81%]
   tests\test_benchmark_harness.py::TestEndToEndBenchmarkExecution::test_fast_e2e_loc_nfst_bound PASSED [ 87%]
   tests\test_benchmark_harness.py::TestEndToEndBenchmarkExecution::test_csv_schema_conformity PASSED [ 93%]
   tests\test_benchmark_harness.py::TestEndToEndBenchmarkExecution::test_cli_execution_subprocess PASSED [100%]

   ============================= 16 passed in 21.40s =============================
   ```

2. `pytest tests/test_baselines.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`:
   ```
   ============================= test session starts =============================
   collected 17 items

   tests\test_baselines.py::test_fed_autoencoder_training PASSED             [  5%]
   tests\test_baselines.py::test_fedprox_training PASSED                    [ 11%]
   tests\test_baselines.py::test_pcgrad_training PASSED                     [ 17%]
   tests\test_baselines.py::test_naive_fedlunar_training PASSED             [ 23%]
   tests\test_baselines.py::test_loc_nfst_bound PASSED                      [ 29%]
   tests\test_baselines.py::test_baseline_factory PASSED                    [ 35%]
   tests\test_cmnp_purging.py::test_cmnp_purges_malicious_client PASSED     [ 41%]
   tests\test_cmnp_purging.py::test_cmnp_preserves_honest_clients PASSED    [ 47%]
   tests\test_cmnp_purging.py::test_cmnp_with_server PASSED                 [ 52%]
   tests\test_cmnp_purging.py::test_cmnp_rejection_history PASSED           [ 58%]
   tests\test_cmnp_purging.py::test_cmnp_all_malicious_fallback PASSED      [ 64%]
   tests\test_droga_alignment.py::test_droga_resolves_conflict PASSED       [ 70%]
   tests\test_droga_alignment.py::test_droga_orthogonal_gradients PASSED    [ 76%]
   tests\test_droga_alignment.py::test_droga_no_conflict PASSED             [ 82%]
   tests\test_droga_alignment.py::test_droga_with_server PASSED             [ 88%]
   tests\test_droga_alignment.py::test_droga_gradient_conflict_ratio PASSED [ 94%]
   tests\test_droga_alignment.py::test_droga_identical_gradients PASSED     [100%]

   ============================= 17 passed in 26.09s =============================
   ```

3. `python tests/run_e2e_tests.py`:
   ```
   ============================================================
     Fed-LUNAR Framework - End-to-End Test Suite Execution
   ============================================================
   Discovered 4 test tiers:
     Tier 1: 70 tests
     Tier 2: 40 tests
     Tier 3: 11 tests
     Tier 4: 5 tests
   Total: 126 tests across 4 tiers
   ------------------------------------------------------------
   [PASS] TIER 1: Core Feature Verification (70/70)
   [PASS] TIER 2: Failure Injection & Robustness (40/40)
   [PASS] TIER 3: Cross-Feature Integration (11/11)
   [PASS] TIER 4: End-to-End Scientific Validation (5/5)
   ============================================================
     ALL TIERS PASSED (126/126 tests)
     Duration: 40.95s
   ============================================================
   ```

4. `python -m fed_lunar.benchmark.run_benchmark --dataset BoTIoT --method LOC_NFST_Bound --clients 3 --alpha 0.5 --rounds 1`:
   - Exited with code 0.
   - Generated `outputs/lunar_results/benchmark_summary.csv` with 20 columns:
     `dataset,method,clients,alpha,rounds,auc_roc,f1_score,f1_optimal,f1_calibrated,far,detection_rate,precision,gradient_conflict_ratio,round_conflict_ratio,convergence_rounds,latency_ms_per_sample,latency_single_ms,peak_memory_mb,train_time_sec,input_dim`
     Row values: `BoTIoT,LOC_NFST_Bound,3,0.5,1,97.25,76.92,76.92,63.89,5.05,92.0,48.94,0.0,0.0,1,0.0068,0.0274,86.94,1.94,35`
   - Generated `outputs/lunar_results/BoTIoT_LOC_NFST_Bound.csv` with per-round trajectory details.

---

## 2. Logic Chain

1. **Synthetic & Real IoT Data Pipeline**:
   - The user specified 4 IoT benchmark datasets (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT) with exact dimensions (35, 42, 46, 115).
   - In offline test environments without pre-downloaded gigabyte-scale raw CSVs, `SyntheticIoTGenerator` creates genuine Gaussian-mixture normal data and perturbed out-of-distribution attack anomalies.
   - `OneClassDatasetLoader` fits scalers exclusively on benign training data ($y=0$), preventing data leakage.
   - The Dirichlet partitioner distributes indices according to $\text{Dir}(\alpha)$, while guaranteeing sample conservation ($\sum n_m = N$) and non-empty client subsets via small residual reallocation.

2. **Accurate Metrics & Calibration**:
   - Optimal F1 is derived by sweeping 200 threshold points across the Precision-Recall curve.
   - Calibrated F1 uses the 95th percentile of normal validation scores to simulate realistic zero-positive operating conditions.
   - GCR and RCR accurately measure negative cosine alignment across client update vectors.
   - `MemoryTracker` captures both process RSS memory and CUDA allocated memory.

3. **Benchmarking Harness & Backward Compatibility**:
   - `run_benchmark.py` instantiates all 8 model variants cleanly, leveraging the existing baseline factory (`BaselineFactory`), DROGA server (`FedLUNARServer`), and FedAvg servers.
   - The LUNAR MLP perturbation radius `sigma_pert` was configured to `1.0` (as unscaled benign cluster distances are $\approx 2.5$), ensuring pseudo-negative anomaly points are placed outside the normal cluster boundary.
   - Adding `MetricsLogger` to `metrics.py` preserved complete compatibility with existing test assertions (`test_tier1_features.py`).

4. **Conclusion Support**:
   - All 16 unit, property, and integration tests pass cleanly in `tests/test_benchmark_harness.py`.
   - Regression suites for baselines, CMNP, and DROGA pass with 17/17 tests.
   - The entire 126-test 4-tier E2E framework passes with 100% success.
   - The CLI runner executes and produces both the 20-column summary CSV and trajectory CSV files.

---

## 3. Caveats

- When evaluated on tiny datasets ($N \le 4$), the 95th percentile normal threshold can yield a non-zero FAR due to discrete rank quantiles. Passing an explicit threshold parameter (e.g., $0.5$) resolves this behavior, as verified in `test_perfect_and_inverted_detection_metrics`.
- If physical datasets are downloaded into `Datascaled/Official_OC_Data`, `OneClassDatasetLoader` automatically prefers real CSV files over synthetic fallback.

---

## 4. Conclusion

Milestone M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics) is fully completed and verified:
- All required modules (`data_loader.py`, `metrics.py`, `run_benchmark.py`) and test suite (`test_benchmark_harness.py`) are implemented with genuine logic, strict sample conservation, and exact metric formulations.
- Zero regressions were introduced into prior milestones (M1 & M2 baselines and components remain 100% passing).
- Both CSV and Markdown reporting conform strictly to the 20-column schema defined in `PROJECT.md`.

---

## 5. Verification Method

To independently verify the implementation:

1. **Harness Test Suite**:
   ```bash
   pytest tests/test_benchmark_harness.py -v
   ```
   *Expected: 16 passed.*

2. **Milestone Regression Suites**:
   ```bash
   pytest tests/test_baselines.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v
   ```
   *Expected: 17 passed.*

3. **Full End-to-End Test Suite**:
   ```bash
   python tests/run_e2e_tests.py
   ```
   *Expected: 126 passed across all 4 tiers.*

4. **CLI Benchmark Runner**:
   ```bash
   python -m fed_lunar.benchmark.run_benchmark --dataset BoTIoT --method LOC_NFST_Bound --clients 3 --alpha 0.5 --rounds 1
   ```
   *Expected: Clean completion, console markdown summary table, and CSV files in `outputs/lunar_results/` (`benchmark_summary.csv` with 20 columns).*
