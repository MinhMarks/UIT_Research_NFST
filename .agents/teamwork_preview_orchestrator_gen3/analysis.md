# Milestone M3 Synthesis & Implementation Architecture

## 1. Consensus Findings Across Explorers 1, 2, and 3
- **Data Loading & One-Class Protocol (`fed_lunar/benchmark/data_loader.py`)**:
  - Remote datasets are hosted on `postmaster.iec` at `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`. Local directory is empty.
  - An embedded, deterministic `SyntheticIoTGenerator` must be implemented to support local unit tests, CI test runs, and offline execution without `FileNotFoundError`.
  - Canonical dimensions: BoTIoT ($D=35$), EdgeIIoTset ($D=42$), CICIoT2023 ($D=46$), N_BaIoT ($D=115$).
  - One-Class protocol strictly requires: normal-only training data ($y=0$) and test stream contamination $\le 5\%$ ($c \le 0.05$).
  - Dirichlet partitioning: 2-stage latent manifold clustering (MiniBatchKMeans $K \ge 2M$) followed by cluster proportion Dirichlet sampling ($\alpha=0.5$). Strict sample conservation $\sum n_m = N$ with non-empty client protection.
- **Metrics Instrumentation (`fed_lunar/benchmark/metrics.py`)**:
  - Detection: AUC-ROC (%), dual F1 (`f1_score` = optimal F1 via PR-curve sweep, `f1_calibrated` at normal 95th percentile cutoff), FAR ($FP / (FP + TN) \times 100\%$), Detection Rate, Precision.
  - Optimization dynamics: `gradient_conflict_ratio` (average pairwise GCR across rounds), `round_conflict_ratio` (% rounds with conflict), `mean_cosine`, `convergence_rounds`.
  - Edge viability: `latency_ms_per_sample`, `latency_single_ms`, and `MemoryTracker` (CPU tracemalloc + GPU VRAM max).
- **Master Benchmark Runner (`fed_lunar/benchmark/run_benchmark.py`)**:
  - Full CLI flags (`--dataset`, `--method`/`--models`, `--clients`, `--alpha`, `--rounds`, `--output_dir`, etc.).
  - Saves per-run trajectory `outputs/lunar_results/{dataset_name}_{method}.csv` and master summary `outputs/lunar_results/benchmark_summary.csv` with exact 20-column schema.
- **Test Harness (`tests/test_benchmark_harness.py`)**:
  - 16 test cases across 3 classes verifying partitioning, metrics, and E2E execution of all 4 methods.

## 2. Implementation Work Assignment for Worker
Worker owns exclusively:
- `fed_lunar/benchmark/data_loader.py`
- `fed_lunar/benchmark/metrics.py`
- `fed_lunar/benchmark/run_benchmark.py`
- `tests/test_benchmark_harness.py`
Worker must run:
1. `pytest tests/test_benchmark_harness.py -v`
2. `pytest tests/test_baselines.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`
3. `python tests/run_e2e_tests.py`
4. `python -m fed_lunar.benchmark.run_benchmark --dataset BoTIoT --method LOC_NFST_Bound --clients 3 --alpha 0.5 --rounds 1` (to verify CSV generation)
