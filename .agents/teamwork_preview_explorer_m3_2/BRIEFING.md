# BRIEFING — 2026-09-23T02:32:00Z

## Mission
Deep technical investigation of Non-IID Dirichlet Partitioning, Baseline Integration (Fed-LUNAR proposed + 3-Tier baselines), and Benchmark Harness architecture for Milestone M3.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Technical Investigator, System Architect
- Working directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_m3_2
- Original parent: 6d043925-dcfa-4d4b-9fed-52cd89b43248
- Milestone: M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or edit source code outside .agents
- Write findings and proposed implementation strategy to handoff.md
- Maintain progress.md heartbeat
- Keep BRIEFING.md updated under ~100 lines

## Current Parent
- Conversation ID: 6d043925-dcfa-4d4b-9fed-52cd89b43248
- Updated: 2026-09-23T02:26:37Z

## Investigation State
- **Explored paths**:
  - `fed_lunar/benchmark/data_loader.py` (DirichletPartitioner, OneClassDatasetLoader, partition_and_prepare_dataset)
  - `fed_lunar/benchmark/run_benchmark.py` (CLI runner, instantiate_model, logging loop)
  - `fed_lunar/benchmark/metrics.py` (detection metrics, inference latency)
  - `fed_lunar/baselines/` (naive_lunar.py, fed_ae.py, fedprox_lunar.py, loc_nfst_bound.py)
  - `fed_lunar/federated/` (fed_lunar.py, strategy.py, sketches.py)
  - `tests/` (test_baselines.py, test_cmnp_purging.py, test_droga_alignment.py)
- **Key findings**:
  - Formulated 2-stage Dirichlet cluster-skew partition for continuous One-Class data: unsupervised latent clustering (K-Means/GMM) + Dirichlet(alpha=0.5) proportion sampling.
  - Proved causal mechanism linking Dirichlet cluster-skew to adversarial negative gradient cancellation.
  - Verified 3-Tier baseline hierarchy; all 6 baseline tests and 11 CMNP/DROGA tests pass cleanly.
  - Specified exact architecture for `run_benchmark.py` including dual CSV logging, CLI options (`--dataset`, `--method`/`--models`, `--clients`, `--alpha`, etc.), and edge resource tracking.
- **Unexplored areas**: None within M3-2 scope.

## Key Decisions Made
- Standardize on `MiniBatchKMeans` with $K = \max(5, 2M)$ for latent manifold segmentation before Dirichlet allocation.
- Require dual CSV output: both individual `outputs/lunar_results/{dataset}_{method}.csv` and global summary `benchmark_summary.csv`.
- Include `tracemalloc` (CPU) and `torch.cuda.max_memory_allocated` (GPU) for peak memory tracking.

## Artifact Index
- `DISPATCH.md` — Recorded instructions
- `BRIEFING.md` — Situational awareness
- `progress.md` — Liveness heartbeat
- `handoff.md` — Comprehensive technical investigation report (5 sections: Observation, Logic Chain, Caveats, Conclusion, Verification Method)
