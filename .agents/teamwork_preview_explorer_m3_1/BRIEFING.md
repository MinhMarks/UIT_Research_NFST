# BRIEFING — 2026-09-23T02:30:30Z

## Mission
Investigate dataset structures, formats, and data loading mechanisms for 4 canonical IoT datasets (BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT), define the One-Class Non-IID protocol, and propose a complete design for `fed_lunar/benchmark/data_loader.py` with offline synthetic fallbacks.

## 🔒 My Identity
- Archetype: explorer
- Roles: Teamwork explorer
- Working directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_m3_1
- Original parent: 6d043925-dcfa-4d4b-9fed-52cd89b43248
- Milestone: M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or edit source code in repository.
- Strictly adhere to .agents workspace boundary: write only inside .agents/teamwork_preview_explorer_m3_1/.
- Produce a 5-component handoff report (handoff.md) with Observation, Logic Chain, Caveats, Conclusion, Verification Method.
- Send completion message to parent via send_message.

## Current Parent
- Conversation ID: 6d043925-dcfa-4d4b-9fed-52cd89b43248
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `DataProcessing/generate_oc_datasets.py`, `DataProcessing/test_oc_datasets.py`
  - `DataProcessing/BoTIoT.py`, `DataProcessing/EdgeIIoTset.py`, `DataProcessing/CICIoT2023.py`, `DataProcessing/N_BaIoT.py`
  - `check_dataset_stats.py`, `fed_lunar/benchmark/data_loader.py`, `fed_lunar/benchmark/run_benchmark.py`, `fed_lunar/benchmark/metrics.py`
  - `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier2_boundaries.py`, `tests/e2e/test_tier4_applications.py`, `tests/e2e/contract_stubs.py`
- **Key findings**:
  - Remote server path `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/` stores pre-scaled CSVs named `Train_{scaler}_data_{dataset}.csv` and `Test_{scaler}_data_{dataset}.csv`.
  - Locally, `Datascaled/Official_OC_Data/` is empty. Existing `data_loader.py` lacks an offline synthetic fallback, causing `FileNotFoundError` in local runs.
  - Dimension and domain specifics: BoTIoT (35 features, smart home DoS/Recon/Theft), EdgeIIoTset (42 features, industrial Modbus/MQTT 14 attacks), CICIoT2023 (46 features, 105 devices volumetric floods), N_BaIoT (115 features, 9 commercial devices botnet 23 stats x 5 windows).
  - One-Class protocol: Train set contains strictly normal telemetry ($y=0$). Test stream anomaly contamination must be bounded to $\le 5\%$ (in pre-scaled files on disk, test anomaly ratio is 50%, requiring subsampling to $\le 5\%$).
  - Dirichlet Partitioner: Latent cluster or direct Dirichlet sample allocation. Empty client handling in current `data_loader.py` can violate sample conservation unless samples are transferred from max-capacity client.
- **Unexplored areas**: Milestone M4 remote deployment scripts and GPU execution on RTX 5090.

## Key Decisions Made
- Architecture of `fed_lunar/benchmark/data_loader.py` must include:
  1. `SyntheticIoTGenerator` providing deterministic 35D, 42D, 46D, and 115D clustered data when local files are absent.
  2. Automatic transparent fallback in `OneClassDatasetLoader.load_dataset` with user warning.
  3. Contamination enforcement routine: `enforce_test_contamination(..., contamination_ratio=0.05)`.
  4. Preserving exact sample conservation $\sum n_m = N$ in `DirichletPartitioner`.

## Artifact Index
- DISPATCH.md — Dispatch log
- BRIEFING.md — Persistent situational awareness memory
- progress.md — Heartbeat and execution status
- handoff.md — Final 5-component handoff report
