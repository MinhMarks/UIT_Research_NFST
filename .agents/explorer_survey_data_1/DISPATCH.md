## 2026-09-24T09:15:10Z

<USER_REQUEST>
You are an Explorer subagent (Survey Phase: Empirical Results & Benchmarks).
Your working directory is: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_data_1
You must read the authoritative user request at: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md (specifically the section with header ## 2026-09-24T09:07:42Z).

### Mission:
Thoroughly inspect and extract the exact numerical results and experimental logs in `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\outputs\lunar_results\` and existing analytical reports (`BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md`, `WALKTHROUGH_FEDERATED_LUNAR.md`).

### Tasks:
1. Examine `outputs/lunar_results/benchmark_summary.csv` and any other CSVs in `outputs/lunar_results/`.
2. Extract exact numbers for:
   - Master Benchmark: Fed-LUNAR vs Naive Fed-LUNAR, FedAutoEncoder, FedProx-LUNAR, PCGrad-FedLUNAR, LOC-NFST across BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT on AUC-ROC, F1-Score, and FAR.
   - Dirichlet Non-IID Sensitivity Sweep: $\alpha \in \{0.1, 0.5, 1.0, 5.0\}$ (from `outputs/lunar_results/sensitivity_sweep/alpha_sensitivity_summary.csv` or similar).
   - Ablation Studies: Isolating MSSP, CMNP (`NoCMNP` drop: -26.23% F1), and DROGA (`NoDROGA` drop: -1.36% F1).
   - Edge Feasibility & Resource Profiling: Per-sample inference latency (0.001 ms), memory footprint (RAM/VRAM), and bandwidth overhead (<5 KB FSDS sketches).
3. Verify that all extracted numbers strictly originate from genuine files in the repository (Zero Hallucination invariant).
4. Draft the exact LaTeX tabular formats for all 4 tables required by Requirement R4.
5. Write your comprehensive report to `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_data_1\handoff.md` and send a message back when completed.
</USER_REQUEST>
