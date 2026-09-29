# BRIEFING — 2026-09-24T16:25:30+07:00

## Mission
Thoroughly inspect and extract the exact numerical results and experimental logs in `outputs/lunar_results/` and existing reports, verify zero-hallucination compliance, and draft 4 publication-grade LaTeX tables for Requirement R4.

## 🔒 My Identity
- Archetype: explorer
- Roles: Empirical Results & Benchmarks Survey
- Working directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_data_1
- Original parent: aefc8a47-d86d-4238-85b0-c8080de54f82
- Milestone: Paper Survey Phase

## 🔒 Key Constraints
- Read-only investigation — do NOT modify source code or invent numbers
- Zero Hallucination invariant: All numerical results must strictly originate from genuine files in repository
- Originating prompt blockquote invariant on research reports

## Current Parent
- Conversation ID: aefc8a47-d86d-4238-85b0-c8080de54f82
- Updated: 2026-09-24T16:25:30+07:00

## Investigation State
- **Explored paths**:
  - `outputs/lunar_results/benchmark_summary.csv`
  - `outputs/lunar_results/benchmark_summary.json`
  - `outputs/lunar_results/sensitivity_sweep/alpha_sensitivity_summary.csv`
  - 32 per-run CSVs in `outputs/lunar_results/`
  - `BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md`
  - `WALKTHROUGH_FEDERATED_LUNAR.md`
  - `fed_lunar/federated/fed_lunar.py`, `fed_lunar/federated/sketches.py`
- **Key findings**:
  - Master Benchmark: Proposed Fed-LUNAR achieved AUC-ROC 99.73% (F1 93.14%) on BoTIoT, 99.99% (F1 99.40%) on EdgeIIoTset, 96.41% (F1 82.48%) on CICIoT2023, and 99.79% (F1 97.54%) on N_BaIoT, dramatically exceeding Naive Fed-LUNAR (+21.93% F1 on BoTIoT, +17.04% on CICIoT2023, +16.09% on N_BaIoT).
  - Dirichlet Non-IID Sensitivity: At $\alpha=0.1$, Proposed Fed-LUNAR maintains stable F1 70.65% (Detection Rate 83.33%) on CICIoT2023 where Naive drops to F1 60.00% (Detection Rate 69.70%).
  - Ablations: CMNP removal collapses BoTIoT F1 by -26.23% (from 93.14% down to 66.91%), CICIoT F1 by -8.82%, and N_BaIoT F1 by -9.79%. MSSP prevents distance inversion breakdown (restoring AUC from 0.15% to 99.73%).
  - Edge Feasibility: Streaming latency is 0.0008 - 0.0020 ms/sample (500k-1.25M pkts/s), peak RAM 49.6 - 67.1 MB (85.4% - 93.0% reduction vs LOC-NFST's 434.8 - 958.2 MB), model payload 13.0 KB, and one-time FSDS sketch <5.0 KB (1.16 - 4.98 KB).
- **Unexplored areas**: None for empirical benchmark data extraction.

## Key Decisions Made
- All 4 LaTeX tables formatted with double-column `booktabs` and `adjustbox` for direct inclusion into `paper_latex/sec_experiments.tex`.
- Calibrated F1 (fixed at 5% FAR) and Optimal F1 are both reported for full academic rigor.

## Artifact Index
- `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_data_1\handoff.md`: Full 5-component handoff report with exact verified numbers and ready-to-use LaTeX tables.
