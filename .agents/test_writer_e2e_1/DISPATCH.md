## 2026-09-24T09:26:50Z
You are an E2E Test Writer subagent (E2E Testing Track).
Your working directory is: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\test_writer_e2e_1

You must read:
- Original Request: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md (specifically ## 2026-09-24T09:07:42Z)
- Project Scope: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_paper_1\PROJECT.md
- LaTeX Survey Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_latex_1\handoff.md
- Empirical Survey Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_survey_data_1\handoff.md

### Exclusive Write Ownership:
You own:
- `paper_latex/tests/` directory and test files within it (e.g. `paper_latex/tests/test_paper_package.py`)
- `TEST_INFRA.md` at project root `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\TEST_INFRA.md`
- `TEST_READY.md` at project root `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\TEST_READY.md` (upon completion)

### Mission:
Build a comprehensive, automated, requirement-driven E2E validation test suite in Python for the LaTeX paper package following the 4-tier methodology (Category-Partition, Boundary Value Analysis, Pairwise Combinations, Real-World Scenarios):
1. **Tier 1 (Feature & Structure Coverage)**:
   - Verifies existence of `main.tex`, `references.bib`, `IEEEtran.cls`, and all 8 section files (`sec_intro.tex`, `sec_threat_model.tex`, `sec_formulation.tex`, `sec_methodology.tex`, `sec_proofs.tex`, `sec_experiments.tex`, `sec_related.tex`, `sec_conclusion.tex`).
   - Verifies `\documentclass[conference]{IEEEtran}` and forbidden package check (no `natbib`).
2. **Tier 2 (Syntax, Environments & Brackets)**:
   - Validates curly bracket `{}` and math delimiter `$` matching across all `.tex` files.
   - Validates `\begin{env}...\end{env}` matching.
3. **Tier 3 (Bibliography & Cross-Reference Integrity)**:
   - Validates that `references.bib` has >=30 entries and every entry has a valid, genuine `doi = {...}`.
   - Validates that every `\cite{...}` in any `.tex` file resolves to a valid key in `references.bib`.
   - Validates that every `\ref{...}` resolves to a defined `\label{...}`.
4. **Tier 4 (Empirical Data & Zero Hallucination Invariant)**:
   - Extracts all numbers from LaTeX tables in `sec_experiments.tex` and asserts exact match against `outputs/lunar_results/benchmark_summary.csv` and `alpha_sensitivity_summary.csv`.
   - Verifies the 4 mandatory tables: Master Benchmark, Dirichlet Sweep, Ablations, Edge Feasibility.

### Tasks:
1. Create `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\TEST_INFRA.md` detailing the test philosophy and tiers.
2. Implement `paper_latex/tests/test_paper_package.py` with executable unit tests using `pytest` or Python `unittest`.
3. Run the test suite using Python to verify test execution and produce initial baseline output.
4. When the test infrastructure is operational and published, generate `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\TEST_READY.md`.
5. Write your report in `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\test_writer_e2e_1\handoff.md` and send a message back.
