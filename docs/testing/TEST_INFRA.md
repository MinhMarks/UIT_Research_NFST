# Automated E2E Verification & Test Infrastructure Specification
## Project: Federated LUNAR (Fed-LUNAR) A* Security Conference Paper Package

### Overview
This document specifies the architecture, methodology, and verification contracts of the automated End-to-End (E2E) testing framework for the IEEEtran LaTeX conference paper package (`paper_latex/`).

The testing framework guarantees that the scientific publication package conforms strictly to:
1. Target venue standards (IEEE S&P / ACM CCS / USENIX Security / NDSS standard double-column format).
2. Clean LaTeX syntax and environment nesting, preventing compilation failures and broken references.
3. Genuine, peer-reviewed academic rigor with zero fabricated citations (all entries verified with DOIs).
4. The **Zero-Hallucination Empirical Invariant**: All experimental figures and metric tables in the manuscript match verbatim the recorded GPU benchmark logs executed on server `postmaster.iec` (NVIDIA RTX 5090).

---

## 1. Test Architecture & 4-Tier Methodology

The test suite implements a rigorous 4-tier verification hierarchy based on Category-Partition, Boundary Value Analysis, Pairwise Combinations, and Real-World Empirical Invariants:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       4-TIER E2E TEST SUITE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Feature & Package Structure Coverage                                │
│   - IEEEtran class existence & configuration (\documentclass[conference])    │
│   - Master entrypoint main.tex & references.bib                             │
│   - 8 Modular section files completeness & \input{} validation              │
│   - Forbidden package audit (strictly no natbib, enforce cite package)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Syntax, Environments & Brackets Validation                          │
│   - Comment-aware curly bracket balance ({ ... }) across all .tex files     │
│   - Math mode delimiter balance ($...$, $$...$$, \[...\])                   │
│   - Stack-based LaTeX environment nesting (\begin{env} ... \end{env})       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Bibliography & Cross-Reference Integrity                            │
│   - BibTeX entry count (>= 30 verified peer-reviewed publications)           │
│   - DOI completeness & format validation (100% genuine DOIs, 0 missing)     │
│   - Citation resolution (all \cite{...} resolve to references.bib)          │
│   - Cross-reference resolution (all \ref, \eqref, \cref resolve to \label)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Empirical Data & Zero-Hallucination Invariant                       │
│   - Table I: Master Benchmark matching against benchmark_summary.csv        │
│   - Table II: Dirichlet Sensitivity Sweep matching alpha_sensitivity_summary│
│   - Table III: Ablation Studies delta validation (NoCMNP -26.23%, etc.)     │
│   - Table IV: Edge Feasibility & Hardware Profiling (<5 KB FSDS sketches)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Tier Specifications

### Tier 1: Feature & Structure Coverage
* **Target Directory**: `paper_latex/`
* **Mandatory Files**:
  * `main.tex`: Master coordinator document.
  * `IEEEtran.cls`: Official IEEE conference document class.
  * `references.bib`: Verified peer-reviewed bibliography.
  * 8 Modular Section Files:
    1. `sec_intro.tex`: Introduction, motivation, 4 scientific contributions.
    2. `sec_threat_model.tex`: Multi-tenant Non-IID edge system model, adversary model, defensible gap.
    3. `sec_formulation.tex`: Mathematical graph distance-ranking formulation & gradient conflict dynamics.
    4. `sec_methodology.tex`: MSSP, FSDS sketches, CMNP purging, DROGA dual simplex QP.
    5. `sec_proofs.tex`: Theorems 1 & 2, Lemmas 2.1 & 2.2 formal step-by-step proofs.
    6. `sec_experiments.tex`: RTX 5090 benchmark results across 4 datasets, Tables I–IV.
    7. `sec_related.tex`: 5-paradigm taxonomy and comprehensive comparative matrix.
    8. `sec_conclusion.tex`: Concluding remarks, limitations, and future research directions.
* **Document Class Verification**: `main.tex` must specify `\documentclass[conference]{IEEEtran}`.
* **Forbidden Package Audit**: `natbib` causes fatal compatibility conflicts with `IEEEtran.cls`. `main.tex` must NOT load `natbib`; it must load standard `cite`.
* **Section Inclusion Verification**: `main.tex` must explicitly include all 8 section files via `\input{sec_...}`.

### Tier 2: Syntax, Environments & Brackets
* **Comment Stripping**: The parser correctly removes LaTeX comments (`% ...`) while preserving escaped percent characters (`\%`).
* **Bracket Matching**: Verifies that opening `{` and closing `}` curly braces match perfectly across every `.tex` file, properly ignoring escaped braces (`\{`, `\}`).
* **Math Delimiter Matching**:
  * Inline math pairs `$ ... $` must be closed and balanced (accounting for `\$`).
  * Display math environments (`$$ ... $$` and `\[ ... \]`) must be closed and balanced.
* **Environment Nesting**:
  * Stack-based LIFO tracking of `\begin{<env>}` and `\end{<env>}`.
  * Checks nested environments including `table`, `table*`, `tabular`, `figure`, `figure*`, `equation`, `align`, `gather`, `algorithm`, `algorithmic`, `proof`, `theorem`, `lemma`, `definition`.

### Tier 3: Bibliography & Cross-Reference Integrity
* **BibTeX Entry Cardinality**: `references.bib` must contain at least 30 entries (target: 35 entries).
* **DOI Invariant**: Every BibTeX entry must include a non-empty `doi = {...}` field. All DOIs must conform to the standard pattern `^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$`.
* **BibTeX Key Uniqueness**: Zero duplicate citation keys in `references.bib`.
* **Citation Resolution**: Every key referenced in any `\cite{...}` command across all `.tex` files must resolve to an entry in `references.bib`.
* **Cross-Reference Resolution**:
  * Every label defined via `\label{<lbl>}` is indexed.
  * Every reference in `\ref{<lbl>}`, `\eqref{<lbl>}`, `\cref{<lbl>}`, or `\Cref{<lbl>}` must resolve to a valid defined label.
  * Zero orphan references (preventing `??` in compiled PDF output).

### Tier 4: Empirical Data & Zero-Hallucination Invariant
* **Authoritative Ground Truth**:
  * `outputs/lunar_results/benchmark_summary.csv`
  * `outputs/lunar_results/sensitivity_sweep/alpha_sensitivity_summary.csv`
* **Table Verification**:
  1. **Table I (Master Benchmark)**:
     - Extracts all numerical values for AUC-ROC, Optimal F1, Calibrated F1, FAR, TPR/DR, Precision, Convergence Rounds, Latency ($\mu$s/sample), and Peak RAM (MB) for all 4 datasets (`BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT`) across 6 evaluated methods.
     - Asserts tolerance-free or strict $\pm 0.05\%$ precision matching against the CSV logs.
  2. **Table II (Dirichlet Sensitivity Sweep)**:
     - Extracts values across $\alpha \in \{0.1, 0.5, 1.0, 5.0\}$ on `BoTIoT` and `CICIoT2023`.
     - Validates AUC-ROC, Optimal F1, FAR, TPR/DR, and Round Conflict Ratio (%) against `alpha_sensitivity_summary.csv`.
  3. **Table III (Ablation Studies)**:
     - Validates performance metrics of Proposed Fed-LUNAR vs. `Ablation_FedLUNAR_NoDROGA`, `Ablation_FedLUNAR_NoCMNP`, and pre-MSSP fixed-radius baseline.
     - Asserts reported $\Delta\text{F1}$ drops:
       - BoTIoT NoCMNP: $-26.23\%$
       - CICIoT2023 NoCMNP: $-8.82\%$
       - N_BaIoT NoCMNP: $-9.79\%$
       - Fixed radius $\epsilon=0.1$ baseline drop: $-83.01\%$ on BoTIoT ($0.15\%$ AUC).
  4. **Table IV (Edge Feasibility & Resource Profiling)**:
     - Verifies streaming latency bounds ($0.8 - 2.0$ $\mu$s / $0.0008 - 0.0020$ ms).
     - Verifies throughput ($500,000 - 1,250,000$ packets/s).
     - Verifies peak memory working sets ($49.64 - 67.06$ MB vs. LOC-NFST $434.83 - 958.21$ MB).
     - Verifies per-round model payload ($13.0$ KB / 3,329 parameters).
     - Verifies one-time FSDS sketch size strictly $< 5.0$ KB ($1.16$ KB for BoTIoT, $2.28$ KB for EdgeIIoTset, $1.93$ KB for CICIoT2023, $4.98$ KB for N_BaIoT).

---

## 3. Test Runner & Execution Guide

The test harness is implemented in Python using `pytest` and standard `unittest` to ensure zero external dependency bloat.

### Running with Pytest
```bash
# Run complete test suite with detailed output
pytest paper_latex/tests/test_paper_package.py -v

# Run a specific tier
pytest paper_latex/tests/test_paper_package.py -k "Tier1" -v
pytest paper_latex/tests/test_paper_package.py -k "Tier2" -v
pytest paper_latex/tests/test_paper_package.py -k "Tier3" -v
pytest paper_latex/tests/test_paper_package.py -k "Tier4" -v
```

### Running with Python Unittest / Direct CLI
```bash
# Direct execution
python paper_latex/tests/test_paper_package.py

# Via unittest module
python -m unittest discover -s paper_latex/tests -p "test_*.py" -v
```

---

## 4. Progressive Testability & Milestone Mapping

| Milestone | Target Deliverables | Associated Test Classes | Expected Pass Condition |
| :--- | :--- | :--- | :--- |
| **M1** | `main.tex`, `IEEEtran.cls`, `references.bib`, `sec_intro.tex` | `TestTier1Structure`, `TestTier2Syntax`, `TestTier3Bibliography` (partial) | M1 files present, valid IEEEtran class, 35 DOIs valid, intro section valid. |
| **M2** | `sec_threat_model.tex`, `sec_formulation.tex` | `TestTier1Structure`, `TestTier2Syntax` | Threat model & mathematical formulation syntax, bracket & environment checks pass. |
| **M3** | `sec_methodology.tex` | `TestTier1Structure`, `TestTier2Syntax` | Methodology algorithms and equations balance. |
| **M4** | `sec_proofs.tex` | `TestTier1Structure`, `TestTier2Syntax` | Formal theorems and step-by-step proofs balance. |
| **M5** | `sec_experiments.tex` | `TestTier4EmpiricalData` | Tables I–IV populate and match RTX 5090 CSV ground truths with zero discrepancy. |
| **M6** | `sec_related.tex`, `sec_conclusion.tex` | `TestTier1Structure`, `TestTier2Syntax`, `TestTier3Bibliography` (full) | All 8 sections present, 100% citation resolution, 100% label resolution. |
| **E2E/FINAL** | Complete Paper Package | All 4 Tiers (`TestTier1` through `TestTier4`) | 100% tests pass, zero warnings, zero missing DOIs, zero ungrounded figures. |
