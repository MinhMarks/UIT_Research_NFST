# Project: Federated LUNAR (Fed-LUNAR) A* Security Conference Paper Package

## Architecture
- Target Venue: IEEE S&P / ACM CCS / USENIX Security / NDSS (Standard double-column IEEEtran conference format, 10–13 pages + references).
- Working Directory: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex`
- Modular Architecture:
  - `main.tex`: Master conference paper entrypoint, package loading, IEEEtran setup.
  - `references.bib`: Verified peer-reviewed bibliography (35 entries with genuine DOIs, zero hallucinations).
  - `sec_intro.tex`: Introduction, motivation, edge IoT challenges, research questions, summary of 4 core contributions.
  - `sec_threat_model.tex`: Decentralized multi-tenant IoT edge system model, Non-IID manifold skew, threat model (Mirai, Gafgyt, volumetric floods), defensible research gap (AD Trilemma vs Autoencoders & LOC-NFST), transparent system trade-offs.
  - `sec_formulation.tex`: Mathematical foundations of graph distance-ranking outlier detection, negative sampling in unbounded metric space, Non-IID gradient conflict dynamics.
  - `sec_methodology.tex`: Fed-LUNAR architecture: MSSP, FSDS (<5 KB sketches), CMNP, DROGA (dual simplex QP), algorithms and complexity analysis.
  - `sec_proofs.tex`: Complete formal proofs: Theorem 1 (Distance Inversion & Monotonicity Recovery), Theorem 2 (Negative Gradient Cancellation), Lemma 2.1 (Purging Invariance), Lemma 2.2 (Pareto Convergence).
  - `sec_experiments.tex`: Comprehensive empirical results from RTX 5090 logs: Table I (Master Benchmark), Table II (Dirichlet Sensitivity Sweep 32 runs), Table III (Ablation Studies), Table IV (Edge Feasibility & Resource Profiling).
  - `sec_related.tex`: 5-paradigm taxonomy and high-density LaTeX comparison matrix.
  - `sec_conclusion.tex`: Synthesis, future directions, reproducibility and ethical disclosures.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | IEEEtran Architecture & Verified Bibliography | Master `main.tex`, `references.bib` with 35 verified DOIs, preamble, macros | M1 | Survey 1 |
| 2 | Introduction & Academic Systems Contributions | `sec_intro.tex` with problem motivation, threat context, 4 core contributions | M1 | Survey 1, 3 |
| 3 | System Model, Threat Model & AD Trilemma | `sec_threat_model.tex` with multi-tenant Non-IID model, adversary capabilities, gap vs AE & LOC-NFST | M2 | Survey 3 |
| 4 | Mathematical Problem Formulation | `sec_formulation.tex` with graph distance-ranking formulation, negative sampling, gradient conflict | M2 | Survey 3 |
| 5 | Fed-LUNAR Algorithmic Methodology | `sec_methodology.tex` with MSSP, FSDS (<5 KB), CMNP, DROGA, pseudocodes | M3 | Survey 2, 3 |
| 6 | Formal Theorems & Step-by-Step Proofs | `sec_proofs.tex` with complete step-by-step proofs of Thm 1, Thm 2, Lem 2.1, Lem 2.2 | M4 | Survey 3 |
| 7 | Multi-Dataset Empirical Evaluation & Tables | `sec_experiments.tex` with Tables I, II, III, IV, empirical narrative, edge profiling | M5 | Survey 2 |
| 8 | Related Work Matrix & Conclusion | `sec_related.tex` (5 paradigms + matrix) & `sec_conclusion.tex` | M6 | Survey 1, 3 |
| 9 | LaTeX Compilation & Quality Test Harness | E2E test harness validating syntax, citations, references, DOIs, math blocks | E2E Track | Survey 1 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Package Foundation & Intro | `paper_latex/main.tex`, `references.bib`, `sec_intro.tex`, IEEEtran class setup | none | IN_PROGRESS |
| M2 | Threat Model & Formulation | `sec_threat_model.tex`, `sec_formulation.tex` | M1 | PLANNED |
| M3 | Methodology & Algorithms | `sec_methodology.tex` (MSSP, FSDS, CMNP, DROGA, pseudocode) | M1, M2 | PLANNED |
| M4 | Mathematical Theorems & Proofs | `sec_proofs.tex` (Thm 1, Thm 2, Lem 2.1, Lem 2.2 step-by-step) | M2, M3 | PLANNED |
| M5 | Empirical Benchmark Presentation | `sec_experiments.tex` (Tables I-IV, RTX 5090 logs, edge profiling) | M1, M3 | PLANNED |
| M6 | Related Works & Conclusion | `sec_related.tex` (5 paradigms, matrix), `sec_conclusion.tex` | M1, M2, M5 | PLANNED |
| E2E | LaTeX Validation & Test Harness | Automated test harness, AST/regex parsing, reference/citation resolution, `TEST_READY.md` | M1 | IN_PROGRESS |
| FINAL | End-to-End Paper Verification | Complete compilation/validation pass, all tests passing, zero warnings | M1-M6, E2E | PLANNED |

## Interface Contracts
### `main.tex` ↔ Section Files
- All section files use standard `\section{...}` and contain clean modular content without redundant `\documentclass` or `\begin{document}`.
- Cross-references use consistent namespace prefixes: `sec:`, `fig:`, `tab:`, `thm:`, `lem:`, `def:`, `alg:`.
- Mathematical notations follow harmonized notation in Survey 3:
  - $\mathcal{G}_i$: edge gateway $i \in \{1, \dots, M\}$.
  - $\mathcal{M}_i \subset \mathbb{R}^D$: nominal manifold for client $i$.
  - $d(x, \mathcal{N}_k(x))$: sorted $k$-NN distance vector.
  - $f_\theta$: distance-ranking MLP.
  - $\mathcal{S}_i = (\mu_i, \Lambda_i, U_i, r_{i,\max})$: FSDS sketch.
  - $g_{\text{aligned}}$: DROGA aligned gradient.

## Code Layout
```
paper_latex/
├── main.tex
├── IEEEtran.cls
├── IEEEtran.bst (if required)
├── references.bib
├── sec_intro.tex
├── sec_threat_model.tex
├── sec_formulation.tex
├── sec_methodology.tex
├── sec_proofs.tex
├── sec_experiments.tex
├── sec_related.tex
├── sec_conclusion.tex
└── tests/
    └── test_paper_package.py
```
