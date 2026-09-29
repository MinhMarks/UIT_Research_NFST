# Handoff Report — Challenger 1: Milestone 1 Verification

**Milestone**: Milestone 1 (Package Foundation & Intro)  
**Role**: EMPIRICAL CHALLENGER (critic, specialist)  
**Verdict**: **REQUEST_CHANGES**  
**Date**: 2026-09-24T10:50:00Z  

---

## 1. Observation

Direct empirical observations collected through command execution, regex/AST token parsing, and independent stress test execution:

1. **Test Suite Execution (`paper_latex/tests/test_paper_package.py`)**:
   - Command: `pytest paper_latex/tests/test_paper_package.py -v`
     - Output: `5 passed, 5 warnings in 0.46s`
     - Verbatim Warning:
       ```text
       PytestReturnNotNoneWarning: Test functions should return None, but paper_latex/tests/test_paper_package.py::test_required_files returned <class 'bool'>.
       Did you mean to use `assert` instead of `return`?
       ```
     - Root cause: Functions in `test_paper_package.py` return `True`/`False` instead of using Python `assert`. In `pytest`, returning `False` does not fail a test; `pytest` records `PASSED`.
   - Command: `python paper_latex/tests/test_paper_package.py`
     - Output: `ALL 5 VERIFICATION SUITES PASSED SUCCESSFULLY (100% CLEAN)`.
     - Observation: The script uses shallow validation—counting only aggregate character counts without depth tracking, and omits checking for unescaped special characters (`&`, `%`, `_`) and Markdown formatting leaks.

2. **Critical LaTeX Syntax Defect in `paper_latex/sec_proofs.tex`**:
   - File: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex\sec_proofs.tex`
   - Line 10:
     ```latex
     \subsection{Proof of Theorem 1: Distance Inversion & Monotonicity Recovery}
     ```
   - Verbatim defect: Character `&` at column 52 is unescaped outside of any table or alignment environment.
   - LaTeX compilation impact: In LaTeX text mode and subsection titles, `&` is reserved as the alignment tab character. Compiling this file via `\input{sec_proofs}` in `main.tex` produces a fatal compilation crash:
     ```text
     ! Misplaced alignment tab character &.
     ```

3. **Markdown Syntax Leakage in `paper_latex/sec_intro.tex`**:
   - File: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex\sec_intro.tex`
   - Line 46:
     ```latex
     driving predicted anomaly logits toward $-\infty$ ($\sigma(f_\theta) \to 0.0000$). Consequently, massive volumetric attacks are classified as more nominal than legitimate traffic, causing empirical AUC-ROC to catastrophically collapse to **0.15\%** on \texttt{BoTIoT} and **4.58\%** on \texttt{CICIoT2023}.
     ```
   - Line 52:
     ```latex
     Under standard Federated Averaging (FedAvg)~\cite{mcmahan2017communication}, opposing client gradients annihilate each other ($\norm{\nabla_\theta \mathcal{L}_{\text{global}}} \to 0$). Empirical logs demonstrate that pairwise gradient conflicts occur in up to **70.0\%** of federated rounds under Dirichlet Non-IID skew ($\alpha = 0.5$), severely depressing Macro F1 to **57.22\%**.
     ```
   - Verbatim defect: Raw Markdown bold syntax `**0.15\%**`, `**4.58\%**`, `**70.0\%**`, and `**57.22\%**` is present in the LaTeX body instead of proper LaTeX command `\textbf{...}`.
   - Typesetting impact: LaTeX does not interpret `**` as bold markup; it renders literal asterisks in the generated camera-ready PDF (`**0.15%**`, `**4.58%**`, etc.), violating publication-grade formatting standards.

4. **Section Stub Verification**:
   - Every `\input{sec_...}` declared in `main.tex` resolves to an existing file with non-zero size:
     - `sec_intro.tex`: 14,684 bytes
     - `sec_threat_model.tex`: 2,017 bytes
     - `sec_formulation.tex`: 1,611 bytes
     - `sec_methodology.tex`: 2,024 bytes
     - `sec_proofs.tex`: 2,074 bytes (Broken: syntax error on line 10)
     - `sec_experiments.tex`: 2,116 bytes
     - `sec_related.tex`: 2,145 bytes
     - `sec_conclusion.tex`: 1,566 bytes
   - All section stubs contain coherent milestone outlines and valid section labels.

5. **Cross-References and BibTeX Bibliography**:
   - Defined labels: 12. Referenced labels: 7. Missing `\ref`: 0.
   - Cited keys: 39. Available keys in `references.bib`: 50. Missing `\cite`: 0.
   - All 50 BibTeX entries contain valid `doi = {...}`, `author`, `title`, and `year`. Zero unescaped `&` in `references.bib`.

6. **Adversarial Syntax Stress Test Suite**:
   - An independent adversarial test suite was authored at `paper_latex/tests/adversarial_syntax_stress.py`.
   - Command: `python paper_latex/tests/adversarial_syntax_stress.py`
     - Result: Exit code 1.
     - Detected defects: 3 (1 unescaped `&` in `sec_proofs.tex:10`, 2 Markdown bold leaks in `sec_intro.tex:46,52`).
   - Command: `pytest paper_latex/tests/adversarial_syntax_stress.py -v`
     - Result: `FAILED paper_latex/tests/adversarial_syntax_stress.py::test_adversarial_syntax_stress - AssertionError: Adversarial syntax stress test found defects in paper_latex!`

---

## 2. Logic Chain

1. **From Observation 1**: When `pytest` executes `test_paper_package.py`, test functions returning `False` do not trigger assertions or test failures because pytest requires an `AssertionError` to mark a test as failed. Thus, relying solely on `pytest` provides a false sense of security.
2. **From Observation 2**: In standard LaTeX parsing, `&` is a reserved alignment delimiter. The token `&` on line 10 of `sec_proofs.tex` is outside any alignment environment. When `main.tex` inputs `sec_proofs.tex`, pdflatex halts with `! Misplaced alignment tab character &.` Therefore, the current package will fail full compilation.
3. **From Observation 3**: The strings `**0.15\%**`, `**4.58\%**`, `**70.0\%**`, and `**57.22\%**` in `sec_intro.tex` were copy-pasted or generated from raw Markdown draft notes without converting to `\textbf{...}`. In LaTeX, this results in literal double asterisks printed in the paper.
4. **From Observations 4, 5 & 6**: While the foundation architecture (`IEEEtran.cls`, `IEEEtran.bst`, `references.bib`, `main.tex`, and mathematical notation) is structurally sound and comprehensive, the presence of an unescaped `&` that breaks compilation and the presence of raw Markdown syntax violate the zero-syntax-error acceptance criteria.
5. **Conclusion**: Therefore, Milestone 1 cannot be approved in its current state. Changes must be requested and resolved.

---

## 3. Caveats

- **Local TeX Engine Absence**: No native `pdflatex`, `xelatex`, or `latexmk` binary is installed in PATH on this local Windows machine. Verification was performed using programmatic regex/AST tokenizers and parsing models mimicking TeX engines.
- **Scope Limit**: As an empirical challenger operating under review-only constraints, I identified, isolated, and wrote tests for the defects, but did not modify `paper_latex/sec_proofs.tex` or `paper_latex/sec_intro.tex` directly.

---

## 4. Conclusion & Actionable Recommendations

### Explicit Verdict
**REQUEST_CHANGES**

### Required Action Items for Milestone 1 Worker:
1. **Fix `paper_latex/sec_proofs.tex:10`**:
   - Change:
     ```latex
     \subsection{Proof of Theorem 1: Distance Inversion & Monotonicity Recovery}
     ```
   - To:
     ```latex
     \subsection{Proof of Theorem 1: Distance Inversion \& Monotonicity Recovery}
     ```
     *(or use `and`)*

2. **Fix `paper_latex/sec_intro.tex:46`**:
   - Change:
     ```latex
     causing empirical AUC-ROC to catastrophically collapse to **0.15\%** on \texttt{BoTIoT} and **4.58\%** on \texttt{CICIoT2023}.
     ```
   - To:
     ```latex
     causing empirical AUC-ROC to catastrophically collapse to \textbf{0.15\%} on \texttt{BoTIoT} and \textbf{4.58\%} on \texttt{CICIoT2023}.
     ```

3. **Fix `paper_latex/sec_intro.tex:52`**:
   - Change:
     ```latex
     pairwise gradient conflicts occur in up to **70.0\%** of federated rounds under Dirichlet Non-IID skew ($\alpha = 0.5$), severely depressing Macro F1 to **57.22\%**.
     ```
   - To:
     ```latex
     pairwise gradient conflicts occur in up to \textbf{70.0\%} of federated rounds under Dirichlet Non-IID skew ($\alpha = 0.5$), severely depressing Macro F1 to \textbf{57.22\%}.
     ```

4. **Enhance `paper_latex/tests/test_paper_package.py`**:
   - Replace boolean return statements (`return False`) with `assert` statements (e.g. `assert not missing, f"Missing files: {missing}"`) so that `pytest` fails immediately when a check fails.
   - Incorporate the checks from `paper_latex/tests/adversarial_syntax_stress.py` to prevent future regressions.

---

## 5. Verification Method

To independently verify these findings and confirm subsequent fixes:

1. **Run Independent Adversarial Syntax Checker**:
   ```powershell
   python paper_latex/tests/adversarial_syntax_stress.py
   ```
   - Expected currently: Exit code 1 with 3 defects reported.
   - Invalidation condition (Fix verified): Exit code 0 with `TOTAL DETECTED ADVERSARIAL DEFECTS: 0`.

2. **Run Pytest Adversarial Suite**:
   ```powershell
   pytest paper_latex/tests/adversarial_syntax_stress.py -v
   ```
   - Expected currently: FAILED with `AssertionError: Adversarial syntax stress test found defects in paper_latex!`.
   - Invalidation condition (Fix verified): PASSED `1 passed in 0.xx s`.

3. **Inspect Modified Lines**:
   - `paper_latex/sec_proofs.tex`: Verify line 10 has `\&` instead of `&`.
   - `paper_latex/sec_intro.tex`: Verify lines 46 and 52 have `\textbf{...}` instead of `**...**`.
