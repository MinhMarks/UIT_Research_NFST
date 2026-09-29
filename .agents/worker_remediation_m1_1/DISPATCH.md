## 2026-09-24T11:11:52Z

You are a Worker subagent for Milestone 1 Remediation (Worker M1 Iteration 2).
Your working directory is: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\worker_remediation_m1_1

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You must read:
- Original Request: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md (specifically ## 2026-09-24T09:07:42Z)
- Project Scope: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_paper_1\PROJECT.md
- Remediation Plan from Explorer: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\explorer_remediation_m1_1\handoff.md
- Forensic Auditor Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\handoff.md
- Reviewer 1 Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\reviewer_m1_1\handoff.md
- Challenger 1 Report: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\challenger_m1_1\handoff.md

### Exclusive Write Ownership:
You own:
- `paper_latex/references.bib`
- `paper_latex/sec_intro.tex`
- `paper_latex/sec_proofs.tex`
- `paper_latex/sec_related.tex`
- `paper_latex/tests/test_paper_package.py`

### Tasks:
1. Implement the 20 verified bibliography replacements in `paper_latex/references.bib` strictly as documented in `explorer_remediation_m1_1/handoff.md`. Ensure that:
   - All 50 entries have genuine, real DOIs verified against official registries.
   - All fake entries (`nguyen2024locnfst`, `aaai2025fedclgn`, `shen2021ares`, etc.) are purged and replaced with authentic top-tier publications (`foley1975optimal`, `li2021model`, `kim2023robust`, etc.).
   - All corrected DOIs (`ferrag2022edgeiiotset`, `bodesheim2013kernel`, `jin2021anemone`, etc.) are updated.
2. Fix `paper_latex/sec_proofs.tex:10`: escape the unescaped ampersand (`\&`).
3. Fix `paper_latex/sec_intro.tex`:
   - Replace raw Markdown bold formatting (`**...**`) on lines 46 and 52 with `\textbf{...}`.
   - Align all citations so they point to the newly verified genuine BibTeX keys.
4. Update `paper_latex/tests/test_paper_package.py`:
   - Convert all return booleans to proper Python `assert` statements.
   - Incorporate the authoritative DOI registry / verification check so that fake DOIs are deterministically rejected.
5. Run the test suite: `pytest paper_latex/tests/test_paper_package.py -v` and `python paper_latex/tests/test_paper_package.py` to confirm that all tests pass cleanly without errors or warnings.
6. Write your completion report in `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\worker_remediation_m1_1\handoff.md` and send a message back.
