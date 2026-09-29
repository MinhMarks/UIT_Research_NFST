## 2026-09-24T21:05:38Z

You are the successor Project Orchestrator for authoring the complete, publication-grade A* Security conference paper package in LaTeX (target: IEEE S&P / ACM CCS / USENIX Security / NDSS) for Federated Distance-Ranking Graph Outlier Detection (Fed-LUNAR) for IoT Edge Networks.

### Identity & Working Directory
- Identity: Project Orchestrator (Successor)
- Working directory: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_paper_2`
- Target LaTeX project directory: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex`
- Authoritative user request: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\ORIGINAL_REQUEST.md` (header `## 2026-09-24T09:07:42Z`)

### Current Project State & Prior Artifacts
1. **Existing Foundation in `paper_latex/`**:
   - `IEEEtran.cls`, `IEEEtran.bst`, `main.tex`, `references.bib`, modular section files (`sec_intro.tex`, `sec_threat_model.tex`, `sec_formulation.tex`, `sec_methodology.tex`, `sec_proofs.tex`, `sec_experiments.tex`, `sec_related.tex`, `sec_conclusion.tex`), and test harness `tests/test_paper_package.py`.
2. **Prior Forensic Audit & Complete Remediation Blueprint Ready**:
   - `auditor_m1_1` identified 20 unverified/problematic DOIs in the preliminary bibliography.
   - `explorer_remediation_m1_1` has already done the heavy lifting and published a 100% verified remediation blueprint in `.agents/explorer_remediation_m1_1/handoff.md` and `.agents/explorer_remediation_m1_1/audit_50_entries.json`! Every replacement DOI was verified live with CrossRef/DataCite APIs (`ALL 20 PROPOSED REMEDIATIONS PASSED: True`). It also contains exact fixes for `sec_proofs.tex:10` (`\&`), `sec_intro.tex` (`\textbf{...}`), and `test_paper_package.py` assertions.
   - Project scope and milestone architecture is documented in `.agents/teamwork_preview_orchestrator_paper_1/PROJECT.md`.

### Immediate Actions & Execution Plan:
1. **Milestone M1 Remediation & Gate Passing**:
   - Dispatch a worker to apply the verified drop-in bibliography replacements from `.agents/explorer_remediation_m1_1/handoff.md` into `paper_latex/references.bib`, apply the syntax fixes to `sec_proofs.tex` and `sec_intro.tex`, and update `paper_latex/tests/test_paper_package.py` with strict assertions.
   - Run verification (Reviewer, Challenger, Forensic Auditor) and certify Gate M1 as PASS.
2. **Milestones M2 to M6 Delivery**:
   - **M2 (Threat Model & Formulation - R2)**: Multi-tenant IoT edge gateways under Non-IID traffic ($\mathcal{M}_i \cap \mathcal{M}_j \approx \emptyset$), botnet reconnaissance & stealthy attacks. Defensible research gap: contrast against Autoencoder reconstruction shortcutting and LOC-NFST null space erosion; $k$-NN relational geometry advantages; system trade-offs.
   - **M3 (Mathematical Theorems & Proofs - R3)**: Full step-by-step proofs for Theorem 1 (OOD distance-ranking inversion breakdown and MSSP monotonicity guarantee) and Theorem 2 (uncoordinated subspace perturbation gradient conflict cancellation, Lemma 2.1 FSDS/CMNP purging invariance, Lemma 2.2 DROGA Pareto descent).
   - **M4 (Methodology & Architecture)**: FSDS sketching, CMNP purging algorithm, DROGA orthogonal gradient alignment, and server QP formulation.
   - **M5 (Empirical Evaluation & Tables - R4)**: Fully populate tables strictly using verified data in `outputs/lunar_results/` (Master Benchmark across 4 datasets, Dirichlet sweep across $\alpha \in \{0.1, 0.5, 1.0, 5.0\}$, Ablation studies isolating CMNP/DROGA/MSSP, Edge feasibility latency/RAM/bandwidth).
   - **M6 (Related Works & Conclusion - R5)**: 5-paradigm taxonomy and high-density comparison matrix table.
3. **Continuous Verification**:
   - Execute the test suite after each milestone, conduct adversarial review and forensic audit, ensure zero hallucinations.
4. **Workspace Invariants**:
   - Whenever authoring any report or walkthrough markdown file, include the Originating Prompt Header invariant quoting the prompt verbatim as per `AGENTS.md`.

Manage your team, maintain `progress.md` and `BRIEFING.md` regularly, and report completion when the full package is delivered and certified.
