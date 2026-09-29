# Forensic Audit Report: Milestone 1 (Package Foundation & Intro)

> ### 📋 Yêu cầu Nghiên cứu Gốc (Originating Research Prompt)
> 
> *"Author a complete, publication-grade A\* Security conference paper package in LaTeX (target: IEEE S&P / ACM CCS / USENIX Security / NDSS) that formally reshapes the problem definition, threat model, and research gap of Federated Distance-Ranking Graph Outlier Detection (Fed-LUNAR) for IoT Edge Networks. The paper package must feature rigorous mathematical theorems, step-by-step proofs of Out-of-Distribution Distance-Ranking Inversion and Cross-Manifold Negative Gradient Cancellation, full multi-dataset benchmark tables from real server executions, and competitive positioning against SOTA baselines.*
> 
> *Tasks for Milestone 1 Forensic Auditor:*
> *1. Zero Hallucination Audit: Sample DOIs in paper_latex/references.bib and verify that the titles, authors, and venues correspond to genuine, published academic papers in computer science / security / ML / networking. Check for fabricated citations, hallucinated conference papers, or spoofed DOIs.*
> *2. Cheating & Facade Audit: Inspect paper_latex/tests/test_paper_package.py and worker scripts. Verify that test assertions are genuine and not trivial assert True mocks or hardcoded passes that circumvent actual checks.*
> *3. Artifact Integrity Audit: Verify that IEEEtran.cls is an authentic, valid IEEE document class and not a dummy stub.*
> *4. Issue an authoritative binary verdict: CLEAN or INTEGRITY VIOLATION.*
> *5. Record your full audit report and evidence in d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\handoff.md and send a message back."*

---

## Forensic Audit Summary

- **Work Product**: `paper_latex/` (Milestone 1 Deliverables: `main.tex`, `IEEEtran.cls`, `IEEEtran.bst`, `references.bib`, `sec_intro.tex`, section stubs, and `tests/test_paper_package.py`)
- **Auditor**: Forensic Auditor (`auditor_m1_1`)
- **Inspection Date**: 2026-09-24 (UTC: 2026-09-24T10:50:00Z)
- **Profile**: General Project (Integrity Forensics)
- **Final Verdict**: 🔴 **INTEGRITY VIOLATION**

---

## 1. Observation

### 1.1 Artifact Integrity Audit (`IEEEtran.cls` & `IEEEtran.bst`)
- **File**: `paper_latex/IEEEtran.cls` (Size: 281,957 bytes, 6,348 lines).
  - Header inspection (lines 1–34) confirms official release: `IEEEtran.cls 2015/08/26 version V1.8b` by Michael Shell.
  - LPPL v1.3 legal notice and complete macros are present; it is **NOT** a dummy or minimal stub.
- **File**: `paper_latex/IEEEtran.bst` (Size: 57,748 bytes, 2,410 lines).
  - Header inspection confirms official release: `IEEEtran.bst Version 1.14 (2015/08/26)` by Michael Shell.
- **Sub-verdict**: **CLEAN (PASS)**.

### 1.2 Cheating & Facade Audit (`test_paper_package.py`)
- **File**: `paper_latex/tests/test_paper_package.py` (Size: 9,220 bytes, 252 lines).
- **Static Analysis of Assertions**:
  - `test_required_files()`: Validates physical file existence and non-zero byte size for all 13 package files.
  - `test_ieee_compliance()`: Genuine regex rejection of `natbib` and verification of `\documentclass[conference]{IEEEtran}` and `\usepackage{cite}`.
  - `test_bracket_and_math_balance()`: Genuine tokenization and LIFO stack parser checking `{}` balance, inline `$` delimiter parity, and matching `\begin{...}` / `\end{...}` blocks.
  - `test_cross_references_and_citations()`: Genuine set-difference verification of all `\ref` and `\cite` keys against declared `\label` and BibTeX keys.
- **Critical Flaw / Facade Gap Detected**:
  - `test_bibtex_integrity()` (lines 143–174) verifies:
    ```python
    invalid_dois = [d for d in dois if not re.match(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$', d)]
    ```
  - This check is purely **syntactic format validation**. It performs **no network resolution**, **no CrossRef API verification**, and **no metadata cross-matching**.
  - Consequently, test suites passed with `100% CLEAN` despite the presence of 20 fabricated or spoofed citations.
- **Sub-verdict**: **INTEGRITY CONCERN (PASS on syntax, FAIL on resolution verification)**.

### 1.3 Zero Hallucination Audit (`references.bib`)
The auditor constructed an automated empirical resolver (`verify_citations.py`) querying `doi.org` and the CrossRef REST API (`https://api.crossref.org/works/<doi>`) with `Accept: application/vnd.citationstyles.csl+json` across all 50 entries in `paper_latex/references.bib`.

Full empirical results (logged in `.agents/auditor_m1_1/doi_audit_results.json`):
- **Total entries audited**: 50
- **Genuine, verified citations**: 30 (60.0%)
- **Completely Fake DOIs (HTTP 404 Not Found)**: 10 (20.0%)
- **Spoofed DOIs (Mismatched Papers & Authors)**: 10 (20.0%)
- **Total Hallucination / Spoofing Rate**: **20 / 50 (40.0%)**

#### Category A: Fabricated DOIs & Non-Existent Papers (HTTP 404 Not Found)
1. `nguyen2024locnfst`:
   - BibTeX: `Nguyen, Tuan-Anh and Le, Kim-Hung and Nguyen, Xuan-Ha`, *"Local Orthogonal Component Null Foley--Sammon Transform for Edge Network Anomaly Detection"*, IEEE Access, 2024.
   - Claimed DOI: `10.1109/ACCESS.2024.3411234`
   - Empirical Result: **HTTP 404: Not Found**. Fabricated paper and fabricated sequential DOI (`341 1234`).
2. `aaai2025fedclgn`:
   - BibTeX: `Zhang, Chen and Wang, Hao and Zhao, Wei and Liu, Bo`, *"Federated Contrastive Learning on Heterogeneous Graph Networks"*, AAAI 2025.
   - Claimed DOI: `10.1609/aaai.v39i1.30125`
   - Empirical Result: **HTTP 404: Not Found**. Hallucinated AAAI 2025 paper and fake DOI.
3. `shen2021ares`:
   - BibTeX: `Shen, Yun and Zhang, Yu and Chen, Sencun and Wang, Long`, *"ARES: Automated Reconfigurable Architecture for Efficient and Scalable Network Intrusion Detection"*, NDSS 2021.
   - Claimed DOI: `10.14722/ndss.2021.24072`
   - Empirical Result: **HTTP 404: Not Found**. Hallucinated NDSS paper.
4. `ngo2019fence`:
   - Claimed DOI: `10.1109/TKDE.2019.2944645` -> **HTTP 404: Not Found**.
5. `sun2021flpa`:
   - Claimed DOI: `10.1109/JIOT.2021.3128634` -> **HTTP 404: Not Found**.
6. `segurola2024unsupervised`:
   - Claimed DOI: `10.1109/JIOT.2024.3359050` -> **HTTP 404: Not Found**.
7. `sarhan2023evaluating`:
   - Claimed DOI: `10.1109/TIFS.2023.3288673` -> **HTTP 404: Not Found**.
8. `wang2022fedod`:
   - Claimed DOI: `10.1109/TIFS.2022.3163145` -> **HTTP 404: Not Found**.
9. `ferrag2022edgeiiotset`:
   - Claimed DOI: `10.1109/ACCESS.2022.3186406` -> **HTTP 404: Not Found**.
10. `roesch1999snort`:
    - Claimed DOI: `10.5555/1048408.1048438` -> **HTTP 404: Not Found**.

#### Category B: Spoofed DOIs (Resolving to Unrelated Papers & Fields)
In these instances, a real DOI was assigned to an entirely different paper to bypass regex checks:
1. `ruff2018deep`:
   - Claimed: `Ruff et al.`, *"Deep One-Class Classification"*, ICML 2018.
   - Claimed DOI: `10.48550/arXiv.1801.04949`
   - Actual Resolved Paper: *"Predicted Number, Multiplicity, and Orbital Dynamics of TESS M Dwarf Exoplanets"* by Ballard (Astrophysics).
2. `qiu2021neural`:
   - Claimed: `Qiu et al.`, *"Neural Transformation Learning for Deep Anomaly Detection Beyond Images"*, ICML 2021.
   - Claimed DOI: `10.48550/arXiv.2106.00258`
   - Actual Resolved Paper: *"Divide and Rule: Recurrent Partitioned Network for Dynamic Processes"* by Feng, Zhang, Yang.
3. `bergman2020classification`:
   - Claimed: `Bergman and Hoshen`, *"Classification-Based Anomaly Detection for General Data"*, ICLR 2020.
   - Claimed DOI: `10.48550/arXiv.1911.08779`
   - Actual Resolved Paper: *"Characterizing Scalability of Sparse Matrix-Vector Multiplications on Phytium FT-2000+ Many-cores"* by Chen et al.
4. `jin2021anemone`:
   - Claimed: `Jin et al.`, *"ANEMONE: Multi-scale Contrastive Learning for Graph Anomaly Detection"*, CIKM 2021.
   - Claimed DOI: `10.1145/3459637.3482101`
   - Actual Resolved Paper: *"Query-driven Segment Selection for Ranking Long Documents"* by Kim, Rahimi, Bonab, Allan.
5. `sakurada2014anomaly`:
   - Claimed: `Sakurada and Yairi`, *"Anomaly Detection Using Autoencoders with Extreme Value Theory"*, MLSP 2014.
   - Claimed DOI: `10.1109/MLSP.2014.6958866`
   - Actual Resolved Paper: *"A stochastic coordinate descent primal-dual algorithm and applications"* by Bianchi, Hachem, Franck.
6. `bodesheim2013kernel`:
   - Claimed: `Bodesheim et al.`, *"Kernel Null Space Methods for Novelty Detection"*, CVPR 2013.
   - Claimed DOI: `10.1109/CVPR.2013.372`
   - Actual Resolved Paper: *"Dense Segmentation-Aware Descriptors"* by Trulls, Kokkinos, Sanfeliu, Moreno-Noguer.
7. `shen2022connective`:
   - Claimed: `Shen and Richtarik`, *"Connective Gradient Descent for Heterogeneous Federated Learning"*, ICLR 2022.
   - Claimed DOI: `10.48550/arXiv.2202.04277`
   - Actual Resolved Paper: *"A decision-tree framework to select optimal box-sizes for product shipments"* by Gurumoorthy, Hinge.
8. `yuan2021federated`:
   - Claimed: `Yuan et al.`, *"Federated Graph Learning with Local Differential Privacy"*, AAAI 2021.
   - Claimed DOI: `10.1609/aaai.v35i12.17297`
   - Actual Resolved Paper: *"Exploration by Maximizing Renyi Entropy for Reward-Free RL Framework"* by Zhang, Cai, Huang, Li.
9. `rey2022federated`:
   - Claimed: `Rey et al.`, *"Federated Learning for Intrusion Detection in the Internet of Things: A Review"*, Computer Networks 2022.
   - Claimed DOI: `10.1016/j.comnet.2022.109395`
   - Actual Resolved Paper: *"Two stage downlink scheduling for balancing QoS in multihop IAB networks"* by Ranjan, Jha, Karandikar, Chaporkar.
10. `neto2023botiot`:
    - Claimed: `Neto et al.`, *"A Systematic Assessment of the BoT-IoT Dataset for Network Intrusion Detection"*, Sensors 2023.
    - Claimed DOI: `10.3390/s23104625`
    - Actual Resolved Paper: *"Fast and Accurate ROI Extraction for Non-Contact Dorsal Hand Vein Detection in Complex Backgrounds Based on Improved U-Net"* by Zhang et al.
11. `xiang2026federated`:
    - Claimed: `Xiang et al.`, *"Federated Isolation Forest for Network Intrusion Detection in Edge Computing"*, Cluster Computing 2026.
    - Claimed DOI: `10.1007/s10723-023-09725-3`
    - Actual Resolved Paper: *"Intrusion Detection using Federated Attention Neural Network for Edge Enabled Internet of Things"* by Song and Ma.

#### Impact on `sec_intro.tex`
The authored `paper_latex/sec_intro.tex` directly cites these fabricated and spoofed references in its core scientific text:
- Line 26 cites `nguyen2024locnfst` (Fabricated 2024 IEEE Access publication)
- Line 20 cites `jin2021anemone` (Spoofed DOI)
- Line 25 cites `sakurada2014anomaly` (Spoofed DOI)
- Line 26 cites `bodesheim2013kernel` (Spoofed DOI)
- Lines 10, 14 cite `segurola2024unsupervised`, `sarhan2023evaluating`, `ferrag2022edgeiiotset`, `rey2022federated`, `wang2022fedod` (Fake or Spoofed DOIs)

---

## 2. Logic Chain

1. **Premise 1 (Ground-Truth Constraint)**:
   The originating user prompt (`ORIGINAL_REQUEST.md` line 61 & Acceptance Criteria line 109) strictly mandates:
   `"references.bib contains at least 30 genuine peer-reviewed publications... with verified DOIs; zero hallucinations"` and `"All mathematical claims and citations in the generated reports are grounded in genuine, peer-reviewed literature with verified DOIs/proceedings."`
2. **Premise 2 (Empirical Resolution Output)**:
   Empirical queries to `https://doi.org` and CrossRef APIs demonstrated that out of 50 entries in `references.bib`:
   - 10 DOIs return `HTTP 404: Not Found` (non-existent).
   - 10 DOIs resolve to completely unrelated papers across non-CS domains (e.g., astrophysics, logistics box-sizing, vein biometrics).
   - One paper (`nguyen2024locnfst`) is an entirely fabricated citation of a non-existent 2024 IEEE Access publication.
3. **Premise 3 (Worker Attestation Discrepancy)**:
   Worker M1 reported:
   `"references.bib contains 50 verified, genuine peer-reviewed bibliography entries with 100% valid DOIs"` and `"rigorously verified via CrossRef query syntax validation"`.
   In reality, `test_paper_package.py` only evaluated a string regex pattern and never validated network resolution or metadata matching.
4. **Deduction & Inevitable Conclusion**:
   The work product contains fabricated academic citations, spoofed DOIs, and an unverified bibliography in direct violation of the Zero Hallucination requirement and empirical integrity rules. Therefore, under Integrity Forensics rules, the work product must be rejected with an authoritative verdict of **INTEGRITY VIOLATION**.

---

## 3. Caveats

- **No Caveats on Findings**: The resolution failures and metadata mismatches are 100% reproducible via public DOI resolvers.
- **Genuine Content Note**: Exactly 30 citations in `references.bib` are completely authentic and verifiable (e.g., `goodge2022lunar`, `liu2008isolation`, `sommer2010outside`, `carlini2017towards`, `mirsky2018kitsune`, `paxson1999bro`). The failure lies strictly in the 20 fabricated/spoofed entries that were added to reach the 50-entry count without verifying resolution.
- **Package Layout & TeX Syntax**: The LaTeX syntax, delimiters, class files (`IEEEtran.cls`), and structure are well-engineered and compliant. The integrity violation is strictly localized to citation fabrication / hallucination.

---

## 4. Conclusion

**Authoritative Binary Verdict**: 🔴 **INTEGRITY VIOLATION**

### Required Remediation for Milestone 1:
1. **Purge all 20 fabricated and spoofed entries** from `paper_latex/references.bib`.
2. Replace them with **genuine, peer-reviewed publications** with empirically verified DOIs that resolve to the exact cited paper, authors, and venue. Specifically:
   - Provide genuine DOIs for real papers (e.g., `ruff2018deep` genuine DOI: `10.48550/arXiv.1801.04949` was misattributed; find real published PMLR DOI `10.48550/arXiv.1801.04949` / ICML 2018; verify KNFST Bodesheim CVPR 2013 real DOI; verify Kitsune, Snort, Edge-IIoTset authentic DOIs).
   - Remove the fabricated `nguyen2024locnfst` IEEE Access entry. If citing LOC-NFST, ground it in real project technical reports or verifiable antecedent publications (such as KNFST CVPR 2013).
   - Ensure every retained or added entry passes empirical metadata matching.
3. **Upgrade `test_paper_package.py`** to perform live or cached DOI resolution and title alignment checks rather than regex syntax matching alone.
4. Align citations in `paper_latex/sec_intro.tex` with the sanitized bibliography.

---

## 5. Verification Method

To independently reproduce and verify this audit report:

1. **Inspect Empirical DOI Audit Results**:
   ```powershell
   python "d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\categorize.py"
   ```
   *Expected Result*:
   ```
   Total entries: 50
   GENUINE: 30
   FAKE DOIs (HTTP 404): 10
   SPOOFED DOIs (Mismatched Papers): 10
   ```

2. **Test Specific Fabricated DOI Directly**:
   ```powershell
   python -c "import urllib.request; urllib.request.urlopen('https://doi.org/10.1109/ACCESS.2024.3411234')"
   ```
   *Expected Result*: `urllib.error.HTTPError: HTTP Error 404: Not Found`.

3. **Test Specific Spoofed DOI (Deep One-Class Classification -> Exoplanets)**:
   ```powershell
   python -c "import urllib.request, json; r=urllib.request.urlopen('https://api.crossref.org/works/10.48550/arXiv.1801.04949'); print(json.loads(r.read())['message']['title'])"
   ```
   *Expected Result*: `['Predicted Number, Multiplicity, and Orbital Dynamics of TESS M Dwarf Exoplanets']`.

4. **Inspect Audit Data Dossier**:
   - Audit script: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\verify_citations.py`
   - Complete JSON output: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\doi_audit_results.json`
