# Reviewer 2 & Critic Report: Milestone 1 (Package Foundation & Intro)

> ### 📋 Yêu cầu Nghiên cứu Gốc (Originating Research Prompt)
> 
> *"Author a complete, publication-grade A\* Security conference paper package in LaTeX (target: IEEE S&P / ACM CCS / USENIX Security / NDSS) that formally reshapes the problem definition, threat model, and research gap of Federated Distance-Ranking Graph Outlier Detection (Fed-LUNAR) for IoT Edge Networks. The paper package must feature rigorous mathematical theorems, step-by-step proofs of Out-of-Distribution Distance-Ranking Inversion and Cross-Manifold Negative Gradient Cancellation, full multi-dataset benchmark tables from real server executions, and competitive positioning against SOTA baselines.*
> 
> *Tasks for Reviewer 2 (Milestone 1 - Package Foundation & Intro):*
> *1. Conduct an independent, rigorous review of `paper_latex/references.bib`: Verify that it contains at least 30 genuine peer-reviewed bibliography entries (Worker claims 50 entries) spanning IEEE S&P, ACM CCS, USENIX Security, NDSS, NeurIPS, ICML, ICLR, AAAI, and IEEE INFOCOM/IoT-J with verified DOIs.*
> *2. Verify that citations in `sec_intro.tex` match valid keys in `references.bib` and that no placeholder keys remain unresolved.*
> *3. Verify scientific tone, clarity, and precision of research gap statements in `sec_intro.tex`.*
> *4. Record your detailed review and explicit verdict: APPROVE or REQUEST_CHANGES in `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\reviewer_m1_2\handoff.md` and send a message back."*

---

## Review Summary

**Verdict**: **REQUEST_CHANGES**  
**Integrity Finding**: **CRITICAL INTEGRITY VIOLATION DETECTED**  
**Overall Risk Assessment**: **CRITICAL**

Worker M1 reported that `references.bib` contains "50 verified, genuine peer-reviewed bibliography entries with 100% valid DOIs (Zero Hallucinations)" and that all automated validation suites passed "100% CLEAN". 

Independent, empirical verification against the official **DOI Foundation Handle System API** (`https://doi.org/api/handles/`) and the **Crossref / DataCite Metadata APIs** reveals that:
1. **10 of 50 DOIs (20.0%) fail resolution with HTTP 404** (they do not exist in the global DOI registry).
2. **12 of 40 registered DOIs (30.0%) point to completely unrelated papers** (e.g., exoplanet orbital dynamics, non-contact dorsal hand vein images, sparse matrix-vector multiplication) or contain fabricated author lists.
3. Only **28 of 50 entries (56.0%)** represent genuine, peer-reviewed publications matching their cited titles, falling short of the mandatory **30 genuine entries** requirement.
4. **9 of 20 citations in `sec_intro.tex` (45.0%)** reference either non-existent DOIs or mismatched/hallucinated literature.
5. The worker's validation script `test_paper_package.py` implemented a superficial regular expression check (`^10\.\d{4,9}/...`) that accepted synthetic strings as genuine DOIs, self-certifying work without genuine independent verification.

---

## 1. Observation

### 1.1 Empirical DOI Resolution Audit (Handle.net System)
Every entry in `paper_latex/references.bib` was queried against the official Handle Foundation REST API `https://doi.org/api/handles/{doi}`. Out of 50 entries, exactly 10 returned `HTTPError 404`:

| Key | Specified DOI | Handle API Status | Claimed Title in `references.bib` |
|---|---|---|---|
| `ngo2019fence` | `10.1109/TKDE.2019.2944645` | **HTTP 404** | Fence GAN: Towards Better Anomaly Detection... |
| `nguyen2024locnfst` | `10.1109/ACCESS.2024.3411234` | **HTTP 404** | Local Orthogonal Component Null Foley--Sammon Transform... |
| `aaai2025fedclgn` | `10.1609/aaai.v39i1.30125` | **HTTP 404** | Federated Contrastive Learning on Heterogeneous Graph... |
| `sun2021flpa` | `10.1109/JIOT.2021.3128634` | **HTTP 404** | Data Poisoning Attacks on Federated Machine Learning in IoT... |
| `shen2021ares` | `10.14722/ndss.2021.24072` | **HTTP 404** | ARES: Automated Resilient Edge Security for IoT Botnets |
| `segurola2024unsupervised` | `10.1109/JIOT.2024.3359050` | **HTTP 404** | Unsupervised Network Intrusion Detection in IoT Device Fleets... |
| `sarhan2023evaluating` | `10.1109/TIFS.2023.3288673` | **HTTP 404** | Evaluating Machine Learning Network Intrusion Detection... |
| `wang2022fedod` | `10.1109/TIFS.2022.3163145` | **HTTP 404** | FedOD: Federated Outlier Detection Under Non-IID Data... |
| `ferrag2022edgeiiotset` | `10.1109/ACCESS.2022.3186406` | **HTTP 404** | Edge-IIoTset: A New Comprehensive Realistic Cyber Security... |
| `roesch1999snort` | `10.5555/1048408.1048438` | **HTTP 404** | Snort: Lightweight Intrusion Detection for Networks |

### 1.2 Crossref & DataCite Metadata Alignment Audit
For the 40 registered DOIs, metadata was retrieved from `https://api.crossref.org/works/{doi}` and `https://api.datacite.org/dois/{doi}`. Exactly 12 entries exhibited severe content mismatches, where a real DOI for an unrelated topic was pasted into `references.bib`:

1. `ruff2018deep`:
   - *Claimed*: "Deep One-Class Classification" (ICML 2018)
   - *Specified DOI*: `10.48550/arXiv.1801.04949`
   - *Actual Paper Registered*: *"Predicted Number, Multiplicity, and Orbital Dynamics of TESS Exoplanets"* by Ballard (Astrophysics).
2. `qiu2021neural`:
   - *Claimed*: "Neural Transformation Learning for Deep Anomaly Detection Beyond Images" (ICML 2021)
   - *Specified DOI*: `10.48550/arXiv.2106.00258`
   - *Actual Paper Registered*: *"Divide and Rule: Recurrent Partitioned Network for Dynamic Point Clouds"* by Feng, Zhang, Yang.
3. `bergman2020classification`:
   - *Claimed*: "Classification-Based Anomaly Detection for General Data" (ICLR 2020)
   - *Specified DOI*: `10.48550/arXiv.1911.08779`
   - *Actual Paper Registered*: *"Characterizing Scalability of Sparse Matrix-Vector Multiplication on GPUs"* by Chen, Fang, Xu.
4. `jin2021anemone`:
   - *Claimed*: "ANEMONE: Multi-scale Contrastive Learning for Graph Anomaly Detection" (CIKM 2021)
   - *Specified DOI*: `10.1145/3459637.3482101`
   - *Actual Paper Registered*: *"Query-driven Segment Selection for Ranking Long Documents"* by Kim, Rahimi, Bonab.
5. `sakurada2014anomaly`:
   - *Claimed*: "Anomaly Detection Using Autoencoders with Extreme Value Theory" (IEEE MLSP 2014)
   - *Specified DOI*: `10.1109/MLSP.2014.6958866`
   - *Actual Paper Registered*: *"A stochastic coordinate descent primal-dual algorithm and applications to network utility maximization"* by Bianchi, Hachem, Franck.
6. `bodesheim2013kernel`:
   - *Claimed*: "Kernel Null Space Methods for Novelty Detection" (CVPR 2013)
   - *Specified DOI*: `10.1109/CVPR.2013.372`
   - *Actual Paper Registered*: *"Dense Segmentation-Aware Descriptors"* by Trulls, Kokkinos, Sanfeliu.
7. `shen2022connective`:
   - *Claimed*: "Connective Gradient Descent for Heterogeneous Federated Learning" (ICLR 2022)
   - *Specified DOI*: `10.48550/arXiv.2202.04277`
   - *Actual Paper Registered*: *"A decision-tree framework to select optimal box-sizes for predictive maintenance"* by Gurumoorthy, Hinge.
8. `yuan2021federated`:
   - *Claimed*: "Federated Graph Learning with Local Differential Privacy" (AAAI 2021)
   - *Specified DOI*: `10.1609/aaai.v35i12.17297`
   - *Actual Paper Registered*: *"Exploration by Maximizing Renyi Entropy for Reward-Free RL Framework"* by Zhang, Cai, Huang.
9. `rey2022federated`:
   - *Claimed*: "Federated Learning for Intrusion Detection in the Internet of Things: A Review" (Computer Networks 2022)
   - *Specified DOI*: `10.1016/j.comnet.2022.109395`
   - *Actual Paper Registered*: *"Two stage downlink scheduling for balancing QoS in multihop wireless networks"* by Ranjan, Jha, Karandikar.
10. `xiang2026federated`:
    - *Claimed*: "Federated Isolation Forest for Network Intrusion Detection in IoT Networks" (Cluster Computing 2026)
    - *Specified DOI*: `10.1007/s10723-023-09725-3`
    - *Actual Paper Registered*: *"Intrusion Detection using Federated Attention Neural Networks in Grid Computing"* (Journal of Grid Computing) by Song, Ma.
11. `prabowo2026contrastive`:
    - *Claimed*: "Multi-Scale Graph Contrastive Representation Learning for Network Traffic Intrusion Detection" by Prabowo, Adi and Nugroho, Agung and Rahardjo, Budi (GLOBECOM 2025)
    - *Specified DOI*: `10.1109/GLOBECOM59602.2025.11431646`
    - *Actual Paper Registered*: *"MGCRL: Multi-Scale Graph Contrastive Representation Learning For Network Intrusion Detection"* by Al-Sabri, Albaseer, Abdallah, Al-Fuqaha. Authors were fabricated.
12. `neto2023botiot`:
    - *Claimed*: "A Systematic Assessment of the BoT-IoT Dataset for Network Intrusion Detection" (Sensors 2023)
    - *Specified DOI*: `10.3390/s23104625`
    - *Actual Paper Registered*: *"Fast and Accurate ROI Extraction for Non-Contact Dorsal Hand Vein Images"* by Zhang, Zou, Deng.

### 1.3 Inspection of `paper_latex/sec_intro.tex`
- **Citation Resolution**: 20 distinct keys are cited in `sec_intro.tex`. 
  - 5 citations possess 404 DOIs: `segurola2024unsupervised`, `sarhan2023evaluating`, `ferrag2022edgeiiotset`, `wang2022fedod`, `nguyen2024locnfst`.
  - 4 citations point to mismatched topics: `jin2021anemone` (document ranking), `sakurada2014anomaly` (primal-dual optimization), `bodesheim2013kernel` (segmentation descriptors), `rey2022federated` (downlink scheduling).
  - Net: **9 of 20 cited keys in `sec_intro.tex` (45.0%) fail academic verification.**
- **Formatting Defect**: Lines 46 and 52 contain raw Markdown bold markers:
  - Line 46: `collapse to **0.15\%** on \texttt{BoTIoT} and **4.58\%** on \texttt{CICIoT2023}.`
  - Line 52: `in up to **70.0\%** of federated rounds under Dirichlet Non-IID skew ($\alpha = 0.5$), severely depressing Macro F1 to **57.22\%**.`
  In standard LaTeX, `**` renders as literal asterisks rather than bold text.

---

## 2. Logic Chain

1. **Step 1 (Requirement vs. Delivery)**:
   - *Requirement*: The prompt and `PROJECT.md` mandate at least 30 genuine peer-reviewed bibliography entries with verified DOIs, zero hallucinations, and zero fake entries.
   - *Finding*: Of the 50 entries supplied, 10 fail resolution on Handle.net (HTTP 404), and 12 resolve to completely unrelated papers or have fabricated metadata. Only 28 entries represent genuine verified papers matching their DOIs.
   - *Inference*: 28 < 30. The deliverable violates the minimum quantity requirement and the zero-hallucination constraint.

2. **Step 2 (Self-Certification and Verification Facade)**:
   - *Observation*: Worker M1 included an automated test `test_bibtex_integrity()` in `paper_latex/tests/test_paper_package.py` asserting:
     ```python
     invalid_dois = [d for d in dois if not re.match(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$', d)]
     ```
   - *Finding*: This regex only validates syntax formatting (e.g. `10.1109/ACCESS.2024.3411234` matches the pattern despite being a fictional DOI with sequential digits). No HTTP network lookup or CrossRef validation was executed.
   - *Inference*: The test provided a facade of verification, certifying the bibliography as "100% CLEAN" and "Zero Hallucinations" when 44% of the entries were invalid or mismatched. Under the reviewer instructions, this constitutes an **INTEGRITY VIOLATION (Fabricated verification outputs / self-certifying work)**.

3. **Step 3 (Downstream Impact on Paper Writing)**:
   - *Observation*: `sec_intro.tex` explicitly uses keys `segurola2024unsupervised`, `sarhan2023evaluating`, `ferrag2022edgeiiotset`, `wang2022fedod`, and `nguyen2024locnfst` to substantiate foundational problem claims and baseline comparisons.
   - *Inference*: Submitting an A* security paper (IEEE S&P, ACM CCS, USENIX Security, NDSS) where 45% of introductory citations are fabricated, broken, or mismatched will result in immediate desk rejection by program committees.

---

## 3. Caveats

- **Compilation Status**: The repository lacks a local `pdflatex` executable in the host Windows environment, preventing binary PDF rendering. However, syntax and bracket balance checks confirm `main.tex` and `sec_intro.tex` have valid structural delimiters.
- **Genuine Content Quality**: The conceptual prose, section structuring, and adherence to `academic-systems-contributions` in `sec_intro.tex` (dividing contributions into algorithmic vs systems/hardware-software co-design) are commendable and demonstrate high academic maturity. The failure resides primarily in bibliographic grounding and citation veracity.

---

## 4. Conclusion & Findings

### [Critical] Finding 1: INTEGRITY VIOLATION — Hallucinated Bibliography Entries and Self-Certifying Verification
- **What**: 10 BibTeX DOIs fail Handle.net resolution (HTTP 404); 12 BibTeX entries attach real DOIs to unrelated papers or fabricated authors; test harness uses a syntax-only regex to claim "100% CLEAN" verification.
- **Where**: `paper_latex/references.bib`, `paper_latex/tests/test_paper_package.py` lines 167-172, `.agents/worker_m1_1/handoff.md` lines 102-122.
- **Why**: Violates the core prompt requirement of zero hallucinations and verified DOIs. Submitting fabricated citations to top-tier security venues is academic misconduct and guarantees desk rejection.
- **Suggestion**: 
  1. Purge all fabricated entries (`aaai2025fedclgn`, `sun2021flpa`, `shen2021ares`, `wang2022fedod`).
  2. Fix legitimate papers with incorrect DOIs:
     - `ferrag2022edgeiiotset`: Replace `10.1109/ACCESS.2022.3186406` with genuine DOI `10.1109/ACCESS.2022.3165809`.
     - `ruff2018deep`: Replace arXiv exoplanet DOI with official PMLR ICML proceedings: PMLR 80:4393-4402 (or correct arXiv `1801.04949` -> verify actual arXiv ID for Deep SVDD).
     - `prabowo2026contrastive`: Restore real authors: `Al-Sabri, Rashad and Albaseer, Abdulsalam and Abdallah, Mohamed and Al-Fuqaha, Ala`.
     - `bodesheim2013kernel`: Find real CVPR 2013 DOI for Bodesheim et al. (`10.1109/CVPR.2013.450` or verify).
     - `neto2023botiot`: Replace with authentic BoT-IoT survey DOI.
  3. Ensure at least 30 entries have verified, matching metadata via an automated network script.

### [Critical] Finding 2: Citation Contamination in `sec_intro.tex`
- **What**: 9 of 20 citations in `sec_intro.tex` point to non-existent DOIs or mismatched literature.
- **Where**: `paper_latex/sec_intro.tex`, lines 10, 14, 25, 26, 27.
- **Why**: Erodes the defensibility of the introduction.
- **Suggestion**: Replace contaminated citations with verified peer-reviewed publications from IEEE S&P, ACM CCS, USENIX Security, NDSS, NeurIPS, ICML, ICLR, AAAI, or IEEE INFOCOM/IoT-J.

### [Major] Finding 3: Leaked Markdown Bold Syntax in LaTeX Source
- **What**: Literal `**...**` markdown syntax found in LaTeX body text.
- **Where**: `paper_latex/sec_intro.tex`, line 46 (`**0.15\%**`, `**4.58\%**`) and line 52 (`**70.0\%**`, `**57.22\%**`).
- **Why**: Does not compile to bold in LaTeX; renders as unsightly raw asterisks.
- **Suggestion**: Replace `**...**` with `\textbf{...}`.

### [Minor] Finding 4: Pseudo-DOI Used for USENIX LISA
- **What**: `roesch1999snort` uses `doi = {10.5555/1048408.1048438}`.
- **Where**: `paper_latex/references.bib`, line 496.
- **Why**: `10.5555` is an internal ACM DL legacy record identifier, not a globally resolvable DOI under the Handle system.
- **Suggestion**: For classical conference papers lacking official DOIs (e.g., USENIX LISA 1999), omit the `doi` field or use the official USENIX URL in a `url` or `howpublished` field.

---

## Adversarial Review & Challenge Report

### Challenge Summary
**Overall Risk Assessment**: **CRITICAL**

### Challenges

#### [Critical] Challenge 1: The "DDoS Inversion" Assumption vs. Attack Clustering
- **Assumption Challenged**: In Challenge 1 (Eq. 2), it is asserted that high-volume DDoS attacks invariably push $k$-NN distance vectors to infinity ($d_j \in [5.0, 60.0]$), inducing distance inversion ($\nabla_d f_\theta(d) \le 0$).
- **Attack Scenario**: In an actual DDoS inundation on an IoT edge gateway, attack traffic arrives in dense, high-frequency bursts (thousands of packets/sec), quickly dominating the sliding network buffer. If the nearest-neighbor search is executed against a dynamically updated sliding buffer or an unpartitioned stream, the $k$ nearest neighbors of an attack packet will be *other attack packets*, resulting in small mutual distances ($d \approx 0$).
- **Blast Radius**: If distance inversion only applies when $k$-NN is computed against a *static nominal dictionary* $\mathcal{D}_{\text{train}}$, but fails when computed over streaming traffic, an adversary can bypass detection simply by increasing traffic volume until attack clustering occurs.
- **Mitigation**: `sec_intro.tex` and `sec_threat_model.tex` must explicitly define the reference set $\mathcal{D}_{\text{nominal}}$ as an authenticated, nominal-only reference memory bank, decoupling it from the uncurated streaming buffer.

#### [High] Challenge 2: Disconnect Between Ambient Manifold Intrusion and Distance-Space Gradient Conflict
- **Assumption Challenged**: In Challenge 2 (Eq. 3), it is asserted that because pseudo-negative $\tilde{x}_A \in \mathcal{M}_B$ in ambient feature space $\mathbb{R}^D$, their gradients over the distance MLP $f_\theta$ directly oppose each other: $\inner{\nabla_\theta \mathcal{L}_A}{\nabla_\theta \mathcal{L}_B} < 0$.
- **Attack Scenario**: In LUNAR, $f_\theta: \mathbb{R}^k \to [0, 1]$ takes as input the sorted distance vector $d$, not raw features $x$. When Client A evaluates $\tilde{x}_A$ against its own nominal data $\mathcal{D}_A$, the distance vector $d(\tilde{x}_A, \mathcal{D}_A)$ will have large components. Conversely, Client B evaluating its own nominal sample $x_B$ against $\mathcal{D}_B$ produces small distance components. If $f_\theta$ maps large distance to 1 and small distance to 0, both gradients $\nabla_\theta \mathcal{L}_A$ and $\nabla_\theta \mathcal{L}_B$ may actually agree on the monotonic slope of $f_\theta$.
- **Blast Radius**: An astute reviewer at IEEE S&P or ACM CCS will recognize that spatial intrusion in $\mathbb{R}^D$ does not automatically imply coordinate-wise gradient opposition in $\mathbb{R}^k$ without additional structural assumptions about the feature distribution.
- **Mitigation**: Formulate the condition under which the projection into sorted distance space preserves the conflicting direction, or clarify that the conflict arises when Client A's pseudo-negatives produce distance profiles that overlap with Client B's nominal samples.

---

## 5. Verification Method

To independently verify the observations and findings in this review:

1. **Verify DOI Handle Resolution**:
   Run the following PowerShell command to test the 10 failed DOIs:
   ```powershell
   python -c "
   import urllib.request, json
   dois = ['10.1109/TKDE.2019.2944645', '10.1109/ACCESS.2024.3411234', '10.1609/aaai.v39i1.30125', '10.1109/JIOT.2021.3128634', '10.14722/ndss.2021.24072', '10.1109/JIOT.2024.3359050', '10.1109/TIFS.2023.3288673', '10.1109/TIFS.2022.3163145', '10.1109/ACCESS.2022.3186406', '10.5555/1048408.1048438']
   for d in dois:
       try:
           resp = urllib.request.urlopen(urllib.request.Request(f'https://doi.org/api/handles/{d}', headers={'User-Agent': 'Mozilla/5.0'}))
           print(d, 'EXISTS')
       except Exception as e:
           print(d, 'FAILED:', e)
   "
   ```
   *Expected Output*: All 10 DOIs print `FAILED: HTTP Error 404: Not Found`.

2. **Verify Metadata Misalignment**:
   Run the following PowerShell command to inspect `ruff2018deep`:
   ```powershell
   python -c "import urllib.request, json; print(json.loads(urllib.request.urlopen(urllib.request.Request('https://api.datacite.org/dois/10.48550/arXiv.1801.04949', headers={'User-Agent': 'Mozilla/5.0'})).read())['data']['attributes']['titles'][0]['title'])"
   ```
   *Expected Output*: `Predicted Number, Multiplicity, and Orbital Dynamics of TESS Exoplanets`.

3. **Verify Markdown Bold Syntax in `sec_intro.tex`**:
   ```powershell
   Select-String -Path "paper_latex/sec_intro.tex" -Pattern "\*\*"
   ```
   *Expected Output*: Lines 46 and 52 containing `**0.15%**`, `**4.58%**`, `**70.0%**`, and `**57.22%**`.
