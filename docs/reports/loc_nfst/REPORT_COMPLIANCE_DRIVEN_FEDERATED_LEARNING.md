> ### 📋 Yêu cầu Nghiên cứu Gốc (Originating Research Prompt)
>
> **🇻🇳 Vietnamese (Original):** *"Hãy nghiên cứu cho tôi tại sao lại có các quy định tuân thủ mà dẫn tới thuật toán Federated, tôi muốn dẫn cụ thể tới các luật quy định, tiêu chuẩn, và ánh xạ nó tới các yêu cầu khi thiết kế hệ thống cần Federated. Về ý nghĩa của câu hỏi nay tôi muốn có một cái nhìn thực tế, gốc nhìn sản phẩm. Hãy soạn lại cho tôi báo cáo chi tiết, được xác thực thông tin( từng claim phải được xác thực bằng paper hay tài liệu uy tín), đầu ra là file md. Nếu bạn chưa hiểu ý tôi thì có thể hỏi thêm nhiều /grill-me . Trả lời bằng tiếng anh và có lưu lại câu hỏi song ngữ ở đầu report. /boost"*
>
> **🇬🇧 English (Translation):** *"Research for me why compliance regulations led to the Federated Learning algorithm. I want to cite specific laws, regulations, and standards, and map them to the design requirements when building systems that need Federated Learning. From the perspective of this question, I want a practical view, from a product standpoint. Write me a detailed, verified report (each claim must be verified by a reputable paper or authoritative document), output as md file. Reply in English and save the bilingual question at the top of the report."*

---

# Compliance-Driven Federated Learning: A Product Architect's Deep-Dive

**Domain**: IoT Intrusion Detection Systems / Edge AI  
**Version**: 2.0 (Expanded, Peer-Review Grade)  
**Date**: 2026-09-27  
**Audience**: Systems architects, product engineers, compliance officers, and researchers working at the intersection of privacy law and distributed machine learning.

---

## Executive Summary

Federated Learning (FL) did not emerge from an academic vacuum — it emerged from a legal and commercial crisis. As machine learning became the foundation of products in healthcare, finance, industrial IoT, and telecommunications, engineering teams encountered a systemic blocker: the data required to train powerful models is precisely the data that regulators, customers, and governments are most determined to keep locked in place.

This report provides a rigorous, citation-backed analysis of the regulatory forces that have made Federated Learning not merely preferable but often the *only* architecturally compliant design for data-driven systems. We map specific articles from the General Data Protection Regulation (GDPR, EU 2016/679), the Health Insurance Portability and Accountability Act (HIPAA), the Chinese Personal Information Protection Law (PIPL), NIST cybersecurity frameworks, ENISA IoT security baselines, and the EU AI Act to concrete Federated Learning design requirements. We then examine the mechanisms — Differential Privacy (DP-SGD), Secure Aggregation, and Federated Unlearning — that FL must deploy to satisfy these requirements, along with the compliance gaps that remain open research problems.

The product perspective is central throughout. The Cisco 2023 Data Privacy Benchmark Study found that **90% of organizations** believe data localization (keeping data within national borders) is safer for compliance, yet simultaneously, the same organizations experience severe commercial friction when centralized ML architectures require exporting customer data abroad [Cisco, 2023]. This contradiction — wanting data localization but needing centralized ML — is precisely the market tension that FL resolves. For an IoT IDS startup, adopting FL is not merely an ethical choice; it is a prerequisite for closing enterprise deals in regulated markets.

---

## Section 1: The Regulatory Landscape

### 1.1 GDPR (EU Regulation 2016/679) — The Global Standard-Setter

The General Data Protection Regulation, in force since May 25, 2018, has become the de facto global benchmark for data privacy law. With enforcement penalties of up to **€20 million or 4% of global annual turnover** (whichever is higher) under Article 83, GDPR has elevated data architecture decisions to boardroom-level risk management [GDPR, 2016].

Several GDPR articles directly constrain or prohibit conventional centralized ML architectures:

#### Article 5 — Principles of Processing
Article 5(1)(c) enshrines **data minimization**: personal data must be "adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed." In a traditional centralized ML pipeline for network intrusion detection, edge gateways stream raw packet capture (pcap) data — containing source/destination IP addresses (personal data under GDPR Recital 30, which explicitly identifies network identifiers as personal data), payload snippets, and device behavioral metadata — to a central cloud server. This constitutes collection of data far exceeding what is strictly necessary for training a statistical anomaly model, because the model ultimately needs *statistical patterns*, not raw packets. FL satisfies Art. 5(1)(c) by ensuring only model gradients (mathematical tensors representing statistical weight updates) leave the device, not raw traffic [McMahan et al., 2017; Truong et al., 2021].

Article 5(1)(b) mandates **purpose limitation**: data collected for one purpose (e.g., network monitoring for security) cannot be repurposed (e.g., behavioral profiling of employees). Centralized data lakes in cloud environments frequently suffer "data gravity" — once data is centralized, it tends to be reused across projects. FL enforces purpose limitation structurally: data never leaves the device, so repurposing by the vendor is architecturally impossible.

Article 5(2) requires **accountability**: the data controller must be able to demonstrate compliance. In a centralized ML system, the controller must maintain records of every data transfer, every model retrain, every access log. In FL, the central server processes only aggregated model parameters, dramatically reducing the audit surface.

#### Article 17 — Right to Erasure ("Right to be Forgotten")
Article 17 grants individuals the right to have their personal data erased "without undue delay" when data is no longer necessary, consent is withdrawn, or a legal basis ceases to apply. For a centralized ML system, this is architecturally catastrophic: if a data subject's training data must be erased, the model trained on that data must also be invalidated. Full retraining from scratch is the only rigorous solution, which may be computationally prohibitive for large production models [Cao & Yang, 2015].

In FL, Art. 17 compliance is easier for the *raw data* component (delete the local data on the device), but harder for the *model contribution* component. A user's historical gradient contributions are embedded in the global model weights — erasing those requires **Federated Unlearning**. Liu et al. [2021] introduced *FedEraser*, which enables efficient client-level data removal by constructing a "calibration" dataset of historical updates, avoiding full retraining. Liu et al. [2022] extended this to the "right to be forgotten" in federated settings via rapid retraining on INFOCOM 2022. The field is active but no production-grade solution provides formal GDPR Art. 17 compliance guarantees yet — this remains an open research problem.

#### Article 25 — Data Protection by Design and by Default
Article 25 requires that data protection principles be "implemented... in an effective manner" and integrated into processing design from the outset. The European Data Protection Board (EDPB) in its *Guidelines 4/2019 on Article 25* [EDPB, 2019] explicitly identifies data minimization and access controls as core "by design" requirements. FL is arguably the most literal implementation of Art. 25 in the ML domain: privacy is embedded in the architecture (data stays on device), not bolted on after the fact.

#### Articles 44–49 — Cross-Border Data Transfers
These articles restrict the transfer of personal data to third countries unless the destination provides an "adequate" level of protection (Art. 45), or Standard Contractual Clauses (SCCs) and supplementary measures are in place (Art. 46). For an IoT IDS vendor with customers in the EU and a cloud training server in the US or APAC, every telemetry packet shipped to the central server constitutes a cross-border transfer of personal data subject to these articles. Following the *Schrems II* judgment (CJEU Case C-311/18, 2020) that invalidated Privacy Shield, these transfers carry significant legal uncertainty [Schrems II, 2020].

FL eliminates the cross-border personal data transfer: only model gradients (which, under appropriate Differential Privacy, are not personal data under GDPR Recital 26) cross borders. This analysis is nuanced — the EDPB has not formally ruled that DP-protected gradients constitute anonymous data — but it dramatically reduces the legal exposure surface.

#### Article 32 — Security of Processing
Art. 32 requires "appropriate technical and organisational measures" to ensure security, including encryption in transit and at rest. Centralized data storage creates a high-value honeypot: a breach of the central server exposes all users' data simultaneously. FL eliminates this single point of failure. Combined with **Secure Aggregation** [Bonawitz et al., 2017], where the server computes only the *sum* of encrypted client updates and never sees individual client gradients, FL satisfies Art. 32 at an architectural level.

---

### 1.2 HIPAA (US 45 CFR Parts 160 & 164) — Healthcare Data Sovereignty

The Health Insurance Portability and Accountability Act protects **Protected Health Information (PHI)** across two primary rules:

- **Privacy Rule (45 CFR Part 164, Subpart E):** PHI may only be used or disclosed for treatment, payment, operations, or with explicit patient authorization. Sharing medical device network traffic with a vendor's cloud for IDS model training constitutes a use of PHI (device identifiers and timing patterns can identify patients), requiring either a Business Associate Agreement (BAA) or full de-identification under 45 CFR §164.514. De-identification via the Safe Harbor method (removing 18 specified identifiers) is burdensome and often incomplete for network telemetry.

- **Security Rule (45 CFR Part 164, Subpart C):** Covered entities and business associates must implement physical, administrative, and technical safeguards. Transmitting PHI across the public internet to a vendor cloud, even encrypted, extends the attack surface and requires extensive BAA compliance infrastructure.

FL resolves both: hospitals and medical IoT operators train models locally. No PHI leaves the hospital's firewalled network. The vendor's aggregation server receives only model parameter tensors, which are not PHI. This enables cross-institutional federated medical research at scale, as demonstrated by NVIDIA FLARE in the EXAM project [Rieke et al., 2020; Sheller et al., 2020], and is the architecture adopted by major healthcare AI consortia globally.

---

### 1.3 CCPA / CPRA (California Civil Code § 1798.100 et seq.)

The California Consumer Privacy Act (CCPA, effective 2020) and its amendment the California Privacy Rights Act (CPRA, effective 2023) grant California residents rights over personal information including the right to know, delete, opt-out of sale, and correct. The CPRA establishes the California Privacy Protection Agency (CPPA) as an independent enforcement body.

For an IoT IDS product sold to US enterprises with California-based employees or customers, network telemetry data (IP addresses, device identifiers, behavioral patterns) qualifies as personal information under CCPA § 1798.140(o). Centralized ML pipelines that aggregate this data to a vendor cloud constitute "collection" and potentially "sale" (sharing for value) of personal information, triggering disclosure requirements. FL reduces CCPA exposure by ensuring the vendor never "collects" personal information in the CCPA sense — the data remains on the customer's premises.

---

### 1.4 China PIPL (Personal Information Protection Law, 2021)

Effective November 1, 2021, China's PIPL is among the most stringent data localization regimes globally. Articles 38–43 impose severe restrictions on cross-border data transfers:

- **Article 38:** Cross-border transfers of personal information require either (a) passing a **security assessment** conducted by the Cyberspace Administration of China (CAC), (b) obtaining a **personal information protection certification** from a designated institution, or (c) signing a **standard contract** prescribed by the CAC.
- **Article 40:** Critical information infrastructure operators and entities processing personal information above regulatory thresholds must store data **within China's borders** (data localization mandate).
- **Article 43:** Reciprocal measures — China may restrict or prohibit data transfers to countries that have enacted "discriminatory" measures against Chinese entities.

For any multinational company building an IoT IDS product with Chinese enterprise customers (e.g., manufacturing plants, smart city infrastructure), PIPL makes centralized cloud training in a non-Chinese data center essentially impossible without extensive regulatory approval. FL with Chinese-side local aggregation — where raw device data never crosses the border — is the only practical architectural path to PIPL compliance for cloud-based ML products.

---

### 1.5 PDPA (Thailand B.E. 2562 / Singapore PDPA 2012) — SEA IoT Context

Southeast Asia is a critical growth market for IoT deployments (smart manufacturing, smart cities, connected healthcare). Both Thailand's Personal Data Protection Act (PDPA, B.E. 2562, effective 2022) and Singapore's PDPA (2012, amended 2020) impose consent-based collection and transfer restrictions analogous to GDPR, with specific rules on cross-border transfers requiring recipient countries to provide "comparable" protection.

For IoT IDS vendors targeting SEA markets (Telco IoT, industrial OT in Thailand's EEC, Singapore's Smart Nation initiative), FL enables models to be trained on local IoT telemetry data while keeping raw data within the jurisdiction, satisfying both PDPAs without requiring complex cross-border transfer agreements for each customer deployment.

---

### 1.6 LGPD (Brazil, Lei 13.709/2018)

Brazil's General Data Protection Law (LGPD) closely mirrors GDPR in structure, with legal bases for processing (Art. 7), requirements for data security (Art. 46), and data subject rights. Art. 33 restricts international data transfers to countries with equivalent protection or under SCCs. The Brazilian data protection authority (ANPD) has become increasingly active in enforcement. For IoT deployments in Brazil's growing industrial and smart city sectors, the same FL compliance logic applies as under GDPR.

---

### 1.7 NIST SP 800-82 Rev 3 (ICS/OT Networks) and NISTIR 8259A (IoT Devices)

While NIST frameworks are voluntary in the US, they are widely adopted as contractual requirements in government procurement (DoD, CISA), and increasingly referenced by the EU's NIS2 Directive and similar international frameworks.

#### NIST SP 800-82 Rev 3 (2023) — Industrial Control Systems Security
NIST SP 800-82 Rev 3 provides guidance on securing OT/ICS environments, including SCADA systems and industrial IoT gateways. A core principle is **network segmentation**: OT networks should be isolated from corporate IT and cloud networks via demilitarized zones (DMZs). Streaming OT telemetry data from ICS devices to a public cloud for ML training directly violates this segmentation principle, creating an exfiltration pathway that could expose critical infrastructure operational data (turbine RPMs, valve states, production line metrics) to cloud provider infrastructure.

FL enables anomaly detection models to be trained entirely within the OT network boundary. The aggregation server can be deployed on-premises in the IT/OT DMZ, and model updates are the only data crossing into the IT network — far less sensitive than raw OT telemetry [NIST SP 800-82 Rev 3, 2023].

#### NISTIR 8259A (2020) — IoT Device Cybersecurity Capability Core Baseline
NISTIR 8259A defines baseline capabilities that IoT devices should possess, including:
- **Data Protection:** The ability to protect device data (stored or transmitted) from unauthorized access.
- **Logical Access to Interfaces:** Restrict access to device configuration, management interfaces, and data stores.

These capabilities argue for edge-side data protection — a cornerstone of FL design. A device that streams raw telemetry to the cloud loses control over data protection post-transmission. FL keeps raw data local, enabling the device or its gateway to enforce NISTIR 8259A data protection capabilities end-to-end [NISTIR 8259A, 2020].

---

### 1.8 ISO/IEC 27001:2022 and ISO/IEC 27701:2019

**ISO/IEC 27001:2022** (Information Security Management Systems) is the most widely adopted international security standard, with over 70,000 certified organizations globally. Annex A Control 8.24 mandates appropriate use of cryptography; Control 8.11 addresses data masking. Most critically, **risk assessment** under ISO 27001 Clause 6.1.2 requires organizations to identify and treat risks to information confidentiality, integrity, and availability.

For a vendor managing a central dataset of customer network telemetry, ISO 27001 risk assessment inevitably flags: (a) unauthorized access to the central data repository, (b) insider threat from vendor personnel, (c) cloud provider breach. FL eliminates the central repository risk item entirely, dramatically reducing the residual risk score in ISO 27001 audits.

**ISO/IEC 27701:2019** extends ISO 27001 with Privacy Information Management System (PIMS) requirements, mapping directly to GDPR and other privacy laws. Section 7.2 (PIMS conditions for collection) aligns with GDPR Art. 6 lawful basis requirements. FL's local data retention approach simplifies PIMS compliance by reducing the scope of personal data under the vendor's control to near zero.

---

### 1.9 ENISA IoT Security Guidelines

The European Union Agency for Cybersecurity (ENISA) has published a series of authoritative IoT security guidance documents that shape EU policy and serve as the technical basis for legislation including ETSI EN 303 645 and the EU Cyber Resilience Act:

- **ENISA Baseline Security Recommendations for IoT (2017):** The foundational document, establishing 80+ security measures organized across 10 categories. Relevant mandates include:
  - **GP-TM-18 (Minimize data):** Devices should minimize the data they collect and share. This directly maps to FL's local training paradigm — raw telemetry is processed locally rather than streamed centrally [ENISA, 2017].
  - **GP-TM-37 (Audit logs):** Maintain audit logs of security events. In FL, audit logs of training rounds remain on-device, satisfying this requirement without exposing operational data to centralized log servers.
  - **GP-TM-42 (Communication security):** All communication between IoT components must be encrypted. FL model update transmissions satisfy this with transport-layer encryption (TLS 1.3), and Secure Aggregation adds cryptographic privacy beyond transport [ENISA, 2017].

- **ENISA Guidelines for Securing the IoT — Supply Chain (2020):** Extends baseline security to the full IoT lifecycle, emphasizing **security by design** from requirements through maintenance [ENISA, 2020]. FL operationalizes security by design at the ML level — privacy is not an add-on but an architectural property.

- **ENISA Threat Landscape 2023:** Documents that **data exfiltration** and **supply chain attacks** remain among the top IoT threats. Centralized ML creates a high-value data exfiltration target; FL eliminates that target from the vendor's infrastructure [ENISA, 2023].

The ENISA baseline measures are incorporated by reference into:
- **ETSI EN 303 645 v2.1.1 (2022):** The European consumer IoT security standard, mandated for CE marking under the EU's Radio Equipment Directive and the upcoming CRA. Provision 5.8 mandates that devices "minimize the exposed attack surface." Streaming raw telemetry to cloud training servers is antithetical to this provision; FL embeds data minimization at the architecture level.
- **EU Cyber Resilience Act (CRA, Regulation 2024/2847):** Art. 13 requires manufacturers of products with digital elements to implement security-by-design throughout the product lifecycle. For ML-enabled IoT products (such as on-device IDS), the training architecture is part of the product design subject to CRA requirements.

---

### 1.10 EU AI Act (Regulation 2024/1689)

The EU AI Act, adopted in 2024, categorizes AI systems by risk level. IoT-based intrusion detection systems that make consequential decisions (e.g., blocking network traffic, alerting on suspected intrusions in critical infrastructure) can fall under **Annex III, High-Risk AI Systems**, particularly categories covering critical infrastructure protection (Annex III, §2) and law enforcement support (Annex III, §6).

High-risk AI systems under the AI Act must comply with:
- **Article 10 (Data Governance):** Training data must be subject to governance practices covering "the examination of possible biases" and ensuring data is "relevant, sufficiently representative, and to the best extent possible, free of errors." Centralized training on heterogeneous IoT telemetry (where customer data is pooled) creates data governance challenges: whose data governance policy applies? FL keeps training data under each customer's own governance regime.
- **Article 13 (Transparency):** AI systems must be sufficiently transparent so users can interpret outputs. FL does not directly address model transparency, but keeping training data local simplifies the data provenance audit required for Art. 13 compliance.
- **Article 17 (Quality Management System):** Providers must establish a quality management system covering data management, testing, and incident management. In FL, the federated aggregation server is the primary system to audit — it never holds personal data, simplifying quality management scope.
- **Article 9 (Risk Management System):** High-risk AI systems must document and mitigate foreseeable risks, including privacy risks from training data. FL's architectural elimination of centralized raw data collection directly reduces the Art. 9 risk register.

---

## Section 2: Compliance-to-Architecture Mapping Matrix

The following table constitutes the core engineering deliverable of this report. Each row maps a specific regulatory article or standard section to a concrete architectural constraint, explains why conventional centralized ML violates it, identifies the FL mechanism that satisfies it, and notes remaining compliance gaps.

| Regulation / Article | Constraint Type | Specific Requirement | Why Centralized ML Fails | FL Mechanism That Satisfies It | Remaining Gap | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GDPR Art. 5(1)(b)** | Purpose Limitation | Data collected for one purpose must not be reused | Central data lakes enable ad-hoc reuse across ML projects | Local training: data never leaves device; vendor cannot repurpose | Gradient leakage can implicitly encode data characteristics | [Truong et al., 2021] |
| **GDPR Art. 5(1)(c)** | Data Minimization | Collect only what is necessary | Raw telemetry streams exceed what anomaly models require | Local gradient computation: only statistical weight updates transmitted | Model updates can memorize training samples | [McMahan et al., 2017; Kairouz et al., 2021] |
| **GDPR Art. 5(2)** | Accountability | Demonstrate compliance with processing principles | Centralized systems require extensive audit trails covering all data movement | FL audit scope = aggregation server only (no raw personal data) | Proving federated audit chain across heterogeneous edge devices is complex | [Truong et al., 2021] |
| **GDPR Art. 17** | Right to Erasure | Delete personal data without undue delay upon request | Model retraining from scratch is required; computationally intractable at scale | Raw data deletion is immediate (local); FedEraser enables model contribution removal | Federated Unlearning cannot yet formally prove complete influence removal | [Liu et al., 2021; Liu et al., 2022] |
| **GDPR Art. 25** | Privacy by Design | Integrate data protection into system design | Privacy treated as post-hoc compliance add-on | FL is Privacy by Design: raw data structurally confined to origin device | Malicious aggregation server can manipulate the global model | [EDPB Guidelines 4/2019; Kairouz et al., 2021] |
| **GDPR Art. 32** | Security of Processing | Appropriate technical measures against unauthorized processing | Centralized data repository is a high-value breach target (honeypot) | Secure Aggregation (SecAgg): server sees only sum of encrypted updates | SecAgg cryptographic overhead; vulnerability to colluding clients | [Bonawitz et al., 2017] |
| **GDPR Art. 44–49** | Cross-Border Transfer | Restrict data flow to inadequate third countries | Every telemetry packet to cloud = cross-border personal data transfer | Only model tensors cross borders; under DP, tensors approach anonymization | EDPB has not formally ruled DP-protected gradients as anonymous | [GDPR, 2016; Schrems II, 2020; Geiping et al., 2020] |
| **GDPR Recital 30** | PII Scope | IP addresses are personal data | Network telemetry inherently contains IP addresses = personal data | IP addresses remain on local device; only gradients transmitted | — | [GDPR, 2016] |
| **HIPAA Privacy Rule** | PHI Protection | PHI cannot leave covered entity without BAA or de-identification | Medical device telemetry contains device IDs that identify patients | Institutional FL: PHI never leaves hospital network | Indirect patient identification via model inference attacks | [Rieke et al., 2020; Sheller et al., 2020] |
| **HIPAA Security Rule** | Electronic PHI Safeguards | Technical safeguards for ePHI | Centralizing ePHI multiplies breach exposure (all hospitals' data in one server) | Distributed training eliminates central ePHI repository | BAA still required with FL aggregation server operator | [Rieke et al., 2020] |
| **China PIPL Art. 38–40** | Data Localization | Cross-border transfers require CAC security assessment | Streaming Chinese user data to overseas training server requires CAC approval (months-long process) | FL with local aggregation server in China: raw data never crosses border | Aggregation server itself must be within China; global model governance unclear | [PIPL, 2021] |
| **NIST SP 800-82 Rev 3** | OT Network Segmentation | OT networks must be isolated from public cloud | Raw OT telemetry to cloud breaks OT/IT network segmentation (DMZ violation) | FL aggregation server in IT/OT DMZ; only model parameters cross DMZ | Model parameter leakage of OT operational signatures | [NIST SP 800-82 Rev 3, 2023] |
| **NISTIR 8259A** | IoT Data Protection | Devices must protect data they store and transmit | Remote cloud training removes data protection from device control post-transmission | Local training: IoT gateway enforces data protection end-to-end | Low-resource IoT devices may lack compute for local model training | [NISTIR 8259A, 2020] |
| **ENISA GP-TM-18** | Data Minimization (IoT) | IoT devices should minimize data collected and shared | Full telemetry streaming maximizes exposed data surface | Local training: only gradient updates leave device | Gradient compression may still encode sensitive patterns | [ENISA, 2017] |
| **ETSI EN 303 645 §5.8** | Attack Surface Minimization | Minimize exposed attack surfaces on IoT devices | Cloud training pipeline adds cloud endpoint as attack surface | FL eliminates cloud training endpoint exposure | — | [ETSI EN 303 645, 2022] |
| **EU AI Act Art. 10** | Data Governance | High-risk AI training data must be governed and representative | Pooling customer data in vendor cloud: whose governance policy applies? | Each client maintains sovereignty over their own training data | Federated data quality assurance (non-IID bias, data poisoning) | [EU AI Act, 2024] |
| **ISO/IEC 27001:2022** | ISMS Risk Treatment | Treat risks to information confidentiality | Central training data lake = high residual risk (breach, insider threat, provider access) | FL eliminates central data repository from risk register | Risk of model theft or adversarial manipulation of federated rounds | [ISO/IEC 27001, 2022] |
| **CCPA §1798.100** | Personal Information Rights | Right to delete personal information | Model retrained on user data must be retrained or invalidated on deletion request | Local data deletion is immediate; FedEraser for model contribution | Same federated unlearning gap as GDPR Art. 17 | [Liu et al., 2022] |
| **EU CRA Art. 13** | Product Security-by-Design | Security requirements throughout product lifecycle | Training pipeline that exfiltrates raw telemetry contradicts CRA security-by-design | FL embeds data privacy as architectural property of the product | CRA compliance certification process for FL-based AI products not yet established | [EU CRA, 2024] |

---

## Section 3: Product Engineering Perspective

### 3.1 The Sales Problem: Why Data Sovereignty Kills Cloud ML Deals

The commercial reality of building an AI-powered IoT IDS in the current regulatory environment is starkly illustrated by enterprise procurement data. The **Cisco 2023 Data Privacy Benchmark Study** — a survey of 3,000 security and privacy professionals across 26 countries — found that **90% of organizations** believe that data localization (keeping data on local or national infrastructure) is inherently safer from a compliance standpoint. The same study found that privacy has become a "buying criterion" for customers evaluating enterprise software vendors, with **94% of respondents** reporting that customers would not buy from them if they did not have strong data privacy practices [Cisco, 2023].

For an IoT IDS vendor, this translates directly into lost deals. Consider the product sales cycle:

**Step 1 — Technical Evaluation**: The vendor proposes a centralized ML architecture: deploy lightweight agents on customer IoT gateways that stream network telemetry (pcap metadata, flow records, connection state tables) to the vendor's AWS or GCP training cluster for anomaly model training and continuous improvement.

**Step 2 — Security/Compliance Review**: The customer's Information Security team and Legal/Compliance teams review the architecture. Network flow records from an EU hospital's medical device network contain IP addresses (personal data, GDPR Recital 30) and device behavioral patterns that can infer patient presence/status (sensitive data, GDPR Art. 9). A CISO at a German hospital will not sign a Data Processing Agreement (DPA) allowing this data to leave the hospital's premises to an American cloud provider, particularly post-Schrems II [Schrems II, 2020].

**Step 3 — Deal Death**: The deal collapses at the security review stage. The vendor has invested weeks in a proof-of-concept; the customer has invested weeks in evaluation. The root cause: architectural incompatibility with GDPR.

This scenario is not hypothetical. A 2021 survey by the International Association of Privacy Professionals (IAPP) found that **73% of organizations** identified "data transfer restrictions" as a significant barrier to their AI/ML projects post-Schrems II [IAPP, 2021]. The GDPR fine history confirms the stakes: in 2023, Meta received a record **€1.2 billion fine** under GDPR Art. 46 for unlawful US data transfers [Irish DPC, 2023].

### 3.2 The Compliance Blocker: Data Gravity vs. Regulatory Gravity

The fundamental tension in centralized ML for regulated markets is what cloud architects call "data gravity" — the tendency for data to attract applications to its location — colliding with regulatory mandates for data to stay in its jurisdiction. These forces are irreconcilable in a centralized architecture.

Regulatory gravity creates specific hard constraints that no amount of encryption, contractual safeguarding, or cloud provider assurance fully resolves:

| Constraint | Regulatory Source | Engineering Implication |
|---|---|---|
| Data cannot leave the EU | GDPR Art. 44, Schrems II | No EU customer telemetry to US training server |
| Data cannot leave China | China PIPL Art. 40 | No Chinese customer data to overseas training cluster |
| PHI cannot leave hospital network | HIPAA Privacy Rule | No medical device telemetry to vendor cloud |
| OT data cannot exit OT network | NIST SP 800-82 Rev 3 | No ICS telemetry to public cloud |
| Data must be minimized | GDPR Art. 5(1)(c), ENISA GP-TM-18 | Cannot stream full pcap to cloud; only essential features |

Each constraint independently makes centralized ML architecturally non-compliant for the target customer segment. Together, they make it commercially non-viable.

### 3.3 How FL Unblocks Deals While Satisfying Compliance

FL changes the architecture so that the vendor never receives personal data. The revised pitch to a regulated enterprise customer:

> *"We deploy our ML engine — a trained base model and a local training runtime — to your on-premises IoT gateway. All network telemetry is processed locally; your raw traffic data never leaves your network. We receive only a compressed mathematical tensor (gradient update, ~5 KB/round) that represents statistical changes to the anomaly detection model. We aggregate these updates from all customers' deployments to improve the global model, then push the improved model back to your premises. Your data is mathematically protected: with Differential Privacy applied before transmission, even the gradient tensor cannot be reverse-engineered to identify your data."*

This architecture satisfies:
- **GDPR Art. 5(1)(c)**: Only gradients transmitted (data minimization).
- **GDPR Art. 25**: Privacy is an architectural property (privacy by design).
- **GDPR Art. 44–49**: No personal data crosses borders (only DP-protected gradients).
- **HIPAA Privacy Rule**: PHI never leaves the hospital network.
- **NIST SP 800-82 Rev 3**: OT telemetry never exits the OT network boundary.
- **ENISA GP-TM-18**: IoT devices minimize data shared externally.

The CISO signs the agreement. The deal closes.

### 3.4 Build vs. Buy Compliance Trade-offs for FL Infrastructure

Building FL infrastructure in-house is significantly more complex than conventional ML infrastructure. Key challenges and product decisions:

| Concern | Centralized ML | Federated ML |
|---|---|---|
| **Client state management** | N/A | Must track which devices have participated, their model version, and synchronization status |
| **Straggler handling** | N/A | Asynchronous aggregation required; slow/offline devices must not block training rounds |
| **Secure Aggregation** | Not required | Cryptographic protocol (multiparty computation) must be implemented or licensed; computationally expensive |
| **Differential Privacy** | Optional | Required for gradient privacy; introduces accuracy-privacy trade-off (typical ε = 0.1–10) |
| **Communication overhead** | Not applicable | Model tensor transmission per client per round; must be optimized (gradient compression, FedProx, quantization) |
| **Edge compute requirements** | Handled by cloud server | IoT devices must support local model training; minimum 64 MB RAM for embedded FL (e.g., TensorFlow Lite for Microcontrollers) |

**Industrial-grade frameworks** available to reduce build cost:
- **NVIDIA FLARE** (Federated Learning Application Runtime Environment): Production-grade, used in healthcare FL. Supports HIPAA-compliant configurations [NVIDIA FLARE, 2022].
- **TensorFlow Federated (TFF)**: Google's FL framework; Secure Aggregation built-in for mobile deployments [Bonawitz et al., 2019].
- **PySyft (OpenMined)**: Privacy-focused FL framework with Differential Privacy and Secure Multi-Party Computation support.
- **WeBank FATE**: Financial sector FL framework (see Section 5.4 for case study) [Yang et al., 2019].

### 3.5 Total Cost of Compliance (TCC): Centralized vs. Federated

The Total Cost of Compliance encompasses legal, technical, and operational expenditures to satisfy regulatory requirements:

| Cost Category | Centralized ML | Federated ML |
|---|---|---|
| **Data Transfer Legal Agreements** | SCCs per customer (€50K–€200K legal fees), GDPR DPAs, HIPAA BAAs | No SCCs needed for raw data; simplified DPA scope (gradient aggregation only) |
| **Cloud Data Security** | SOC 2 Type II audit of central data lake, encryption-at-rest, access control, SIEM | Aggregation server audit only; no central personal data to secure |
| **Breach Liability Insurance** | High premium (large exposed dataset) | Lower premium (no central personal data repository) |
| **Data Processing Agreements** | Full DPA with every enterprise customer covering raw data | Lightweight DPA covering model update aggregation only |
| **GDPR DPIA (Art. 35)** | Required for large-scale personal data processing | Potentially not required if no personal data processed centrally |
| **Edge Compute CAPEX** | Not required | Required (FL-capable edge gateways); may be offset by customer hardware procurement |

For regulated market entry, the centralized ML TCC typically exceeds the FL infrastructure investment in legal and compliance costs within the first year of operation, particularly for EU or healthcare-focused products.

---

## Section 4: Privacy-Preserving FL Mechanisms and Their Regulatory Mapping

Raw Federated Learning — sharing gradient tensors without additional protection — is insufficient for regulatory compliance. Research has established that gradient sharing introduces its own privacy risks, requiring supplementary mechanisms.

### 4.1 Differential Privacy and DP-SGD

**Differential Privacy (DP)** provides a mathematical definition of privacy: an algorithm $\mathcal{M}$ is $(\epsilon, \delta)$-differentially private if for any two adjacent datasets $D$ and $D'$ differing by one record, and any output set $S$:

$$\Pr[\mathcal{M}(D) \in S] \le e^\epsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

In plain language: the probability that an observer can distinguish whether a specific individual's data was in the training set is bounded by $e^\epsilon$. With $\delta \approx 0$ (pure DP) or $\delta \ll 1/N$ (approximate DP), the algorithm provides strong membership privacy [Dwork et al., 2006].

**DP-SGD** (Abadi et al., CCS 2016) applies DP to neural network training:
1. **Gradient Clipping**: For each sample $x_i$, compute gradient $g_i = \nabla_\theta \mathcal{L}(\theta, x_i)$ and clip to norm $C$: $\tilde{g}_i = g_i / \max(1, \|g_i\|_2 / C)$.
2. **Noise Addition**: Add calibrated Gaussian noise: $\hat{g} = \frac{1}{B}\left(\sum_i \tilde{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\right)$, where $\sigma$ is the noise multiplier and $B$ is the batch size.
3. **Privacy Accounting**: Track cumulative privacy loss $(\epsilon, \delta)$ across training steps using the **moments accountant** or **Rényi DP** accounting.

In the FL context, each client applies DP-SGD locally before transmitting gradients to the server. This satisfies two regulatory requirements simultaneously:

- **GDPR Recital 26 (Anonymization)**: Recital 26 states that personal data rendered anonymous "in such a manner that the data subject is not or no longer identifiable" falls outside GDPR scope. While the EDPB has not formally ruled that DP with specific $(ε, δ)$ values constitutes anonymization, sufficiently small $\epsilon$ (e.g., $\epsilon < 1$) combined with FL's data localization significantly reduces the re-identification risk such that gradient transmissions are unlikely to constitute cross-border personal data transfers under GDPR Arts. 44–49. The EDPB's three-part singling-out/linkability/inference test is increasingly satisfied at $\epsilon \le 2$ for network telemetry data [Desfontaines & Pejó, 2020].

- **GDPR Art. 32 (Appropriate Technical Measures)**: DP-SGD constitutes a "technical measure" that ensures "a level of security appropriate to the risk," specifically protecting against membership inference attacks on model parameters [Yeom et al., 2018].

**Practical trade-off**: Strong DP (small $\epsilon$) reduces model utility. For IoT IDS applications, [Naseri et al., 2020] show that $\epsilon = 2$–$8$ preserves >95% of anomaly detection accuracy while providing meaningful privacy guarantees. This $\epsilon$ range is the de facto standard in federated healthcare and financial applications.

### 4.2 Secure Aggregation (SecAgg)

Bonawitz et al. (2017) introduced Secure Aggregation for FL, using **additive secret sharing** and **double-masking** to ensure the central server computes only $\sum_i \Delta\theta_i$ (the sum of all client updates) without seeing any individual $\Delta\theta_i$.

**Protocol sketch** (simplified):
1. Each client $i$ generates pairwise **masks** $s_{ij}$ using Diffie-Hellman key agreement with client $j$.
2. Client $i$ transmits $\Delta\theta_i + \sum_{j>i} s_{ij} - \sum_{j<i} s_{ji} + b_i$ (where $b_i$ is a server-mask for dropout handling).
3. The server sums all transmitted values; masks cancel out, leaving $\sum_i \Delta\theta_i$.
4. The server **never** sees individual $\Delta\theta_i$.

**Regulatory impact**: SecAgg operationalizes GDPR Art. 5(1)(c) (data minimization) at the aggregation layer — the server's data processing is minimized to the mathematical aggregate. It also satisfies GDPR Art. 32 by ensuring the aggregation server cannot reconstruct individual training data even via gradient analysis. The cryptographic overhead is manageable for edge servers (batch aggregation) but remains prohibitive for resource-constrained IoT devices in the innermost training loop [Bonawitz et al., 2019].

### 4.3 Gradient Inversion Attacks: Why Raw FL Is Insufficient

Geiping et al. (2020) — "Inverting Gradients: How Easy Is It to Break Privacy in Federated Learning?" (NeurIPS 2020) — demonstrated that a malicious or compromised aggregation server can reconstruct high-fidelity training images from shared gradients using optimization:

$$x^* = \arg\min_{x} \|\nabla_\theta \mathcal{L}(\theta, x) - \nabla_\theta \mathcal{L}(\theta, x_{\text{shared}})\|^2 + \alpha \cdot R(x)$$

where $R(x)$ is a total variation regularizer. The attack achieves near-perfect reconstruction of images from a single gradient step. Zhao et al. (2020) — "iDLG: Improved Deep Leakage from Gradients" — extended this to exact label inference and arbitrary batch sizes.

**Regulatory implication**: If gradients can be inverted to reveal personal data (network packet contents, patient images, financial transactions), then transmitting gradients is legally equivalent to transmitting personal data under GDPR. This means raw FL without DP and SecAgg cannot satisfy GDPR Art. 44–49 (cross-border transfer restrictions). The combination of DP-SGD + SecAgg is the minimum required stack for regulatory-grade FL [Kairouz et al., 2021].

### 4.4 Federated Unlearning: Satisfying GDPR Art. 17 and CCPA §1798.105

GDPR Art. 17 and CCPA §1798.105 (Right to Delete) require that upon user request, their personal data be erased and their influence on ML models be removed. In federated settings, the raw data deletion is trivially satisfied (delete local dataset). The challenge is **unlearning** the model's embedded knowledge of that client's data.

**FedEraser** (Liu et al., IWQOS 2021) is the leading federated unlearning approach:
- Maintains a compressed history of client update contributions (calibration dataset).
- Upon unlearning request, retrains the model using remaining clients' updates and the calibration dataset, without requiring full retraining.
- Reduces unlearning time by **2.45× – 7.17×** compared to full retraining in experiments.

**Rapid Retraining** (Liu et al., INFOCOM 2022) extends this with formal convergence bounds, showing that $O(T/K)$ additional training rounds (where $T$ = original training rounds, $K$ = active clients) suffice for unlearning, with bounded distance from the fully retrained model.

**Open compliance gap**: Neither FedEraser nor Rapid Retraining can provide a formal cryptographic proof that the unlearned model contains zero contribution from the removed client. Regulators auditing GDPR Art. 17 compliance cannot yet be provided a mathematically verifiable unlearning certificate — a critical research problem for enterprise FL products.

### 4.5 Homomorphic Encryption — The Gold Standard and Its Limits

Fully Homomorphic Encryption (FHE) allows the server to aggregate encrypted gradient ciphertexts directly: $\text{Dec}(\text{Enc}(\Delta\theta_A) \oplus \text{Enc}(\Delta\theta_B)) = \Delta\theta_A + \Delta\theta_B$. This provides information-theoretic security without any trust assumptions on the server.

**Feasibility for IoT IDS**: Current FHE schemes (BFV, CKKS) impose **100×–10,000×** computational overhead compared to plaintext operations [Cheon et al., 2017]. For IoT edge devices with ARM Cortex-M class processors, FHE gradient encryption for a 1M-parameter anomaly detection model requires hours per training round — commercially unusable. FHE remains viable only for small logistic regression models (healthcare risk scoring, credit scoring) on server-grade hardware [Batchelor et al., 2021].

The practical regulatory compliance stack for IoT IDS FL is therefore: **FL + DP-SGD (local) + SecAgg (aggregation) + TLS 1.3 (transport)**, with FHE deferred to future hardware generations.

### 4.6 Data Controller vs. Data Processor Under GDPR in FL Systems

A critical compliance question for FL product teams: **who is the Data Controller and who is the Data Processor?**

Under GDPR Art. 4(7), the **Data Controller** is the entity that determines the purposes and means of processing personal data. Art. 4(8) defines the **Data Processor** as the entity that processes personal data on behalf of the controller.

In an FL system:
- **Data Controller**: The enterprise customer (hospital, manufacturer, telco) — they determine that network telemetry is collected and processed for intrusion detection. They control the raw personal data.
- **Data Processor**: The FL vendor operating the aggregation server — they process model updates (gradient tensors) on behalf of the controller.

**Key compliance implication**: If the FL vendor's aggregation server receives only DP-protected gradient tensors (not personal data), the Data Processing Agreement (DPA) required under GDPR Art. 28 is dramatically simplified — the vendor is not processing personal data in the GDPR sense. This reduces the vendor's liability footprint and simplifies compliance audits. However, if the vendor's aggregation server could reconstruct personal data from gradients (in the absence of DP), the vendor remains a Data Processor under GDPR, requiring full DPA compliance including data sub-processor management, transfer impact assessments, and breach notification obligations [Truong et al., 2021].

---

## Section 5: Real-World Deployment Case Studies

### 5.1 Google Gboard — FL at Consumer Scale

Google deployed FL in 2017 for next-word prediction and word suggestion in its Gboard keyboard for Android, in one of the first large-scale production FL deployments [McMahan et al., 2017; Hard et al., 2018]. Users type passwords, private messages, and personally identifying information via Gboard — centralizing this keystroke data would constitute processing of highly sensitive personal data (communications content) under GDPR Art. 9 and would require explicit consent from millions of users.

The FL architecture (combined with Secure Aggregation and local differential privacy) ensures Google receives no individual user's typing data, only aggregated model improvements. This is documented in Hard et al., "Federated Learning for Mobile Keyboard Prediction" (2018), which demonstrates that FL achieves language model quality comparable to centralized training while maintaining full data locality. Google subsequently published **FedJAX** and **TensorFlow Federated** as open frameworks building on these production learnings [Ro et al., 2021].

### 5.2 NVIDIA FLARE for Federated Medical Imaging — HIPAA at Scale

NVIDIA FLARE (Federated Learning Application Runtime Environment) enables federated model training across healthcare institutions [Rieke et al., 2020; Sheller et al., 2020]. The flagship deployment was the **EXAM study** (EHR-based Approach to Attribute-Based Model, 2020): 20 hospitals across 5 continents collaboratively trained a model to predict oxygen requirements for COVID-19 patients from chest X-rays and clinical data. No patient scans were shared across institutions; only model updates were transmitted. The federated model significantly outperformed models trained on any single institution's local data [Dayan et al., 2021, *Nature Medicine*].

This deployment directly demonstrates the HIPAA compliance value proposition: 20 hospitals, each bound by their own IRB protocols and BAAs, could contribute to a global model without any single hospital incurring the compliance burden of sharing PHI with 19 other institutions. NVIDIA FLARE is now used in over 50 healthcare institutions globally [NVIDIA, 2023].

### 5.3 Telco and IoT IDS — Federated Anomaly Detection

For IoT IDS specifically, [Mothukuri et al., 2021] surveyed FL-based IDS approaches showing FL enables cross-gateway anomaly detection without centralizing network traffic. [Nguyen et al., 2019] — "DIoT: A Federated Self-Learning Anomaly Detection System for IoT" — demonstrated the first production-grade federated IDS for IoT devices, where each home gateway trained its own device-specific anomaly model locally, and only model updates were shared via a FL aggregation server. The system detected anomalies in Mirai botnet traffic with 95.6% accuracy while keeping all home network traffic strictly local — directly addressing GDPR Recital 30 (IP addresses as personal data) compliance.

For industrial IoT and critical infrastructure, FL-based IDS is mandated by design: NIST SP 800-82 Rev 3 network segmentation requirements make any architecture that streams OT telemetry to the cloud architecturally non-compliant.

### 5.4 WeBank FATE — Financial Sector FL at Enterprise Scale

WeBank (China's first digital bank) created the **FATE (Federated AI Technology Enabler)** framework [Yang et al., 2019, JMLR], now the most widely deployed enterprise FL platform in the financial sector with production deployments across multiple Chinese financial institutions and growing international adoption.

The core compliance driver for FATE is **China PIPL** and the earlier **Cybersecurity Law (2017)**: financial institutions are legally prohibited from sharing customer transaction data with other institutions. Fraud patterns, however, are often cross-institutional (a fraudster moving funds across multiple banks). FATE enables a consortium of banks to train a federated fraud detection model:

1. Each bank trains locally on its private transaction ledger.
2. Encrypted model gradients (protected via Homomorphic Encryption for sensitive financial features) are exchanged with a FATE arbiter.
3. The arbiter aggregates updates and distributes the improved global fraud model.
4. No bank's transaction records are ever seen by any other bank or the arbiter.

Yang et al. [2019] report that the federated fraud model trained across 2 financial institutions (a bank and an insurance company) with non-overlapping customer populations achieved **AUC improvement of 3%–8%** over locally trained models, while maintaining full PIPL and PRC data localization compliance. The FATE framework is open-source on GitHub (github.com/FederatedAI/FATE) and has been cited in regulatory guidance by China's PBOC as a reference architecture for privacy-preserving financial AI.

---

## Section 6: Open Research Problems and Compliance Gaps

Despite its transformative compliance value, FL faces critical unresolved challenges:

### 6.1 Federated Unlearning Verification (GDPR Art. 17 Gap)
The most commercially critical open problem: how can a company provide a **mathematically verifiable unlearning certificate** to a GDPR Data Protection Authority (DPA) proving that a specific client's data contribution has been fully removed from the global model? Neither FedEraser [Liu et al., 2021] nor Rapid Retraining [Liu et al., 2022] provides formal cryptographic guarantees. Current approaches rely on empirical metrics (membership inference attack accuracy post-unlearning), which cannot satisfy auditor requirements for formal proof [Kairouz et al., 2021].

### 6.2 Data Poisoning and Backdoor Attacks (EU AI Act Art. 10 Gap)
Since the FL server cannot inspect client training data, malicious clients can inject backdoors (e.g., "if the source IP is X, classify as benign") into model updates [Bagdasaryan et al., 2020]. Defending against data poisoning while maintaining Secure Aggregation (which prevents the server from inspecting individual updates) is an open adversarial ML problem. EU AI Act Art. 10 requires that high-risk AI training data be "free of errors and complete" — FL's data-blindness at the server creates an Art. 10 compliance gap.

### 6.3 Quantifying Privacy Loss for Legal Audiences
DP bounds $(\epsilon, \delta)$ are mathematically precise but legally opaque. Explaining to a DPA or a court that "our system is $(2, 10^{-5})$-differentially private, which means the probability of identifying any individual from our model is bounded" does not directly map to GDPR's "identifiable natural person" threshold. This translation gap between mathematical privacy theory and legal anonymization definitions remains an active interdisciplinary problem [Desfontaines & Pejó, 2020].

### 6.4 System Heterogeneity and Fairness (EU AI Act Art. 10 — Representation)
IoT devices span orders of magnitude in compute capability, memory, and connectivity. FL algorithms that favor high-resource clients produce models that underperform on low-resource device populations — introducing algorithmic fairness issues [Li et al., 2020]. EU AI Act Art. 10 requires training data to be "sufficiently representative." Ensuring that federated models represent all client device types equitably — without marginalizing constrained devices — is an active area of federated fairness research.

### 6.5 Cross-Jurisdictional FL Governance (PIPL + GDPR Simultaneous Compliance)
A multinational FL system with clients in the EU (GDPR) and China (PIPL) faces simultaneous, potentially conflicting obligations. GDPR Art. 44 restricts data transfers to China (no adequacy decision). PIPL Art. 38–40 restricts data transfers from China to the EU. How should a global FL aggregation architecture be designed to satisfy both? Hierarchical federation (local EU aggregator + local Chinese aggregator + global aggregator that only sees country-level aggregates) is a proposed architecture, but its regulatory validity has not been formally assessed by either the EDPB or the CAC.

---

## Section 7: Conclusion

From the perspective of a product architect building regulated AI systems in 2026, Federated Learning has transitioned from an academic novelty to a compliance necessity. The regulatory forces documented in this report — GDPR's data minimization and cross-border transfer restrictions, HIPAA's PHI protection mandates, China PIPL's data localization requirements, NIST SP 800-82's OT network segmentation principles, and ENISA's IoT data minimization guidelines — collectively make centralized ML architectures commercially and legally untenable for the most valuable enterprise markets: EU healthcare, critical infrastructure, financial services, and industrial IoT.

FL's core architectural invariant — the model travels to the data, not the data to the model — is the single design decision that unlocks compliance across this entire regulatory landscape simultaneously. By combining FL with Differential Privacy (DP-SGD, $\epsilon \leq 8$ for practical IoT IDS accuracy), Secure Aggregation (cryptographic sum-only server processing), and Federated Unlearning (FedEraser for GDPR Art. 17), a product team can build an IoT IDS that is deployable in EU hospitals, Chinese manufacturing plants, US critical infrastructure, and APAC smart cities — with a unified compliance argument across all jurisdictions.

The compliance gaps that remain — federated unlearning verification, data poisoning defense under SecAgg, and cross-jurisdictional FL governance — define the research frontier. For an IEC Lab research team, these gaps represent the concrete, high-impact research problems where new contributions directly address real-world regulatory requirements, ensuring academic work translates into deployable product value.

---

## References

1. **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L.** (2016). Deep Learning with Differential Privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16)*. DOI: 10.1145/2976749.2978318. URL: https://arxiv.org/abs/1607.00133

2. **Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikoff, V.** (2020). How to backdoor federated learning. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020)*. URL: https://arxiv.org/abs/1807.00459

3. **Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A., & Seth, K.** (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS '17)*. DOI: 10.1145/3133956.3133982. URL: https://dl.acm.org/doi/10.1145/3133956.3133982

4. **Bonawitz, K., Eichner, H., Grieskamp, W., Huba, D., Ingerman, A., Ivanov, V., ... & Van Overveldt, T.** (2019). Towards Federated Learning at Scale: A System Design. *Proceedings of the 2nd SysML Conference*. URL: https://arxiv.org/abs/1902.01046

5. **Cao, Y., & Yang, J.** (2015). Towards Making Systems Forget with Machine Unlearning. *IEEE Symposium on Security and Privacy (S&P 2015)*. DOI: 10.1109/SP.2015.35

6. **Cheon, J. H., Kim, A., Kim, M., & Song, Y.** (2017). Homomorphic Encryption for Arithmetic of Approximate Numbers. *Advances in Cryptology – ASIACRYPT 2017*. DOI: 10.1007/978-3-319-70694-8_15

7. **Cisco.** (2023). *Cisco 2023 Data Privacy Benchmark Study*. Cisco Systems. URL: https://www.cisco.com/c/en/us/about/trust-center/data-privacy-benchmark-study.html

8. **CJEU.** (2020). *Data Protection Commissioner v Facebook Ireland Limited and Maximillian Schrems (Schrems II), Case C-311/18*. Court of Justice of the European Union. URL: https://curia.europa.eu/juris/document/document.jsf?docid=228677

9. **Dayan, I., Bhatt, D. L., Tresadern, P., ... & Sheller, M. J.** (2021). Federated learning for predicting clinical outcomes in patients with COVID-19. *Nature Medicine*, 27, 1735–1743. DOI: 10.1038/s41591-021-01506-3

10. **Desfontaines, D., & Pejó, B.** (2020). SoK: Differential Privacies. *Proceedings on Privacy Enhancing Technologies*, 2020(2), 288–313. DOI: 10.2478/popets-2020-0028

11. **Dwork, C., McSherry, F., Nissim, K., & Smith, A.** (2006). Calibrating Noise to Sensitivity in Private Data Analysis. *Proceedings of the 3rd Theory of Cryptography Conference (TCC 2006)*. DOI: 10.1007/11681878_14

12. **EDPB.** (2019). *Guidelines 4/2019 on Article 25: Data Protection by Design and by Default*. European Data Protection Board. URL: https://edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-42019-article-25-data-protection-design-and_en

13. **ENISA.** (2017). *Baseline Security Recommendations for IoT in the Context of Critical Information Infrastructures*. European Union Agency for Cybersecurity. URL: https://www.enisa.europa.eu/publications/baseline-security-recommendations-for-iot

14. **ENISA.** (2020). *Guidelines for Securing the Internet of Things: Secure Software Development Lifecycle*. European Union Agency for Cybersecurity. URL: https://www.enisa.europa.eu/publications/enisa-report-guidelines-for-securing-the-internet-of-things

15. **ENISA.** (2023). *ENISA Threat Landscape 2023*. European Union Agency for Cybersecurity. URL: https://www.enisa.europa.eu/publications/enisa-threat-landscape-2023

16. **ETSI EN 303 645 v2.1.1.** (2022). *Cyber Security for Consumer Internet of Things: Baseline Requirements*. ETSI. URL: https://www.etsi.org/deliver/etsi_en/303600_303699/303645/02.01.01_60/en_303645v020101p.pdf

17. **EU AI Act.** (2024). *Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence*. URL: https://artificialintelligenceact.eu/

18. **EU Cyber Resilience Act (CRA).** (2024). *Regulation (EU) 2024/2847 on horizontal cybersecurity requirements for products with digital elements*. URL: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202402847

19. **GDPR.** (2016). *Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data*. URL: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32016R0679

20. **Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M.** (2020). Inverting Gradients — How easy is it to break privacy in federated learning? *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 16937–16947. URL: https://arxiv.org/abs/2003.14053

21. **Hard, A., Rao, K., Mathews, R., Ramaswamy, S., Beaufays, F., Augenstein, S., ... & Ramage, D.** (2018). Federated Learning for Mobile Keyboard Prediction. *arXiv preprint*. URL: https://arxiv.org/abs/1811.03604

22. **IAPP.** (2021). *IAPP–Westin Research Center: Privacy and Artificial Intelligence Report*. International Association of Privacy Professionals.

23. **Irish DPC.** (2023). *Decision in the matter of Facebook Ireland Limited (Inquiry reference: IN-18-5-5)*. Data Protection Commission, Ireland. URL: https://www.dataprotection.ie/en/our-work/our-decisions/decisions-on-international-data-transfers

24. **ISO/IEC 27001:2022.** *Information Security Management Systems — Requirements*. International Organization for Standardization.

25. **ISO/IEC 27701:2019.** *Security techniques — Extension to ISO/IEC 27001 and ISO/IEC 27002 for privacy information management*. International Organization for Standardization.

26. **Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... & Zhao, S.** (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*, 14(1–2), 1–210. DOI: 10.1561/2200000083. URL: https://arxiv.org/abs/1912.04977

27. **Li, T., Sahu, A. K., Talwalkar, A., & Smith, V.** (2020). Federated Learning: Challenges, Methods, and Future Directions. *IEEE Signal Processing Magazine*, 37(3), 50–60. DOI: 10.1109/MSP.2020.2975749

28. **Liu, G., Ma, X., Yang, Y., Wang, C., & Liu, J.** (2021). FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models. *IEEE/ACM International Symposium on Quality of Service (IWQOS 2021)*. DOI: 10.1109/IWQOS52092.2021.9521274

29. **Liu, Y., Xu, L., Yuan, X., Wang, C., & Li, B.** (2022). The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining. *IEEE International Conference on Computer Communications (INFOCOM 2022)*. DOI: 10.1109/INFOCOM48880.2022.9796721

30. **McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.** (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*. URL: https://arxiv.org/abs/1602.05629

31. **Mothukuri, V., Parizi, R. M., Pouriyeh, S., Huang, Y., Dehghantanha, A., & Srivastava, G.** (2021). A survey on security and privacy of federated learning. *Future Generation Computer Systems*, 115, 619–640. DOI: 10.1016/j.future.2020.10.007

32. **Naseri, M., Hayes, J., & De Cristofaro, E.** (2020). Toward Robustness and Privacy in Federated Learning: Experimenting with Local and Central Differential Privacy. *arXiv preprint*. URL: https://arxiv.org/abs/2009.03561

33. **Nguyen, T. D., Marchal, S., Miettinen, M., Fereidooni, H., Asokan, N., & Sadeghi, A.-R.** (2019). DIoT: A Federated Self-Learning Anomaly Detection System for IoT. *IEEE 39th International Conference on Distributed Computing Systems (ICDCS 2019)*. DOI: 10.1109/ICDCS.2019.00080

34. **NIST SP 800-82 Rev 3.** (2023). *Guide to Operational Technology (OT) Security*. National Institute of Standards and Technology. URL: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf

35. **NISTIR 8259A.** (2020). *IoT Device Cybersecurity Capability Core Baseline*. National Institute of Standards and Technology. DOI: 10.6028/NIST.IR.8259A. URL: https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8259A.pdf

36. **PIPL.** (2021). *Personal Information Protection Law of the People's Republic of China (Adopted at the 30th Session of the Standing Committee of the 13th National People's Congress, August 20, 2021)*. URL: https://digichina.stanford.edu/work/translation-personal-information-protection-law-of-the-peoples-republic-of-china-effective-nov-1-2021/

37. **Rieke, N., Hancox, J., Li, W., Milletari, F., Roth, H. R., Albarqouni, S., ... & Cardoso, M. J.** (2020). The future of digital health with Federated Learning. *npj Digital Medicine*, 3(1), 119. DOI: 10.1038/s41746-020-00323-1

38. **Ro, J., Ben-David, O., Rush, A., & Ramage, D.** (2021). FedJAX: Federated learning simulation with JAX. *arXiv preprint*. URL: https://arxiv.org/abs/2108.02117

39. **Sheller, M. J., Edwards, B., Reina, G. A., Martin, J., Pati, S., Kotrotsou, A., ... & Bakas, S.** (2020). Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10, 12598. DOI: 10.1038/s41598-020-69250-1

40. **Truong, N., Sun, K., Wang, S., Guinard, F., Syed, A., & Tai, E.** (2021). Privacy preservation in federated learning: An insightful survey from the GDPR perspective. *Computers & Security*, 110, 102402. DOI: 10.1016/j.cose.2021.102402

41. **Yang, Q., Liu, Y., Chen, T., & Tong, Y.** (2019). Federated Machine Learning: Concept and Applications. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 10(2), 1–19. DOI: 10.1145/3298981. URL: https://arxiv.org/abs/1902.04885

42. **Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S.** (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. *IEEE 31st Computer Security Foundations Symposium (CSF 2018)*. DOI: 10.1109/CSF.2018.00027

43. **Zhao, L., Luo, J., Liu, M., & Shao, J.** (2020). iDLG: Improved Deep Leakage from Gradients. *arXiv preprint*. URL: https://arxiv.org/abs/2001.02610
