# Academic Systems Contributions (Edge AI / IoT)

When asked to formulate academic contributions or write related work for an Edge AI, NIDS, or IoT system running on resource-constrained hardware (e.g., Raspberry Pi, Edge Gateway), strictly follow these guidelines to elevate the work to Tier-1 (Q1 Journal / A* Conference) standards.

## 1. Differentiate Algorithms from Systems
Never blend algorithmic improvements (like Federated Learning, new loss functions) with system optimizations. Structure the introduction with two explicit subsections:
- **Algorithmic Contributions**
- **Systems and Hardware-Software Co-design Contributions**

## 2. Core System Contribution Angles
Always evaluate and propose at least three of the following hardware/system angles:
- **Hardware-Aware Optimization:** How the model leverages specific micro-architectures (e.g., ARM NEON SIMD, Float16 Quantization, L1/L2 Cache utilization).
- **Adaptive Micro-batching:** The trade-off between real-time streaming latency and batched hardware throughput.
- **Micro-architectural Bottleneck Profiling:** Using hardware performance counters (like `perf`) to measure context switching, interrupts, and cache miss rates, moving beyond generic "CPU/RAM %" metrics.
- **Edge-Native Resilience / Overload Control:** Graceful degradation mechanisms during DDoS/anomaly bursts to prevent Out-Of-Memory (OOM) crashes.
- **Security Hardware Penalty:** The quantifiable cost (in ms latency and Joules of energy) of securing the telemetry pipeline (e.g., TLS offloading).

## 3. High-Tier References Requirement
When citing baseline systems or related works, DO NOT invent papers. You must retrieve real, verified papers from top venues such as:
- **Systems & Networking:** USENIX NSDI, ACM SIGCOMM, IEEE INFOCOM
- **Security:** NDSS, USENIX Security, ACM CCS, IEEE TIFS, IEEE TDSC
- **IoT & Edge:** IEEE Internet of Things Journal, ACM/IEEE SEC

*Example verified benchmarks to reference for Edge NIDS:* 
- *Kitsune (NDSS 2018)* for baseline Edge NIDS.
- *Hyperscan (NSDI 2019)* for micro-architectural optimizations.
- *Clipper (NSDI 2017)* for adaptive micro-batching.
- *Pigasus (SIGCOMM 2020)* for hardware-software co-design.

## 4. Prior-Art Triangulation & Cross-Domain Novelty Verification Protocol
Before formulating any new academic contribution or claiming an idea is "novel / unattempted", strictly execute this 4-stage literature and boundary audit:

1. **Cross-Domain Keyword Permutation:**
   Search across at least 3 terminology paradigms:
   - *Core Problem Domain:* (e.g., "Null Foley-Sammon", "Null Space LDA", "KNFST")
   - *Cross-Disciplinary Synonyms:* (e.g., "Novelty Detection" in CV vs. "Anomaly / Intrusion Detection" in Security vs. "Subspace Tracking" in Signal Processing)
   - *Algorithmic Mechanisms:* (e.g., "Rank-1 update", "thin SVD update", "concept drift", "dynamic clustering")

2. **Top-Tier Conference & Journal Archive Verification:**
   Directly query the published proceedings of top venues in the last 10 years:
   - Security: IEEE S&P, USENIX Security, ACM CCS, NDSS, IEEE TIFS, IEEE TDSC
   - AI / ML / CV: NeurIPS, ICML, ICLR, CVPR, ICCV, KDD
   - Systems & Networking: ACM SIGCOMM, USENIX NSDI, IEEE INFOCOM, IEEE IoT-J

3. **Boundary Condition & Invariance Sanity Audit:**
   Stress-test all theoretical shortcuts and analytical assumptions:
   - Does matrix invariance (e.g., $\Delta S_t = 0$) hold only for static re-partitioning, or also when new data arrives (streaming ingestion/eviction)?
   - Are statistical distance distributions truly Gaussian $\mathcal{N}(0, 1)$ or Chi-distributed $\chi(L)$?

4. **Adversarial Poisoning Stress-Test:**
   For any adaptive or dynamic mechanism, formulate the attack vector:
   - How would an intelligent adversary exploit the adaptation (e.g., Boiling Frog / Evasion Poisoning attacks)?
   - Formulate the corresponding countermeasure (e.g., Dual-Quarantine buffer with Flow Entropy Gating).
