# Project Plan: Federated LUNAR Novel Adaptation for IoT Intrusion Detection

## Objective
Design, implement, benchmark and verify novel Federated LUNAR resolving negative gradient conflict and cross-manifold intrusion under Non-IID distributions across 4 IoT datasets on postmaster.iec on branch feature/federated-lunar-novel.

## Methodology & Pattern
Project Pattern with Dual Track (Implementation Track + E2E Testing Track) and strict verification gates (Explorers -> Workers -> Reviewers -> Challengers -> Forensic Auditor).

## Step-by-Step Execution Plan

### Step 0: Survey & Codebase/Environment Mapping
- Dispatch 3 Explorers in parallel:
  * Explorer 1: Map local repository structure, existing LUNAR implementation, LOC-NFST null-space baseline, and dataset loaders.
  * Explorer 2: Inspect remote server `postmaster.iec`, remote directory `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/`, dataset directory `Datascaled/Official_OC_Data/`, python environment `/opt/tljh/user/bin/python3`, and SSH/git configuration.
  * Explorer 3: Investigate mathematical formulations of LUNAR negative generation, gradient conflict dynamics ($\langle \nabla \mathcal{L}_A, \nabla \mathcal{L}_B \rangle < 0$), PCGrad/CAGrad orthogonal projection, and debiased contrastive learning.
- Synthesize Explorer reports into `PROJECT.md` (Feature Inventory, Architecture, Milestones, Interface Contracts, Code Layout).

### Step 1: E2E Testing Infrastructure Track (Parallel)
- Dispatch E2E Testing Orchestrator / Test Writer to establish `TEST_INFRA.md` and multi-tiered opaque-box test cases (Tiers 1-4) covering all features.
- Publish `TEST_READY.md`.

### Step 2: Milestone M1 — Mathematical Formulation & Novel Federated LUNAR Algorithm
- Formulate gradient conflict dynamics under uncoordinated pseudo-negative generation across disjoint client manifolds.
- Implement Cross-Manifold Negative Purging and Orthogonal Gradient Alignment (PCGrad/CAGrad adaptation for distance-ranking LUNAR).
- Gate verification (Explorer -> Worker -> Reviewer -> Challenger -> Auditor).

### Step 3: Milestone M2 — 3-Tier Baseline Hierarchy
- Implement Tier 1: Naive Federated LUNAR (Standard FedAvg with local subspace perturbation).
- Implement Tier 2: FedAvg with Deep Autoencoder (Fed-AE) and PCGrad/FedProx-adapted LUNAR.
- Implement Tier 3: LOC-NFST (Null-Space closed-form baseline analytical bound).
- Gate verification.

### Step 4: Milestone M3 — One-Class Non-IID Benchmark Harness & Evaluation Suite
- Standardized One-Class protocol on 4 datasets: BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT ($M \ge 3$ Non-IID clients, contamination $\le 5\%$).
- Metrics logging: AUC-ROC, F1, FAR, gradient conflict ratio, convergence rounds, per-sample latency, peak memory.
- Gate verification.

### Step 5: Milestone M4 — Remote Server Execution & Automated Verification
- Synchronize branch `feature/federated-lunar-novel` to server `postmaster.iec`.
- Execute full benchmark suite using `/opt/tljh/user/bin/python3`.
- Verify CSV results generated in `outputs/lunar_results/`.
- Gate verification.

### Step 6: Milestone M5 — Documentation, Git Commits & Walkthrough Report
- Generate comprehensive `WALKTHROUGH_FEDERATED_LUNAR.md` with theoretical analysis, ablation study, and peer-reviewed literature citations.
- Clean git commits and push branch `feature/federated-lunar-novel` to GitHub.
- Final gate, audit, and completion report to Sentinel.
