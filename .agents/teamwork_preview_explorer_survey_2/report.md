# Remote Server Environment & Datasets Survey Report

**Author**: Explorer 2 (Remote Server Environment & Datasets)  
**Date**: 2026-09-22 / 2026-09-23  
**Working Directory**: `d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_explorer_survey_2`  
**Target Host**: `postmaster.iec` (IP: `172.16.50.173` / `192.168.50.173`)  
**Remote Directory**: `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst`  
**Remote Python**: `/opt/tljh/user/bin/python3`  
**Remote Datasets**: `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`

---

## 1. Executive Summary

This report delivers a comprehensive, empirical audit of the remote execution environment on `postmaster.iec` for the implementation and benchmarking of the novel **Federated LUNAR (Cross-Manifold Negative Purging & Orthogonal Gradient Alignment)** architecture.

Key findings:
1. **Hardware Powerhouse**: Server `postmaster.iec` is equipped with a high-end **Intel Core i9-13900K** (24 cores / 32 threads, 62 GiB RAM) and a next-generation flagship **NVIDIA GeForce RTX 5090** (32 GB GDDR7 VRAM, Blackwell architecture SM 12.0), providing exceptional compute throughput for batched k-NN graph construction and neural network training.
2. **Pre-Scaled Datasets Confirmed**: All 4 requested benchmark datasets (`BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT`) are fully pre-scaled and verified in `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/` across all 5 standard scalers (`MinMaxScaler`, `StandardScaler`, `RobustScaler`, `QuantileTransformer`, `Normalizer`).
3. **Python Runtime & Frameworks**: Python 3.10.14 is active with **PyTorch 2.11.0.dev+cu128** (full RTX 5090 CUDA acceleration), **Flower 1.30.0** (federated learning orchestrator), and **PyOD 3.5.2** (with native `pyod.models.lunar.LUNAR` confirmed importable).
4. **Git Repository Status**: Current branch is `feature/federated-loc-nfst` tracking `origin/feature/federated-loc-nfst`. The dedicated branch `feature/federated-lunar-novel` can be cleanly branched from `origin/main` or `feature/federated-loc-nfst`.

---

## 2. Server Connectivity & Hardware Specifications

### 2.1 Network Connectivity & SSH Configuration
- **Host Alias**: `postmaster.iec`
- **Primary Host IP**: `172.16.50.173` (also bound to internal interface `192.168.50.173`)
- **SSH User**: `jupyter-iec_duongnt`
- **Authentication**: Public key authentication via `~/.ssh/id_rsa`
- **Kernel Version**: Linux `iec-gv 5.15.0-191-generic #201-Ubuntu SMP x86_64 GNU/Linux`
- **SSH Command Discipline**: To ensure non-blocking execution in automated scripts, all SSH commands should use `ssh -n postmaster.iec "<command>"` to prevent SSH from capturing stdin in headless environments.

### 2.2 Compute & Memory Architecture
| Subsystem | Specification | Current Operating Status |
| :--- | :--- | :--- |
| **CPU** | Intel Core i9-13900K (13th Gen) | 24 physical cores (8 P-cores + 16 E-cores), 32 vCPUs |
| **CPU Architecture** | x86_64, 1 Socket, 1 NUMA node | Load average ~0.25 (idle / high compute capacity available) |
| **Host RAM** | 62 GiB DDR5 | 36 GiB used, 24 GiB available, 22 GiB buff/cache |
| **Swap Space** | 8.0 GiB | 4.5 GiB used, 3.5 GiB free |
| **Primary Storage** | 2.7 TB NVMe LVM (`/dev/mapper/ubuntu--vg-ubuntu--lv`) | 1.8 TB used, **835 GB available** (69% utilization) |
| **Boot Storage** | 2.0 GB `/dev/nvme0n1p2` | 1.4 GB available (24% utilization) |

### 2.3 GPU Accelerator
- **Model**: **NVIDIA GeForce RTX 5090**
- **VRAM Total**: **32,607 MiB (~32 GB)** GDDR7
- **Compute Capability**: **SM 12.0** (Blackwell micro-architecture)
- **NVIDIA Driver**: `580.105.08`
- **CUDA System Version**: `13.0`
- **Power Envelope**: 600W Cap, idle power draw ~35W
- **Current VRAM Usage**: ~17.8 GiB occupied by resident Jupyter kernels, **~14.7 GiB free VRAM** directly allocatable for PyTorch workloads. Batch memory allocation easily accommodates LUNAR MLP and k-NN embeddings.

---

## 3. Python Runtime & Environment Libraries

### 3.1 Python Interpreter
- **Binary Path**: `/opt/tljh/user/bin/python3`
- **Python Version**: `3.10.14 | packaged by conda-forge | (main, Mar 20 2024, 12:45:18) [GCC 12.3.0]`

### 3.2 Deep Learning & Scientific Library Matrix
| Library | Installed Version | Status & Capability |
| :--- | :--- | :--- |
| **PyTorch** | `2.11.0.dev20251222+cu128` | CUDA acceleration active; recognizes RTX 5090 (31.36 GB addressable) |
| **TorchVision** | `0.25.0.dev20251223+cu128` | Pre-installed |
| **TorchAudio** | `2.10.0.dev20251223+cu128` | Pre-installed |
| **Torch-Geometric** | `2.7.0` | Graph neural networks & k-NN graph modeling |
| **Triton** | `3.6.0+git6213a0e8` | Fast GPU kernel compilation |
| **Flower (`flwr`)** | `1.30.0` | Federated learning framework (FedAvg, FedProx, custom aggregators) |
| **PyOD** | `3.5.2` | Confirmed `from pyod.models.lunar import LUNAR` importable |
| **SUOD** | `0.1.4` | Scalable unsupervised outlier detection acceleration |
| **FAISS** | `faiss-cpu 1.14.2` | High-speed dense vector similarity & nearest-neighbor search |
| **Scikit-Learn** | `1.7.2` | Preprocessing, metrics (ROC-AUC, PR-AUC, F1), baseline models |
| **NumPy** | `2.2.6` | Numerical linear algebra |
| **SciPy** | `1.15.3` | Optimization & statistical distributions |
| **Pandas** | `2.2.2` | Tabular data processing |
| **PyArrow** | `25.0.1` | Fast columnar serialization |
| **TensorFlow / Keras** | `2.18.0` / `3.5.0` | Available |
| **WandB** | `0.17.3` | Experiment tracking |
| **Pytest** | `9.1.1` | Unit test execution |
| **uv** | `0.11.33` | Ultra-fast Python package installer |

---

## 4. Benchmark Datasets Survey

Directory: `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/`

All datasets are configured strictly according to the **One-Class Classification (OCC)** protocol:
- **Training Set**: Exclusively **Normal** samples (`label = 0`), zero anomaly contamination.
- **Testing Set**: Mixture of **Normal** (`label = 0`) and diverse real **Attack/Anomaly** samples (`label = 1`).
- All 5 standard scalers are pre-computed for every dataset:
  1. `MinMaxScaler`
  2. `StandardScaler`
  3. `RobustScaler`
  4. `QuantileTransformer`
  5. `Normalizer`

### 4.1 Dataset 1: BoTIoT (Botnet DDoS & Information Theft)
- **Domain**: IoT botnet attacks, keylogging, data exfiltration, reconnaissance, and high-volume DoS/DDoS.
- **Dimensionality**: 35 raw features (IP flows, packet rates, frame sizes, inter-arrival times, sequence stats).
- **Split Configuration**:
  - `Train`: **7,633** samples (100% Normal)
  - `Test`: **16,909** samples (1,909 Normal + 15,000 Anomaly attacks)
- **Exact Remote File Footprint**:
  | Scaler | Train File Size (bytes) | Test File Size (bytes) |
  | :--- | :--- | :--- |
  | `MinMaxScaler` | 2,809,869 | 6,943,296 |
  | `StandardScaler` | 3,719,812 | 8,569,473 |
  | `QuantileTransformer` | 2,791,021 | 6,899,897 |
  | `RobustScaler` | 2,845,676 | 7,357,378 |
  | `Normalizer` | 3,045,649 | 7,287,529 |

### 4.2 Dataset 2: EdgeIIoTset (Industrial IoT Multi-Protocol Traffic)
- **Domain**: Industrial IoT (IIoT) testbed incorporating 14 cyber-attack vectors (DDoS UDP/TCP/HTTP, SQL injection, MITM, Ransomware, Vulnerability scanners, XSS) over heterogeneous protocols (MQTT, Modbus, CoAP, HTTP).
- **Dimensionality**: **42 numerical features** (excluding label).
- **Split Configuration**:
  - `Train`: **30,000** samples (100% Normal)
  - `Test`: **30,000** samples (15,000 Normal + 15,000 Anomaly attacks)
- **Exact Remote File Footprint**:
  | Scaler | Train File Size (bytes) | Test File Size (bytes) |
  | :--- | :--- | :--- |
  | `MinMaxScaler` | 10,808,630 | 12,909,583 |
  | `StandardScaler` | 24,646,846 | 24,510,662 |
  | `QuantileTransformer` | 10,692,714 | 11,631,508 |
  | `RobustScaler` | 10,022,042 | 12,879,027 |
  | `Normalizer` | 15,825,760 | 16,888,777 |

### 4.3 Dataset 3: CICIoT2023 (High-Throughput IoT Flood Attacks)
- **Domain**: Modern high-throughput IoT attacks including 33 attack types across 7 classes (DDoS/DoS Flood, Reconnaissance, Web-based, Brute-Force, Spoofing, Mirai).
- **Dimensionality**: **46 numerical flow features** (Header lengths, protocol types, inter-arrival times, flag counters, packet size variance).
- **Split Configuration**:
  - `Train`: **30,000** samples (100% Normal)
  - `Test`: **30,000** samples (15,000 Normal + 15,000 Anomaly attacks)
- **Exact Remote File Footprint**:
  | Scaler | Train File Size (bytes) | Test File Size (bytes) |
  | :--- | :--- | :--- |
  | `MinMaxScaler` | 14,440,217 | 13,835,233 |
  | `StandardScaler` | 22,669,207 | 22,697,396 |
  | `QuantileTransformer` | 14,607,293 | 13,913,421 |
  | `RobustScaler` | 14,374,932 | 14,983,948 |
  | `Normalizer` | 17,720,145 | 16,703,186 |

### 4.4 Dataset 4: N_BaIoT (High-Dimensional 115-Feature Commercial IoT Hardware Botnet)
- **Domain**: Real network traffic from 9 infected commercial IoT devices (Danmini Doorbell, Ecobee Thermostat, Ennio Doorbell, Philips Baby Monitor, Provision Security Cameras, Samsung Webcam, SimpleHome Cameras) attacked by Mirai and BASHLITE.
- **Dimensionality**: **115 high-dimensional statistical features** capturing stream statistics (mean, variance, weight, magnitude, radius, covariance, PCC) across 5 decaying time windows ($100\text{ms}, 500\text{ms}, 1.5\text{s}, 10\text{s}, 1\text{min}$).
- **Split Configuration**:
  - `Train`: **30,000** samples (100% Normal)
  - `Test`: **30,000** samples (15,000 Normal + 15,000 Anomaly attacks)
- **Exact Remote File Footprint**:
  | Scaler | Train File Size (bytes) | Test File Size (bytes) |
  | :--- | :--- | :--- |
  | `MinMaxScaler` | 59,987,760 | 53,408,092 |
  | `StandardScaler` | 67,324,283 | 67,176,931 |
  | `QuantileTransformer` | 60,054,864 | 51,550,990 |
  | `RobustScaler` | 63,023,378 | 59,822,158 |
  | `Normalizer` | 62,957,273 | 56,970,537 |

---

## 5. Remote Repository & Git Structure

### 5.1 Repository Path
- **Remote Absolute Path**: `/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst`
- **Remote Git Remote**: `origin -> https://github.com/MinhMarks/UIT_Research_NFST.git`

### 5.2 Branch State
- **Active Branch**: `feature/federated-loc-nfst`
- **Status**: Up to date with `origin/feature/federated-loc-nfst`.
- **Target Branch**: `feature/federated-lunar-novel` (to be created as requested in `ORIGINAL_REQUEST.md`).
- **Untracked Directories**:
  - `DataProcessing/edgeiiotset-cyber-security-dataset-of-iot-iiot/`
  - `notebooks/experiments/outputs/adyn_results/`
  - `scratch/`

### 5.3 Key Remote Directories
- `Datascaled/Official_OC_Data/`: Location of all pre-scaled training and testing CSVs.
- `baseline_model/`: Contains existing baseline implementations (LOF, EIF, OCSVM, DeepSVDD, LOC-NFST).
- `outputs/`: Output folder for benchmark runs and results. The runner will write to `outputs/lunar_results/`.

---

## 6. Recommendations & Guidelines for Implementation Agents

1. **Non-IID Partition Strategy**: For $M \ge 3$ client nodes under the One-Class protocol, partition the 30,000 normal training samples using Dirichlet distribution ($\alpha \in \{0.1, 0.5\}$) or feature-clustering sub-manifolds to reproduce uncoordinated pseudo-negative generation and gradient conflict dynamics.
2. **GPU Optimization**: Utilize the RTX 5090 GPU (`cuda:0`). PyOD's LUNAR natively uses PyTorch tensors; batched k-NN computations can be accelerated using PyTorch or FAISS.
3. **Execution Scripting**: Always invoke python via `/opt/tljh/user/bin/python3` and run headless SSH jobs with `ssh -n postmaster.iec "cd ... && nohup ... &"` or structured runner scripts redirecting output to log files.
4. **Scaler Selection**: `QuantileTransformer` and `StandardScaler` are the standard scalers with the strongest empirical baseline performance in prior LOC-NFST evaluations.
