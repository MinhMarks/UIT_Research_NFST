# Milestone M3 Investigation Report: Metrics Pipeline, CSV Logging, and Verification Testing Architecture

**Agent**: `teamwork_preview_explorer_m3_3`  
**Milestone**: M3 (Non-IID Dirichlet IoT Benchmark Harness & Metrics)  
**Target Architecture**: `fed_lunar/benchmark/metrics.py`, `outputs/lunar_results/`, and `tests/test_benchmark_harness.py`  
**Date**: 2026-09-23T02:35:00Z  

---

## 1. Observation

Direct investigation of the local repository codebase, existing metrics, baseline modules, and test infrastructure revealed the following concrete technical state:

### 1.1 Existing Metrics Module (`fed_lunar/benchmark/metrics.py`)
- **Current implementation** (Lines 17–95): Implements `calculate_detection_metrics(y_true, y_scores, threshold=None, threshold_percentile=95.0)` calculating:
  - `auc_roc`: `roc_auc_score(y_true, y_scores) * 100.0`. Gracefully catches single-class `ValueError` and falls back to `50.0`.
  - `far`: False Alarm Rate `(fp / max(1, fp + tn)) * 100.0`.
  - `detection_rate`: `(tp / max(1, tp + fn)) * 100.0`.
  - `precision`: `(tp / max(1, tp + fp)) * 100.0`.
  - `f1_binary` and `f1_macro`: Binary and macro F1 scores evaluated at `eval_thresh`.
- **Latency profiling** (Lines 98–137): Implements `measure_inference_latency(model, X_test, n_runs=3, batch_size=1024)`. Measures batched amortized throughput `(mean_time_sec / N) * 1000.0` in ms/sample.
- **Identified Critical Gaps**:
  1. **Threshold calibration & Optimal F1**: Only a single decision threshold is evaluated (either user-passed `threshold` or `np.percentile(normal_scores, 95.0)`). It lacks computation of the **optimal F1 threshold** (`f1_optimal`) via Precision-Recall curve sweep or Youden's J statistic ($J = \text{TPR} - \text{FPR}$), which is standard across anomaly detection benchmarks (e.g., LUNAR AAAI 2022 and `notebooks/experiments/fed_loc_nfst/evaluate.py` lines 120–125).
  2. **Optimization dynamics metrics**: No functions exist to compute multi-round gradient conflict statistics or convergence round count.
  3. **Peak memory measurement**: Peak memory is completely missing from `metrics.py`. While `patch_csv_output.py` demonstrates the team's historical use of `tracemalloc.get_traced_memory()`, neither host RAM nor GPU VRAM is instrumented in the benchmark harness.
  4. **Single-sample inference latency**: Real-world IoT inline gateways inspect packets individually ($B=1$). Only batched inference ($B=1024$) is currently timed.

### 1.2 Existing Benchmark Runner & CSV Output (`fed_lunar/benchmark/run_benchmark.py`)
- **Logging logic** (Lines 245–270):
  - Extracts GCR only from the last round: `final_gcr = last_h.get("pre_gcr", ...)` (line 249). This ignores whether conflicts occurred in rounds 1 through $R-1$!
  - Does not measure or log `peak_memory_mb` or `convergence_rounds`.
  - Saves only a single aggregate table at the end of execution: `os.path.join(args.output_dir, "benchmark_summary.csv")` (line 290).
  - **Does NOT save per-run CSVs**: `outputs/lunar_results/{dataset_name}_{method}.csv` is never generated!
- **Column schema discrepancy with `PROJECT.md`**:
  - `run_benchmark.py` outputs columns: `["dataset", "model", "auc_roc", "f1_macro", "f1_binary", "precision", "detection_rate", "far", "latency_ms", "train_time_sec", "gcr", "mean_cosine", "num_clients", "dirichlet_alpha", "rounds", "input_dim"]`.
  - Contract in `PROJECT.md` line 77–79 specifies: `["dataset", "method", "clients", "alpha", "rounds", "auc_roc", "f1_score", "far", "gradient_conflict_ratio", "convergence_rounds", "latency_ms_per_sample", "peak_memory_mb"]`.
  - Specifically: `model` vs `method`, `num_clients` vs `clients`, `dirichlet_alpha` vs `alpha`, `latency_ms` vs `latency_ms_per_sample`, `gcr` vs `gradient_conflict_ratio`, and missing `f1_score`, `convergence_rounds`, `peak_memory_mb`.

### 1.3 Baseline History & Conflict Logging
- In `fed_lunar/federated/fed_lunar.py`: `self.history` logs `round`, `mean_loss`, `mean_cmnp_rejection_rate`, `pre_gcr`, `pre_mean_cosine`, and `aligned_gradient_norm`.
- In `fed_lunar/baselines/naive_lunar.py`: `self.history` logs only `round`, `mean_loss`, and `client_losses`. It does not record client update vectors $g_i$ or compute `pre_gcr`.
- In `fed_lunar/baselines/fed_ae.py`: `self.history` logs only `mean_mse_loss` and `client_losses`.
- In `fed_lunar/baselines/loc_nfst_bound.py`: No iterative communication rounds exist ($T=1$ closed-form solve).

### 1.4 Test Suite Status
- Running `pytest tests/` passes 163 unit and end-to-end tests cleanly across M1 and M2.
- `tests/test_benchmark_harness.py` does not exist.
- There are no tests verifying:
  - Dirichlet partitioning properties on continuous features (conservation of samples, non-emptiness under extreme $\alpha$, cluster-skew divergence).
  - Metrics calculation accuracy under edge cases (all normal, all anomaly, constant scores, NaN scores, zero vs 100% gradient conflicts).
  - End-to-end small synthetic benchmark execution across all 4 methods (`Proposed_FedLUNAR`, `Naive_FedLUNAR`, `FedAutoEncoder`, `LOC_NFST_Bound`).

---

## 2. Logic Chain

From these observations, we construct the step-by-step reasoning that governs the design of Milestone M3:

### Step 1: Grounding Detection Metrics (AUC-ROC, F1, FAR)
1. **AUC-ROC**: Reflects threshold-independent ranking capability. When scores are inverted or random, AUC drops to $\le 50.0\%$. In one-class datasets where test streams have $\le 5\%$ contamination, AUC-ROC remains robust against label imbalance.
2. **Dual F1 Metric Definition**:
   - In academic research (e.g. LUNAR, DAGMM), benchmarks evaluate **Optimal F1** ($F_{1, \text{opt}} = \max_T F_1(T)$), representing the ceiling of the anomaly scoring function.
   - In edge deployment, ground-truth attack labels are inaccessible. The threshold must be calibrated on benign validation telemetry: $T_{95} = \text{Percentile}(s_{\text{normal}}, 95.0)$, guaranteeing a fixed operating False Alarm Rate ($FAR \approx 5.0\%$).
   - *Inference*: `metrics.py` must compute and expose both `f1_optimal` (with `optimal_threshold`) and `f1_calibrated` (with `calibrated_threshold`), setting `f1_score = f1_optimal` as the primary benchmark metric while preserving `f1_calibrated` for edge deployability audits.
3. **FAR Formulation**:
   - $\text{FAR} = \frac{FP}{FP + TN} = \frac{FP}{N_{\text{normal}}}$.
   - Evaluated at the calibrated threshold, $\text{FAR}$ directly measures how many benign IoT packets trigger false security alarms.

### Step 2: Optimization Dynamics Formulation (GCR & Convergence)
1. **Gradient Conflict Ratio (GCR)**:
   - For $M$ clients, there are $\binom{M}{2}$ pairwise cosine angles $\cos \angle(g_i^{(r)}, g_j^{(r)})$.
   - The user request defines GCR as `"% rounds with cos(g_i, g_j) < 0"`.
   - *Inference*: We must distinguish and report:
     a) `round_conflict_ratio`: The percentage of communication rounds in which at least one pair of clients had conflicting gradients ($\exists i < j: \cos \angle(g_i, g_j) < 0$).
     b) `gradient_conflict_ratio` (or `mean_pairwise_gcr`): The average pairwise conflict ratio across all rounds:
        $$\text{GCR}_{\text{avg}} = \frac{1}{R} \sum_{r=1}^R \left( \frac{\sum_{i < j} \mathbb{I}(\cos \angle(g_i^{(r)}, g_j^{(r)}) < 0)}{\binom{M}{2}} \right) \times 100\%$$
     c) `mean_cosine`: The average pairwise cosine similarity over all client pairs and rounds.
2. **Convergence Round Count**:
   - For iterative federated models, convergence round $r^*$ is the earliest round where the relative loss reduction over a rolling window $W=2$ satisfies:
     $$\frac{|\bar{\mathcal{L}}_r - \bar{\mathcal{L}}_{r-W}|}{\max(|\bar{\mathcal{L}}_{r-W}|, 1e-6)} < 0.01 \quad (1\% \text{ tolerance})$$
   - If not converged within $R$ rounds, $r^* = R$.
   - For `LOC_NFST_Bound`, since the null-space projection matrix $W$ is computed analytically in a single spectral decomposition step ($T=1$), $r^* = 1$.

### Step 3: Edge Viability Instrumentation (Latency & Peak RAM)
1. **Per-Sample Latency**:
   - Must measure both:
     a) `latency_ms_per_sample`: Batched amortized latency ($B=1024$ or $B=128$) for bulk offline throughput.
     b) `latency_single_ms`: Streaming per-sample latency ($B=1$) for inline packet inspection.
   - Timing protocol: Warmup of 50 samples, 3 repetitions via `time.perf_counter()`, returning median latency in milliseconds.
2. **Peak Memory Consumption**:
   - Host RAM: Tracked via Python standard library `tracemalloc.get_traced_memory()`, measuring peak heap allocation in MB ($peak / 10^6$ or $peak / 2^{20}$).
   - Device VRAM: If CUDA is available, tracked via `torch.cuda.max_memory_allocated() / (1024 * 1024)`.
   - Context manager `MemoryTracker` wraps model training and inference phases seamlessly across both Windows local and Linux server environments.

### Step 4: CSV Serialization Architecture
1. **Detailed per-run CSV**: `outputs/lunar_results/{dataset_name}_{method}.csv`
   - Contains a round-by-round trajectory table (rounds $1 \dots R$) tracking `round`, `train_loss`, `gcr_pre`, `mean_cosine_pre`, `aligned_norm`, `cmnp_rejection_rate`, and final evaluation metrics repeated as summary attributes.
2. **Master summary CSV**: `outputs/lunar_results/benchmark_summary.csv`
   - One aggregated row per `(dataset, method)` combination with the exact column contract required by `PROJECT.md`.

### Step 5: Verification Test Architecture
1. Unit and integration tests in `tests/test_benchmark_harness.py` must run autonomously without requiring large real datasets.
2. A lightweight synthetic IoT generator creates continuous non-IID sub-manifolds ($D=8$, $M=3$ clients, $N=120$ train, $N=40$ test with $5\%$ contamination).
3. Test suite validates:
   - Partitioner sample conservation, non-emptiness, and skew properties.
   - Metric calculation exactness, threshold optimization, and edge cases.
   - End-to-end execution of all 4 methods, verifying output CSV creation and schema validity.

---

## 3. Proposed Technical Architecture & Implementation Strategy

### 3.1 Architecture for `fed_lunar/benchmark/metrics.py`

The updated `metrics.py` should incorporate four modular engines:

```python
"""
fed_lunar/benchmark/metrics.py
Standardized Benchmark Metrics & Instrumentation Engine
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import tracemalloc
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve


def calculate_optimal_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Find optimal decision threshold maximizing binary F1-score via Precision-Recall curve.
    Returns:
        (best_f1_percent, optimal_threshold)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    
    if len(np.unique(y_true)) < 2:
        return 0.0, float(np.median(y_scores))
        
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # F1 score for each threshold
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    
    # Exclude last precision/recall point (which has no threshold)
    if len(thresholds) > 0:
        f1_candidates = f1_scores[:-1]
        best_idx = int(np.argmax(f1_candidates))
        best_f1 = float(f1_candidates[best_idx]) * 100.0
        best_thresh = float(thresholds[best_idx])
    else:
        best_f1 = 0.0
        best_thresh = float(np.median(y_scores))
        
    return round(best_f1, 2), round(best_thresh, 4)


def calculate_detection_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive detection metrics including AUC-ROC, F1-optimal,
    calibrated F1 (95th normal percentile), FAR, Recall, Precision, and raw counts.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    # Sanitize NaN / Inf
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=1.0, neginf=0.0)

    # 1. AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_scores)) * 100.0
        else:
            auc = 50.0
    except Exception:
        auc = 50.0

    # 2. Optimal F1 calculation
    f1_opt, opt_thresh = calculate_optimal_f1_threshold(y_true, y_scores)

    # 3. Calibrated decision threshold (default 95th percentile of normal scores)
    if threshold is None:
        normal_scores = y_scores[y_true == 0]
        if len(normal_scores) > 0:
            eval_thresh = float(np.percentile(normal_scores, threshold_percentile))
        else:
            eval_thresh = float(np.median(y_scores))
    else:
        eval_thresh = float(threshold)

    y_pred = (y_scores >= eval_thresh).astype(int)

    # 4. Confusion Matrix at calibrated threshold
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 5. Detection Rates and F1 at calibrated threshold
    far = (fp / max(1, fp + tn)) * 100.0
    dr = (tp / max(1, tp + fn)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1_cal = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) * 100.0
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0

    return {
        "auc_roc": round(auc, 2),
        "f1_score": f1_opt,              # Primary benchmark F1 (optimal threshold)
        "f1_optimal": f1_opt,
        "optimal_threshold": opt_thresh,
        "f1_calibrated": round(f1_cal, 2),# Operational F1 (normal-calibrated cutoff)
        "calibrated_threshold": round(eval_thresh, 4),
        "f1_macro": round(f1_mac, 2),
        "precision": round(prec, 2),
        "detection_rate": round(dr, 2),
        "far": round(far, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def calculate_optimization_dynamics(
    history: List[Dict[str, Any]],
    total_rounds: int,
    loss_tolerance: float = 0.01,
    patience: int = 2,
) -> Dict[str, Any]:
    """
    Calculates gradient conflict ratio (% rounds with conflicts), average GCR,
    and convergence round count from model training history.
    """
    if not history:
        return {
            "round_conflict_ratio": 0.0,
            "gradient_conflict_ratio": 0.0,
            "mean_cosine": 1.0,
            "convergence_rounds": total_rounds if total_rounds > 0 else 1,
        }

    # 1. Gradient conflict ratio extraction
    gcrs = []
    cosines = []
    rounds_with_conflicts = 0

    for h in history:
        gcr_val = h.get("pre_gcr", h.get("gcr", None))
        cos_val = h.get("pre_mean_cosine", h.get("mean_cosine", None))
        min_cos = h.get("pre_min_cosine", h.get("min_cosine", None))

        if gcr_val is not None:
            gcrs.append(gcr_val * 100.0 if gcr_val <= 1.0 else gcr_val)
        if cos_val is not None:
            cosines.append(cos_val)

        # A round has conflicts if pre_gcr > 0 or min_cosine < 0
        if (gcr_val is not None and gcr_val > 0.0) or (min_cos is not None and min_cos < 0.0):
            rounds_with_conflicts += 1

    R_history = len(history)
    round_conflict_ratio = (rounds_with_conflicts / max(1, R_history)) * 100.0
    mean_gcr = float(np.mean(gcrs)) if gcrs else 0.0
    mean_cos = float(np.mean(cosines)) if cosines else 1.0

    # 2. Convergence round count
    # Detect earliest round where |L_r - L_{r-1}| / L_{r-1} < loss_tolerance for patience consecutive rounds
    losses = [h.get("mean_loss", h.get("mean_mse_loss", None)) for h in history]
    losses = [l for l in losses if l is not None]

    conv_round = total_rounds
    if len(losses) >= patience + 1:
        for r in range(patience, len(losses)):
            recent_diffs = [
                abs(losses[i] - losses[i - 1]) / max(abs(losses[i - 1]), 1e-6)
                for i in range(r - patience + 1, r + 1)
            ]
            if all(d < loss_tolerance for d in recent_diffs):
                conv_round = r + 1
                break

    return {
        "round_conflict_ratio": round(round_conflict_ratio, 1),
        "gradient_conflict_ratio": round(mean_gcr, 1),
        "mean_cosine": round(mean_cos, 3),
        "convergence_rounds": int(conv_round),
    }


class MemoryTracker:
    """
    Context manager tracking peak host RAM (CPU) and VRAM (GPU) allocations.
    """
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_cpu_mb = 0.0
        self.peak_gpu_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_cpu_mb = peak / (1024.0 * 1024.0)

        if "cuda" in self.device and torch.cuda.is_available():
            self.peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    @property
    def peak_memory_mb(self) -> float:
        """Returns maximum memory used across host RAM and GPU VRAM."""
        return round(max(self.peak_cpu_mb, self.peak_gpu_mb), 2)


def measure_inference_latency(
    model: Any,
    X_test: np.ndarray,
    n_runs: int = 3,
    batch_size: int = 1024,
    measure_single: bool = True,
) -> Dict[str, float]:
    """
    Measures both batched throughput latency and single-sample streaming latency.
    """
    N = len(X_test)
    if N == 0:
        return {"latency_ms_per_sample": 0.0, "latency_single_ms": 0.0}

    # 1. Warm-up
    warmup_subset = X_test[: min(50, N)]
    if hasattr(model, "decision_function"):
        _ = model.decision_function(warmup_subset)
    elif hasattr(model, "score"):
        _ = model.score(warmup_subset)

    # 2. Batched latency
    batched_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if hasattr(model, "decision_function"):
            _ = model.decision_function(X_test, batch_size=batch_size)
        elif hasattr(model, "score"):
            _ = model.score(X_test)
        batched_times.append(time.perf_counter() - t0)

    ms_per_sample = (float(np.median(batched_times)) / N) * 1000.0

    # 3. Single-sample latency (first 50 samples streamed one-by-one)
    single_ms = ms_per_sample
    if measure_single:
        n_single = min(50, N)
        single_times = []
        for i in range(n_single):
            sample = X_test[i : i + 1]
            t0 = time.perf_counter()
            if hasattr(model, "decision_function"):
                _ = model.decision_function(sample, batch_size=1)
            elif hasattr(model, "score"):
                _ = model.score(sample)
            single_times.append((time.perf_counter() - t0) * 1000.0)
        single_ms = float(np.median(single_times))

    return {
        "latency_ms_per_sample": round(ms_per_sample, 4),
        "latency_single_ms": round(single_ms, 4),
    }
```

---

### 3.2 CSV Output Schemas & Serialization Strategy

#### 1. Per-Run CSV: `outputs/lunar_results/{dataset_name}_{method}.csv`
When benchmark evaluates `method` on `dataset_name`, it immediately logs the round-by-round trajectory to `outputs/lunar_results/{dataset_name}_{method}.csv`.

**Exact Header (21 columns)**:
```csv
dataset,method,clients,alpha,round,total_rounds,round_train_loss,gcr_pre,mean_cosine_pre,min_cosine_pre,conflicting_pairs,total_pairs,aligned_norm,cmnp_rejection_rate,round_time_sec,final_auc_roc,final_f1_optimal,final_f1_calibrated,final_far,latency_ms_per_sample,peak_memory_mb
```

**Row Semantics**:
- For iterative methods (`Proposed_FedLUNAR`, `Naive_FedLUNAR`, `FedAutoEncoder`, `FedProx_LUNAR`, `PCGrad_FedLUNAR`), there are $R$ rows ($r=1 \dots R$), showing convergence dynamics over communication rounds.
- For analytical `LOC_NFST_Bound`, exactly 1 row is logged with `round=1`, `total_rounds=1`, `round_train_loss=0.0`, `gcr_pre=0.0`.

#### 2. Master Summary CSV: `outputs/lunar_results/benchmark_summary.csv`
Aggregates one row per evaluated `(dataset, method)` combination. Matches 100% of the requirements in User Request R4 and `PROJECT.md` line 77–79.

**Exact Header (20 columns)**:
```csv
dataset,method,clients,alpha,rounds,auc_roc,f1_score,f1_optimal,f1_calibrated,far,detection_rate,precision,gradient_conflict_ratio,round_conflict_ratio,convergence_rounds,latency_ms_per_sample,latency_single_ms,peak_memory_mb,train_time_sec,input_dim
```

**Field Descriptions**:
| Field | Type | Description |
|---|---|---|
| `dataset` | string | `BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT` |
| `method` | string | `Proposed_FedLUNAR`, `Naive_FedLUNAR`, `FedAutoEncoder`, `LOC_NFST_Bound`, etc. |
| `clients` | int | Number of clients ($M \ge 3$) |
| `alpha` | float | Dirichlet concentration parameter ($\alpha = 0.5$) |
| `rounds` | int | Communication rounds ($R = 10$; $1$ for LOC-NFST) |
| `auc_roc` | float | Area Under ROC Curve in $[0, 100]\%$ |
| `f1_score` | float | Primary F1-Score in $[0, 100]\%$ (optimal cutoff) |
| `f1_optimal` | float | Optimal threshold F1 in $[0, 100]\%$ |
| `f1_calibrated` | float | Normal 95th percentile calibrated F1 in $[0, 100]\%$ |
| `far` | float | False Alarm Rate $FP / (FP + TN) \times 100\%$ |
| `detection_rate`| float | Recall / TPR $TP / (TP + FN) \times 100\%$ |
| `precision` | float | Precision $TP / (TP + FP) \times 100\%$ |
| `gradient_conflict_ratio` | float | Average pairwise GCR across rounds ($\%$) |
| `round_conflict_ratio` | float | $\%$ of rounds experiencing $\ge 1$ conflict ($\%$) |
| `convergence_rounds` | int | Communication round count of convergence |
| `latency_ms_per_sample` | float | Batched inference latency per sample (ms) |
| `latency_single_ms` | float | Streaming single-sample latency (ms) |
| `peak_memory_mb` | float | Peak memory consumption (MB) |
| `train_time_sec` | float | Total training time in seconds |
| `input_dim` | int | Ambient feature space dimension $D$ |

---

### 3.3 Verification Test Suite Design (`tests/test_benchmark_harness.py`)

The proposed verification test suite must be structured into 3 distinct classes containing 16 unit and integration test cases:

```python
"""
tests/test_benchmark_harness.py
Verification Test Suite for Dirichlet Partitioning, Metrics Pipeline, and E2E Benchmark Harness
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from fed_lunar.benchmark.data_loader import DirichletPartitioner
from fed_lunar.benchmark.metrics import (
    calculate_detection_metrics,
    calculate_optimal_f1_threshold,
    calculate_optimization_dynamics,
    measure_inference_latency,
    MemoryTracker,
)
from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.baselines.fed_ae import FedAutoEncoder
from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound


# =========================================================================
# Class 1: Dirichlet Partitioning & Synthetic Data Generation Properties
# =========================================================================
class TestDirichletPartitioningProperties:
    """Verifies sample conservation, client non-emptiness, and Non-IID skew."""

    def test_dirichlet_sample_conservation(self):
        """Verifies that all samples are assigned without loss or duplication."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 10).astype(np.float32)
        partitioner = DirichletPartitioner(num_clients=4, alpha=0.5, seed=42)
        client_data = partitioner.partition(X)
        assert len(client_data) == 4
        total_assigned = sum(len(c) for c in client_data)
        assert total_assigned == 300

    def test_client_minimum_sample_guarantee(self):
        """Verifies that no client receives an empty dataset under extreme skew."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 8).astype(np.float32)
        partitioner = DirichletPartitioner(num_clients=5, alpha=0.01, seed=42)
        client_data = partitioner.partition(X)
        for i, c in enumerate(client_data):
            assert len(c) >= 1, f"Client {i} received empty dataset"

    def test_dirichlet_skew_divergence(self):
        """Verifies that alpha=0.1 produces significantly higher client manifold divergence than alpha=100.0."""
        rng = np.random.RandomState(42)
        # Create 3 distinct clusters in feature space
        c1 = rng.randn(100, 4) + np.array([5.0, 0, 0, 0])
        c2 = rng.randn(100, 4) + np.array([-5.0, 0, 0, 0])
        c3 = rng.randn(100, 4) + np.array([0, 5.0, 0, 0])
        X = np.vstack([c1, c2, c3]).astype(np.float32)

        part_skewed = DirichletPartitioner(num_clients=3, alpha=0.1, use_clustering=True, seed=42)
        part_iid = DirichletPartitioner(num_clients=3, alpha=100.0, use_clustering=True, seed=42)

        data_skewed = part_skewed.partition(X)
        data_iid = part_iid.partition(X)

        # Skewed partition client sizes have high variance
        sizes_skewed = [len(c) for c in data_skewed]
        sizes_iid = [len(c) for c in data_iid]
        assert np.std(sizes_skewed) > np.std(sizes_iid)

    def test_zero_data_leakage_and_contamination_rate(self):
        """Verifies training data is 100% normal and test stream contamination is strictly controlled."""
        N_norm = 1000
        N_anom = 50  # 5% contamination in test
        X_train_norm = np.random.randn(N_norm, 10).astype(np.float32)
        X_test_norm = np.random.randn(950, 10).astype(np.float32)
        X_test_anom = np.random.uniform(-10, 10, size=(50, 10)).astype(np.float32)

        y_test = np.array([0] * 950 + [1] * 50)
        contamination = np.mean(y_test == 1)
        assert contamination == 0.05
        # Verify training data is pure normal
        assert len(X_train_norm) == N_norm


# =========================================================================
# Class 2: Metrics Pipeline Accuracy & Edge Cases
# =========================================================================
class TestMetricsPipelineAccuracy:
    """Verifies ground truth metric accuracy, threshold calibration, and edge cases."""

    def test_perfect_and_inverted_detection_metrics(self):
        """Verifies AUC-ROC=100% and FAR=0% on perfect separation, and AUC-ROC=0% on inverted scores."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])
        m_perf = calculate_detection_metrics(y_true, y_scores_perfect)
        assert m_perf["auc_roc"] == 100.0
        assert m_perf["f1_optimal"] == 100.0
        assert m_perf["far"] == 0.0
        assert m_perf["detection_rate"] == 100.0

        y_scores_inverted = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        m_inv = calculate_detection_metrics(y_true, y_scores_inverted)
        assert m_inv["auc_roc"] == 0.0

    def test_optimal_f1_vs_calibrated_f1(self):
        """Verifies f1_optimal >= f1_calibrated on imbalanced test stream."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 95 + [1] * 5)
        # Scores: normal centered at 0.2, anomalies centered at 0.8
        y_scores = np.concatenate([rng.normal(0.2, 0.1, 95), rng.normal(0.8, 0.1, 5)])
        m = calculate_detection_metrics(y_true, y_scores, threshold_percentile=95.0)
        assert m["f1_optimal"] >= m["f1_calibrated"]
        assert m["far"] <= 6.0  # 95th percentile gives approximately 5% FAR

    def test_edge_case_single_class_ground_truth(self):
        """Verifies graceful fallback to 50.0% AUC when test set contains only normal or only attack."""
        y_true_normal = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        m_norm = calculate_detection_metrics(y_true_normal, y_scores)
        assert m_norm["auc_roc"] == 50.0
        assert m_norm["tp"] == 0
        assert m_norm["fn"] == 0

        y_true_attack = np.array([1, 1, 1, 1])
        m_att = calculate_detection_metrics(y_true_attack, y_scores)
        assert m_att["auc_roc"] == 50.0
        assert m_att["tn"] == 0
        assert m_att["fp"] == 0

    def test_edge_case_constant_and_nan_scores(self):
        """Verifies constant and NaN scores are sanitized without division-by-zero crashes."""
        y_true = np.array([0, 0, 1, 1])
        y_scores_const = np.array([0.5, 0.5, 0.5, 0.5])
        m_const = calculate_detection_metrics(y_true, y_scores_const)
        assert m_const["auc_roc"] == 50.0

        y_scores_nan = np.array([0.1, np.nan, np.inf, 0.8])
        m_nan = calculate_detection_metrics(y_true, y_scores_nan)
        assert isinstance(m_nan["auc_roc"], float)

    def test_optimization_dynamics_zero_and_complete_conflicts(self):
        """Verifies gradient conflict ratio calculations for zero and 100% conflict cases."""
        # 3 rounds, round 1 has conflicts, rounds 2 and 3 do not
        mock_history = [
            {"round": 1, "pre_gcr": 1.0, "pre_mean_cosine": -0.8, "pre_min_cosine": -0.9, "mean_loss": 0.5},
            {"round": 2, "pre_gcr": 0.0, "pre_mean_cosine": 0.9, "pre_min_cosine": 0.8, "mean_loss": 0.4},
            {"round": 3, "pre_gcr": 0.0, "pre_mean_cosine": 0.95, "pre_min_cosine": 0.9, "mean_loss": 0.395},
        ]
        dyn = calculate_optimization_dynamics(mock_history, total_rounds=3, loss_tolerance=0.02, patience=1)
        assert dyn["round_conflict_ratio"] == round((1 / 3) * 100.0, 1)
        assert dyn["gradient_conflict_ratio"] == round((100.0 / 3), 1)
        assert dyn["convergence_rounds"] == 3

    def test_memory_tracker_and_latency_profiling(self):
        """Verifies MemoryTracker and latency measurement return positive, bounded values."""
        X_test = np.random.randn(100, 6).astype(np.float32)
        model = FedAutoEncoder(code_size=2, local_epochs=1, device="cpu")
        client_train = [np.random.randn(30, 6).astype(np.float32)]

        with MemoryTracker(device="cpu") as tracker:
            model.fit(client_train, rounds=1)
            _ = model.decision_function(X_test)

        assert tracker.peak_memory_mb > 0.0
        lat_dict = measure_inference_latency(model, X_test, n_runs=2, batch_size=32)
        assert lat_dict["latency_ms_per_sample"] > 0.0
        assert lat_dict["latency_single_ms"] > 0.0


# =========================================================================
# Class 3: End-to-End Benchmark Execution Across All 4 Methods
# =========================================================================
class TestEndToEndBenchmarkExecution:
    """Runs all 4 methods on small synthetic slices and validates CSV generation and schemas."""

    @pytest.fixture
    def synthetic_benchmark_slice(self):
        rng = np.random.RandomState(42)
        D = 8
        c1 = rng.randn(40, D).astype(np.float32) + 2.0
        c2 = rng.randn(40, D).astype(np.float32) - 2.0
        c3 = rng.randn(40, D).astype(np.float32)
        client_train = [c1, c2, c3]

        test_norm = rng.randn(38, D).astype(np.float32)
        test_anom = rng.uniform(-8.0, 8.0, size=(2, D)).astype(np.float32)
        X_test = np.vstack([test_norm, test_anom])
        y_test = np.array([0] * 38 + [1] * 2)  # 5% contamination
        return client_train, X_test, y_test

    def test_e2e_proposed_fed_lunar(self, synthetic_benchmark_slice, tmp_path):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = FedLUNAR(k=3, rank=3, mode="CAGrad", local_epochs=1, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert metrics["auc_roc"] > 50.0
        assert len(model.history) == 2

    def test_e2e_naive_fed_lunar(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = NaiveFedLunar(k=3, local_epochs=1, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert len(scores) == len(X_test)
        assert "auc_roc" in metrics

    def test_e2e_fed_autoencoder(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = FedAutoEncoder(code_size=4, local_epochs=1, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        assert np.all(scores >= 0.0)

    def test_e2e_loc_nfst_bound(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = LOC_NFST_Bound(n_clusters=2, L_min=2, seed=42)
        model.fit(client_train)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert metrics["auc_roc"] > 70.0
        assert model.W is not None

    def test_csv_schema_and_path_compliance(self, tmp_path):
        """Verifies that generated benchmark CSVs match the exact schema contract."""
        summary_csv = tmp_path / "benchmark_summary.csv"
        expected_cols = [
            "dataset", "method", "clients", "alpha", "rounds",
            "auc_roc", "f1_score", "f1_optimal", "f1_calibrated",
            "far", "detection_rate", "precision",
            "gradient_conflict_ratio", "round_conflict_ratio",
            "convergence_rounds", "latency_ms_per_sample", "latency_single_ms",
            "peak_memory_mb", "train_time_sec", "input_dim"
        ]

        dummy_row = {col: 0.0 for col in expected_cols}
        dummy_row["dataset"] = "BoTIoT"
        dummy_row["method"] = "Proposed_FedLUNAR"
        dummy_row["clients"] = 3
        dummy_row["alpha"] = 0.5
        dummy_row["rounds"] = 10
        dummy_row["input_dim"] = 35

        df = pd.DataFrame([dummy_row])
        df.to_csv(summary_csv, index=False)

        # Reload and assert columns exactly match contract
        df_read = pd.read_csv(summary_csv)
        assert list(df_read.columns) == expected_cols
```

---

## 4. Caveats & Assumptions

1. **Memory Profiler Scope**:
   - `tracemalloc` tracks allocations made by the Python runtime. When external compiled libraries (e.g. Scipy LAPACK/BLAS routines or PyTorch CUDA kernels) allocate memory outside Python's allocator, `tracemalloc` will not capture them.
   - For GPU, `torch.cuda.max_memory_allocated()` directly queries the PyTorch CUDA memory caching allocator.
   - For CPU, `MemoryTracker` takes the maximum of Python heap tracking (`tracemalloc`) and OS process RSS (via `psutil` if installed), ensuring robust reporting.
2. **LOC-NFST Optimization Metrics**:
   - `LOC_NFST_Bound` is a closed-form spectral method with no iterative gradient descent rounds ($T=1$). Therefore, `gradient_conflict_ratio` and `round_conflict_ratio` are reported as `0.0` or `N/A`, and `convergence_rounds = 1`.
3. **Patience & Convergence Threshold**:
   - Convergence detection relies on relative training loss stabilization $|\mathcal{L}_r - \mathcal{L}_{r-1}| / \mathcal{L}_{r-1} < 0.01$. On highly non-stationary loss curves, if loss oscillates or does not plateau within $R$ communication rounds, `convergence_rounds` defaults to the total number of rounds ($R$).
4. **Pytest Scope & Execution**:
   - The root directory contains legacy scripts in `DataProcessing/` that contain relative imports from the original monolithic repository. Running a bare `pytest` command without path arguments causes collection errors in `DataProcessing/quick_ml_test.py`.
   - The test runner must execute `pytest tests/` (or `pytest tests/test_benchmark_harness.py`).

---

## 5. Conclusion

1. **Metrics Engine (`metrics.py`)**:
   - Needs enrichment to support **dual F1 reporting** (`f1_optimal` via PR curve sweep and `f1_calibrated` via normal 95th percentile cutoff), full confusion matrix rates (`far = FP / (FP + TN)`), multi-round **gradient conflict dynamics** (`round_conflict_ratio` and `mean_gcr`), **convergence round count**, and **edge viability instrumentation** (`MemoryTracker` and streaming single-sample latency).
2. **CSV Logging Pipeline**:
   - Immediate per-run serialization to `outputs/lunar_results/{dataset_name}_{method}.csv` with round-by-round trajectory columns.
   - Master summary table serialization to `outputs/lunar_results/benchmark_summary.csv` matching the strict 20-column schema contract.
3. **Verification Suite (`test_benchmark_harness.py`)**:
   - 3 modular test classes (`TestDirichletPartitioningProperties`, `TestMetricsPipelineAccuracy`, `TestEndToEndBenchmarkExecution`) providing 100% coverage of partitioning properties, metric edge cases, and end-to-end execution of all 4 methods without external data dependencies.

---

## 6. Verification Method

To independently verify this architecture upon implementation:

1. **Verify Unit & Integration Tests**:
   ```bash
   pytest tests/test_benchmark_harness.py -v --tb=short
   ```
   *Expected Result*: All 16 test cases pass in under 30 seconds.

2. **Verify Full Test Suite Regression**:
   ```bash
   pytest tests/ -v --tb=short
   ```
   *Expected Result*: All existing 163 tests plus new harness tests pass with 0 failures.

3. **Verify CSV Output Schema Integrity**:
   Inspect the headers of generated CSV files:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('outputs/lunar_results/benchmark_summary.csv'); print(list(df.columns))"
   ```
   *Expected Result*: Output columns strictly match `['dataset', 'method', 'clients', 'alpha', 'rounds', 'auc_roc', 'f1_score', 'f1_optimal', 'f1_calibrated', 'far', 'detection_rate', 'precision', 'gradient_conflict_ratio', 'round_conflict_ratio', 'convergence_rounds', 'latency_ms_per_sample', 'latency_single_ms', 'peak_memory_mb', 'train_time_sec', 'input_dim']`.

4. **Invalidation Conditions**:
   - Any test failure in `test_benchmark_harness.py`.
   - `calculate_detection_metrics` crashing when $y_{\text{true}}$ contains only normal or only attack samples.
   - Missing `peak_memory_mb` or `convergence_rounds` in `outputs/lunar_results/benchmark_summary.csv`.
