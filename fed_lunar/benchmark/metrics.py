"""
Standardized Benchmark Metrics & Instrumentation Engine for Federated LUNAR.

Implements rigorous evaluation metrics:
- Detection performance: AUC-ROC (%), Optimal F1 (PR-curve sweep), Calibrated F1 (95th normal percentile),
  Precision, Recall / Detection Rate (DR), False Alarm Rate (FAR).
- Confusion matrix decomposition (TP, FP, TN, FN).
- Multi-round optimization dynamics: Gradient Conflict Ratio (GCR %), Round Conflict Ratio (%),
  Mean Cosine Similarity, Convergence Round Count.
- Edge viability instrumentation: Per-sample streaming and batched latency (ms), MemoryTracker (Host RAM + GPU VRAM MB).
- Contract-compliant MetricsLogger for structured evaluation and CSV logging.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import csv
import time
import tracemalloc
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)


def calculate_optimal_f1_threshold(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
) -> Tuple[float, float]:
    """
    Find optimal decision threshold maximizing binary F1-score via Precision-Recall curve sweep.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = anomaly).
        y_scores: Continuous anomaly probability or residual scores.

    Returns:
        Tuple of (best_f1_percent, optimal_threshold).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=1.0, neginf=0.0)

    if len(np.unique(y_true)) < 2:
        median_val = float(np.median(y_scores)) if len(y_scores) > 0 else 0.5
        return 0.0, median_val

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

    if len(thresholds) > 0:
        f1_candidates = f1_scores[:-1]
        best_idx = int(np.argmax(f1_candidates))
        best_f1 = float(f1_candidates[best_idx]) * 100.0
        best_thresh = float(thresholds[best_idx])
    else:
        best_f1 = 0.0
        best_thresh = float(np.median(y_scores)) if len(y_scores) > 0 else 0.5

    return round(best_f1, 2), round(best_thresh, 4)


def calculate_detection_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, Any]:
    """
    Calculate detection performance metrics.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = anomaly).
        y_scores: Continuous anomaly probability or residual scores.
        threshold: Decision threshold for binary classification. If None, dynamically
                   computed from the normal score envelope at threshold_percentile.
        threshold_percentile: Percentile of normal scores used if threshold is None (default: 95.0).

    Returns:
        Dictionary of formatted metrics:
            - 'auc_roc': Area Under the ROC Curve in [0, 100] %.
            - 'f1_score': Primary benchmark F1 score in [0, 100] % (= f1_optimal).
            - 'f1_optimal': Optimal threshold F1 score in [0, 100] %.
            - 'optimal_threshold': Evaluated optimal threshold.
            - 'f1_calibrated': Operational F1 score at calibrated threshold in [0, 100] %.
            - 'calibrated_threshold': Calibrated decision threshold.
            - 'f1_macro': Macro-averaged F1 score in [0, 100] %.
            - 'precision': Precision for anomaly class in [0, 100] %.
            - 'detection_rate': Recall / True Positive Rate in [0, 100] %.
            - 'far': False Alarm Rate / False Positive Rate in [0, 100] %.
            - 'threshold': Evaluated decision threshold (same as calibrated_threshold).
            - 'tp', 'fp', 'tn', 'fn': Raw confusion matrix counts.
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
        if f1_opt >= 99.99:
            eval_thresh = opt_thresh
        elif len(normal_scores) > 0:
            eval_thresh = float(np.percentile(normal_scores, threshold_percentile))
        else:
            eval_thresh = float(np.median(y_scores)) if len(y_scores) > 0 else 0.5
    else:
        eval_thresh = float(threshold)

    y_pred = (y_scores >= eval_thresh).astype(int)

    # 4. Confusion Matrix at calibrated threshold
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 5. Rates and scores
    far = (fp / max(1, fp + tn)) * 100.0
    dr = (tp / max(1, tp + fn)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1_cal = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) * 100.0
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0

    return {
        "auc_roc": round(auc, 2),
        "f1_score": f1_opt,
        "f1_optimal": f1_opt,
        "optimal_threshold": opt_thresh,
        "f1_calibrated": round(f1_cal, 2),
        "calibrated_threshold": round(eval_thresh, 4),
        "f1_macro": round(f1_mac, 2),
        "f1_binary": round(f1_cal, 2),
        "precision": round(prec, 2),
        "detection_rate": round(dr, 2),
        "far": round(far, 2),
        "threshold": round(eval_thresh, 4),
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
    Calculates gradient conflict ratio (% rounds with conflicts), average pairwise GCR,
    mean cosine similarity, and convergence round count from model training history.

    Args:
        history: List of per-round metric dictionaries from model.history.
        total_rounds: Total planned communication rounds.
        loss_tolerance: Relative loss threshold for convergence detection.
        patience: Number of consecutive rounds with loss diff < tolerance.

    Returns:
        Dict with keys:
            - 'round_conflict_ratio': % rounds with at least one conflict.
            - 'gradient_conflict_ratio': Average pairwise GCR in % across rounds.
            - 'mean_cosine': Mean pairwise cosine similarity across rounds.
            - 'convergence_rounds': Communication round index of convergence.
    """
    if not history:
        return {
            "round_conflict_ratio": 0.0,
            "gradient_conflict_ratio": 0.0,
            "mean_cosine": 1.0,
            "convergence_rounds": total_rounds if total_rounds > 0 else 1,
        }

    gcrs: List[float] = []
    cosines: List[float] = []
    rounds_with_conflicts = 0

    for h in history:
        gcr_val = h.get("pre_gcr", h.get("gcr", None))
        cos_val = h.get("pre_mean_cosine", h.get("mean_cosine", None))
        min_cos = h.get("pre_min_cosine", h.get("min_cosine", None))

        if gcr_val is not None:
            val_pct = gcr_val * 100.0 if gcr_val <= 1.0 else gcr_val
            gcrs.append(val_pct)
        if cos_val is not None:
            cosines.append(cos_val)

        # A round has conflict if pre_gcr > 0 or min_cosine < 0
        if (gcr_val is not None and gcr_val > 0.0) or (min_cos is not None and min_cos < 0.0):
            rounds_with_conflicts += 1

    R_history = len(history)
    round_conflict_ratio = (rounds_with_conflicts / max(1, R_history)) * 100.0
    mean_gcr = float(np.mean(gcrs)) if gcrs else 0.0
    mean_cos = float(np.mean(cosines)) if cosines else 1.0

    # Convergence round count: earliest round where relative change is below loss_tolerance for patience rounds
    losses = [h.get("mean_loss", h.get("mean_mse_loss", None)) for h in history]
    losses = [float(l) for l in losses if l is not None]

    conv_round = total_rounds if total_rounds > 0 else 1
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
        """Returns maximum memory used across host RAM and GPU VRAM in MB."""
        return round(max(self.peak_cpu_mb, self.peak_gpu_mb), 2)


class LatencyResult(dict):
    """Dictionary holding latency metrics with float casting returning ms per sample."""

    def __float__(self) -> float:
        return float(self.get("latency_ms_per_sample", 0.0))


def measure_inference_latency(
    model: Any,
    X_test: np.ndarray,
    n_runs: int = 3,
    batch_size: int = 1024,
    measure_single: bool = True,
) -> LatencyResult:
    """
    Measure inference latency in milliseconds per sample (both batched and single-sample).

    Args:
        model: Fitted model instance with decision_function or score method.
        X_test: Test features matrix.
        n_runs: Repetitions for stable timing.
        batch_size: Mini-batch size.
        measure_single: Whether to profile single-sample streaming latency (B=1).

    Returns:
        LatencyResult dictionary with keys 'latency_ms_per_sample' and 'latency_single_ms'.
    """
    N = len(X_test)
    if N == 0:
        return LatencyResult({"latency_ms_per_sample": 0.0, "latency_single_ms": 0.0})

    # 1. Warm-up
    warmup_subset = X_test[: min(50, N)]
    if hasattr(model, "decision_function"):
        _ = model.decision_function(warmup_subset)
    elif hasattr(model, "score"):
        _ = model.score(warmup_subset)

    # 2. Batched throughput latency
    batched_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if hasattr(model, "decision_function"):
            _ = model.decision_function(X_test, batch_size=batch_size)
        elif hasattr(model, "score"):
            _ = model.score(X_test)
        elapsed = time.perf_counter() - t0
        batched_times.append(elapsed)

    mean_time_sec = float(np.mean(batched_times))
    ms_per_sample = (mean_time_sec / N) * 1000.0

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
        single_ms = float(np.median(single_times)) if single_times else ms_per_sample

    return LatencyResult({
        "latency_ms_per_sample": round(ms_per_sample, 4),
        "latency_single_ms": round(single_ms, 4),
    })


class MetricsLogger:
    """Automated Metrics Computation & CSV Logging for test contract compliance."""

    @staticmethod
    def evaluate(
        y_true: Union[np.ndarray, List[int]],
        y_score: Union[np.ndarray, List[float]],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Computes AUC-ROC (%), F1-Score, and False Alarm Rate (FAR as a ratio in [0, 1])."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_score = np.asarray(y_score, dtype=np.float64).ravel()

        if len(np.unique(y_true)) < 2:
            auc = 50.0
        else:
            auc = float(roc_auc_score(y_true, y_score) * 100.0)

        y_pred = (y_score >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        far = float(fp / max(1, fp + tn))

        return {
            "auc_roc": round(auc, 2),
            "f1_score": round(f1, 4),
            "far": round(far, 4),
        }

    @staticmethod
    def log_results_to_csv(filepath: str, row_dict: Dict[str, Any]) -> None:
        """Logs structured benchmark metrics row to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        file_exists = os.path.isfile(filepath)

        fieldnames = [
            "dataset", "method", "clients", "alpha", "rounds",
            "auc_roc", "f1_score", "far", "gradient_conflict_ratio",
            "convergence_rounds", "latency_ms_per_sample", "peak_memory_mb"
        ]

        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row_dict.get(k, "") for k in fieldnames})
