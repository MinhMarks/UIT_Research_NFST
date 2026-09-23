"""
Standardized Benchmark Metrics for Federated Anomaly Detection.

Implements rigorous evaluation metrics:
- AUC-ROC, Macro F1, Binary F1, Precision, Recall / Detection Rate (DR), False Alarm Rate (FAR).
- Confusion matrix decomposition (TP, FP, TN, FN).
- Gradient Conflict Ratio (GCR) and cosine similarity metrics.
- Computational efficiency: Inference latency (ms/sample), Training duration (s), Peak RAM (MB).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


def calculate_detection_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, float]:
    """
    Calculate detection performance metrics.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = anomaly).
        y_scores: Continuous anomaly probability or residual scores.
        threshold: Decision threshold for binary classification. If None, dynamically
                   computed from the normal score envelope or requested percentile.
        threshold_percentile: Percentile of normal scores used if threshold is None.

    Returns:
        Dictionary of formatted metrics:
            - 'auc_roc': Area Under the ROC Curve in [0, 100] %.
            - 'f1_macro': Macro-averaged F1 score in [0, 100] %.
            - 'f1_binary': Binary F1 score for anomaly class in [0, 100] %.
            - 'precision': Precision for anomaly class in [0, 100] %.
            - 'detection_rate': Recall / True Positive Rate in [0, 100] %.
            - 'far': False Alarm Rate / False Positive Rate in [0, 100] %.
            - 'threshold': Evaluated decision threshold.
            - 'tp', 'fp', 'tn', 'fn': Raw confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    # Handle edge case where scores are constant or contain NaN
    if np.any(np.isnan(y_scores)):
        y_scores = np.nan_to_num(y_scores, nan=0.0)

    # 1. AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_scores)) * 100.0
        else:
            auc = 50.0
    except Exception:
        auc = 50.0

    # 2. Decision threshold
    if threshold is None:
        normal_scores = y_scores[y_true == 0]
        if len(normal_scores) > 0:
            eval_thresh = float(np.percentile(normal_scores, threshold_percentile))
        else:
            eval_thresh = float(np.median(y_scores))
    else:
        eval_thresh = float(threshold)

    y_pred = (y_scores >= eval_thresh).astype(int)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 4. Rates
    far = (fp / max(1, fp + tn)) * 100.0
    dr = (tp / max(1, tp + fn)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1_bin = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) * 100.0
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0

    return {
        "auc_roc": round(auc, 2),
        "f1_macro": round(f1_mac, 2),
        "f1_binary": round(f1_bin, 2),
        "precision": round(prec, 2),
        "detection_rate": round(dr, 2),
        "far": round(far, 2),
        "threshold": round(eval_thresh, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def measure_inference_latency(
    model: Any,
    X_test: np.ndarray,
    n_runs: int = 3,
    batch_size: int = 1024,
) -> float:
    """
    Measure inference latency in milliseconds per sample.

    Args:
        model: Fitted model instance with decision_function or score method.
        X_test: Test features matrix.
        n_runs: Repetitions for stable timing.
        batch_size: Mini-batch size.

    Returns:
        Latency in milliseconds per individual test sample.
    """
    # Warm-up run
    sample_subset = X_test[: min(100, len(X_test))]
    if hasattr(model, "decision_function"):
        _ = model.decision_function(sample_subset)
    elif hasattr(model, "score"):
        _ = model.score(sample_subset)

    N = len(X_test)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if hasattr(model, "decision_function"):
            _ = model.decision_function(X_test, batch_size=batch_size)
        elif hasattr(model, "score"):
            _ = model.score(X_test)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    mean_time_sec = float(np.mean(times))
    ms_per_sample = (mean_time_sec / N) * 1000.0
    return round(ms_per_sample, 4)
