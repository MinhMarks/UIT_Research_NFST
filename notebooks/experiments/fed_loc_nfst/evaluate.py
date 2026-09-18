"""
FL-LOC-NFST Evaluation Utilities
==================================
Shared scoring and evaluation functions for both centralized and FL modes.
Mirror the logic in OC_NFST_memory_optimized_simple_scoring.py for
apples-to-apples comparison.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================
# Projection
# ============================================================

def project_to_null(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project X (n, d) through W (d, L) → (n, L). float32."""
    return (X.astype(np.float32) @ W.astype(np.float32)).astype(np.float32)


# ============================================================
# FAISS / NumPy Scoring
# ============================================================

def _numpy_min_dist(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Brute-force L2 distance from each query point to nearest ref point.
    query: (n_q, L), ref: (n_r, L) → (n_q,)
    """
    diffs = query[:, np.newaxis, :] - ref[np.newaxis, :, :]  # (n_q, n_r, L)
    return np.min(np.linalg.norm(diffs, axis=2), axis=1)


def _faiss_min_dist(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    FAISS-accelerated L2 distance (n_q → nearest in ref).
    query: (n_q, L), ref: (n_r, L) → (n_q,)
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    ref   = np.ascontiguousarray(ref,   dtype=np.float32)
    index = faiss.IndexFlatL2(ref.shape[1])
    index.add(ref)
    D, _ = index.search(query, k=1)
    return np.sqrt(np.maximum(D[:, 0], 0.0))


def min_dist_to_centers(query: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Distance from each test point to nearest null-space center.
    Automatically uses FAISS if available, else numpy fallback.
    """
    if FAISS_AVAILABLE:
        return _faiss_min_dist(query, centers)
    return _numpy_min_dist(query, centers)


# ============================================================
# Scoring
# ============================================================

def compute_fl_scores(
    X_test: np.ndarray,
    W: np.ndarray,
    null_centers: np.ndarray,
    max_train: float,
) -> np.ndarray:
    """
    Score test points using the FL-broadcast W matrix.

    Mirrors compute_scores() in OC_NFST_memory_optimized_simple_scoring.py.

    Returns
    -------
    y_proba : (n_test, 2) float32
        Column 0 = proximity-to-normal (high = normal)
        Column 1 = anomaly score (high = anomalous)
    """
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    null_test = project_to_null(X_test, W)
    y_score = min_dist_to_centers(null_test, null_centers)

    y_proba = np.zeros((len(y_score), 2), dtype=np.float32)
    y_proba[:, 1] = np.minimum(y_score / (max_train + 1e-10), 1.0)
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    return y_proba


# ============================================================
# Evaluation (Youden's J threshold — exact replica of centralized)
# ============================================================

def evaluate_scores(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Evaluate FL model performance using Youden's J optimal threshold.

    Note: y_true convention — 0 = normal, 1 = anomaly (same as centralized).
    y_true_flipped = 1 - y_true → normal becomes positive class (matching notebook).
    """
    y_true_flipped = (1 - y_true).astype(int)
    y_prob = y_proba[:, 0]  # proximity-to-normal

    # Guard: if all labels same, metrics are degenerate
    if len(np.unique(y_true_flipped)) < 2:
        return {
            "AUCROC": 0.0, "AUCPR": 0.0, "Accuracy": 0.0,
            "MCC": 0.0, "F1 Score": 0.0, "Precision": 0.0,
            "Recall": 0.0, "Threshold": 0.5
        }

    fpr, tpr, thresholds = roc_curve(y_true_flipped, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thr = thresholds[optimal_idx]

    y_pred = (y_prob >= optimal_thr).astype(int)

    return {
        "AUCROC":    round(roc_auc_score(y_true_flipped, y_prob) * 100, 4),
        "AUCPR":     round(average_precision_score(y_true_flipped, y_prob) * 100, 4),
        "Accuracy":  round(accuracy_score(y_true_flipped, y_pred) * 100, 4),
        "MCC":       round(matthews_corrcoef(y_true_flipped, y_pred), 4),
        "F1 Score":  round(f1_score(y_true_flipped, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true_flipped, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true_flipped, y_pred, zero_division=0), 4),
        "Threshold": round(float(optimal_thr), 6),
    }
