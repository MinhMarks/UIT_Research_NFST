"""
OC-NFST Inverted Variant — "Anomaly as Template"
==================================================

BẢN ĐẢO NGƯỢC: Train trên anomaly (label=1), phát hiện điểm normal (label=0).

Lý do tạo bản này:
  - Dataset có label: normal=0, anomaly=1
  - AUC-PR ưu tiên class có label=1 (positive class)
  - Phiên bản gốc train trên normal → y_proba[:,0] = normal score
    → AUC-PR của normal thấp vì normal là label=0 (negative)
  - Bản này: train trên anomaly → y_proba[:,1] = distance to anomaly
    → Điểm XA khỏi anomaly = likely NORMAL
    → Dùng y_proba[:,1] làm normal score
    → Remap y_true: normal→1 (positive), anomaly→0
    → AUC-PR tính trên normal class với label=1 → chính xác

Chỉ 3 điểm thay đổi so với OC_NFST_memory_optimized.py:
  1. preprocess_data_noise: train trên label=1 (anomaly), noise từ label=0 (normal)
  2. evaluate: y_prob = y_proba[:, 1] thay vì y_proba[:, 0]
  3. Output naming: thêm '_anomaly_template' vào tên file
"""

import os
import sys
import time
import random
import logging
import tracemalloc
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not installed. Falling back to numpy scoring.")
    print("          Install with: pip install faiss-cpu  (or faiss-gpu for GPU)")


# ============================================================================
# SEEDING
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# ============================================================================
# LOGGING
# Setup a logger that writes to BOTH console and a timestamped .log file.
# All print() inside run_experiment use logger so every run is fully recorded.
# ============================================================================
def setup_logger(log_path: str, name: str = "nfst") -> logging.Logger:
    """
    Create (or reuse) a logger that mirrors output to:
      1. Console (stdout)
      2. The given log_path (append mode)

    Args:
        log_path: Absolute path to the .log file.
        name:     Logger name (use one per dataset to avoid handler duplication).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — append mode so multiple runs accumulate in the same file
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_data_noise(train_data, test_data, noise_percentage=1):
    """
    [INVERTED] Train trên anomaly (label=1).
    Noise được lấy từ normal (label=0) đưa vào training.
    X_test giữ nguyên label gốc để evaluate.
    """
    X_train_total = train_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train_total = train_data.iloc[:, -1].to_numpy()

    # [CHANGE 1] Train trên ANOMALY (label=1) thay vì normal (label=0)
    X_train = X_train_total[y_train_total == 1]
    y_train = y_train_total[y_train_total == 1]

    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_percentage / 100))

    # [CHANGE 1b] Noise được lấy từ NORMAL (label=0) đưa vào training
    X_noise = X_train_total[y_train_total == 0]
    if noise_count > 0 and len(X_noise) > 0:
        idx = np.random.choice(X_noise.shape[0],
                               size=min(noise_count, len(X_noise)),
                               replace=False)
        X_noise = X_noise[idx]
        X_train = np.vstack((X_train, X_noise))
        y_train = np.concatenate((y_train, np.ones(X_noise.shape[0])))

    X_test = test_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_data.iloc[:, -1].to_numpy()
    return X_train, y_train, X_test, y_test


# ============================================================================
# CLUSTERING
# ============================================================================
def cluster_kmeans(data: np.ndarray, k: int):
    """K-Means clustering. Returns sorted data & labels.
    
    Bug fix: k is capped to n_samples so KMeans never crashes when
    n_clusters_list contains values larger than the training set size.
    """
    k = min(k, len(data))          # cap k so KMeans never gets k > n_samples
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    sorted_idx = np.argsort(labels)
    return data[sorted_idx], labels[sorted_idx], kmeans.cluster_centers_.astype(np.float32)


# ============================================================================
# MEMORY-OPTIMIZED NPD CALCULATION
# Key changes vs original:
#   - float32 input (halves RAM vs float64)
#   - Incremental S_w: never materializes the full d×N P_w matrix
#   - Single SVD call (was 3 calls previously)
#   - Explicit del of large intermediates
# ============================================================================
def calculate_NPD_optimized(X: np.ndarray, y: np.ndarray, epsilon: float = 1e-6):
    """
    Compute Null Projecting Directions (NPDs) with minimal RAM usage.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), float32
    y : np.ndarray, shape (n_samples,) — cluster labels
    epsilon : float — threshold for singular value near-zero detection

    Returns
    -------
    W : np.ndarray, shape (n_features, L) — projection matrix
    training_time : float — wall-clock seconds
    """
    t0 = time.time()

    # Work in (d, N) layout — still float32
    X = np.ascontiguousarray(X.T, dtype=np.float32)   # shape: (d, N)
    d, N = X.shape
    classes = np.unique(y)

    # --- Total scatter basis (one SVD call) ---
    mean_total = np.mean(X, axis=1, keepdims=True)     # (d, 1)
    P_t = X - mean_total                                # (d, N) — temporary

    # OPT: single SVD, reuse U for both rank detection and projection
    U, s_t, _ = np.linalg.svd(P_t, full_matrices=False)  # U: (d, min(d,N))
    rank_Pt = int(np.sum(s_t > epsilon))
    Q = U[:, :rank_Pt].astype(np.float32)              # (d, rank_Pt)

    del P_t, U, s_t                                     # OPT: free large arrays early

    # --- Within-class scatter (incremental, no P_w matrix) ---
    # S_w = sum_{i} sum_{x in C_i} (x - m_i)(x - m_i)^T  /  N
    # Build incrementally to avoid storing d×N P_w
    S_w = np.zeros((d, d), dtype=np.float32)
    for cls in classes:
        mask = (y == cls)
        X_cls = X[:, mask]                             # (d, n_i) — view, no copy
        m_cls = np.mean(X_cls, axis=1, keepdims=True)  # (d, 1)
        diff = X_cls - m_cls                           # (d, n_i) — temporary
        S_w += diff @ diff.T                           # accumulate
        del diff                                        # OPT: free each cluster's diff

    S_w /= N

    del X                                               # OPT: free transposed data copy

    # --- Null space in the projected subspace ---
    A = Q.T @ S_w @ Q                                  # (rank_Pt, rank_Pt)
    del S_w                                             # OPT: free scatter matrix

    B = null_space(A)                                   # (rank_Pt, L)
    del A

    W = (Q @ B).astype(np.float32)                     # (d, L)

    training_time = time.time() - t0
    print(f"  NPD computed: W shape={W.shape}, time={training_time:.4f}s")
    return W, training_time


# ============================================================================
# PROJECTION
# ============================================================================
def project_to_null(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project X (n, d) through W (d, L) → (n, L). float32, no sparse."""
    return (X @ W).astype(np.float32)


# ============================================================================
# FAISS SCORING — matches the notebook's learn() function exactly
#
# Notebook logic (OC-NSFT_old_noise_Kmean_threshold.ipynb):
#   null_point_X      = sp(X_train) @ sp(npd)   → projected train
#   null_point_X_test = sp(X_test)  @ sp(npd)   → projected test
#
#   train_score_tmp = distance_vector(null_point_X, null_point_X)   # n×n matrix
#   train_score_tmp[i,i] = 1e9                                       # mask diagonal
#   train_score = np.amin(train_score_tmp, axis=1)                   # min NN distance
#
#   y_score = minimum_distance(null_point_X_test, null_point_X)     # test→train min dist
#
#   y_proba[:, 1] = np.minimum(y_score / np.max(train_score), 1)    # normalised score
#   y_proba[:, 0] = 1 - y_proba[:, 1]
#
# FAISS replaces the O(n²) and O(n·m) loops with O(n log n) / O(m log n).
# ============================================================================
def _build_faiss_index(X: np.ndarray):
    """Build a FAISS flat L2 index from float32 matrix (n, d)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index


def faiss_train_score(null_train: np.ndarray) -> np.ndarray:
    """
    Compute for each training point its distance to its nearest *other* training point.
    Equivalent to: distance_vector(X,X) with diagonal set to 1e9, then amin(axis=1).
    FAISS: search k=2, take the 2nd result (1st = self with dist=0).
    Returns squared-L2 → sqrt applied. Shape: (n_train,)
    """
    index = _build_faiss_index(null_train)
    distances, _ = index.search(null_train, k=2)      # (n, 2) squared L2
    return np.sqrt(np.maximum(distances[:, 1], 0.0))  # nearest non-self


def faiss_test_score(null_test: np.ndarray, null_train: np.ndarray) -> np.ndarray:
    """
    Compute for each test point distance to nearest training point.
    Equivalent to: minimum_distance(X_test, X_train).
    Returns shape: (n_test,)
    """
    index = _build_faiss_index(null_train)
    distances, _ = index.search(null_test, k=1)       # (n_test, 1) squared L2
    return np.sqrt(np.maximum(distances[:, 0], 0.0))


def _numpy_train_score(null_train: np.ndarray) -> np.ndarray:
    """Numpy fallback for train_score (O(n²))."""
    sq = np.sum(null_train ** 2, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2 * (null_train @ null_train.T)
    np.fill_diagonal(dist2, np.inf)
    return np.sqrt(np.maximum(dist2.min(axis=1), 0.0))


def _numpy_test_score(null_test: np.ndarray, null_train: np.ndarray) -> np.ndarray:
    """Numpy fallback for test_score (O(n·m))."""
    sq_te = np.sum(null_test  ** 2, axis=1, keepdims=True)
    sq_tr = np.sum(null_train ** 2, axis=1)
    cross = null_test @ null_train.T
    dist2 = sq_te + sq_tr - 2 * cross
    return np.sqrt(np.maximum(dist2.min(axis=1), 0.0))


def compute_scores(null_train: np.ndarray, null_test: np.ndarray):
    """
    Compute y_proba matching the notebook exactly:
        y_proba[:, 1] = min(y_score / max(train_score), 1)
        y_proba[:, 0] = 1 - y_proba[:, 1]

    Returns y_proba (n_test, 2).
    """
    null_train = np.ascontiguousarray(null_train, dtype=np.float32)
    null_test  = np.ascontiguousarray(null_test,  dtype=np.float32)

    if FAISS_AVAILABLE:
        train_score = faiss_train_score(null_train)            # (n_train,)
        y_score     = faiss_test_score(null_test, null_train)  # (n_test,)
    else:
        train_score = _numpy_train_score(null_train)
        y_score     = _numpy_test_score(null_test, null_train)

    max_train = np.max(train_score)
    y_proba = np.zeros((len(y_score), 2), dtype=np.float32)
    y_proba[:, 1] = np.minimum(y_score / (max_train + 1e-10), 1.0)
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    y_proba = np.nan_to_num(y_proba, nan=1.0)
    return y_proba


# ============================================================================
# EVALUATION — INVẬRTED: train trên anomaly, xác định normal
#
#   y_true_flipped = 1 - y_true          (normal → 1, anomaly → 0, giữ AUCPR ưu tiên normal)
#   y_prob = y_proba[:, 1]               [CHANGE 2] distance-to-anomaly-template score
#                                         (cao = xa anomaly = likely NORMAL)
#   Youden’s J  → optimal threshold     (từ ROC curve)
#   Metrics: AUC-ROC, AUC-PR, Accuracy, MCC, F1, Precision, Recall
# ============================================================================
def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Inverted evaluation: model trained on anomaly data.
    y_proba[:, 1] = distance to anomaly template (high = normal = positive class).
    Remap: normal→1 (positive for AUC-PR), anomaly→0.
    """
    print("..............................Report Parameter...............................")

    y_true_flipped = (1 - y_true).astype(int)   # normal → 1, anomaly → 0

    # [CHANGE 2] Dùng y_proba[:, 1] thay vì [:, 0]
    # y_proba[:, 1] = min_dist(test, anomaly_train) / max_train_dist
    # Điểm xa anomaly (y_proba[:,1] cao) = likely NORMAL
    y_prob = y_proba[:, 1]                        # distance-to-anomaly = normal score

    # Youden’s J statistic → optimal classification threshold
    fpr, tpr, thresholds = roc_curve(y_true_flipped, y_prob)
    j_scores     = tpr - fpr
    optimal_idx  = np.argmax(j_scores)
    optimal_thr  = thresholds[optimal_idx]

    print(f"Optimal threshold (Youden's J): {optimal_thr}")

    y_pred_opt = (y_prob >= optimal_thr).astype(int)

    mcc      = matthews_corrcoef(y_true_flipped, y_pred_opt)
    f1       = f1_score(y_true_flipped, y_pred_opt, zero_division=0)
    ppv      = precision_score(y_true_flipped, y_pred_opt, zero_division=0)
    recall   = recall_score(y_true_flipped, y_pred_opt, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_opt)
    auc_roc  = roc_auc_score(y_true_flipped, y_prob)
    auc_pr   = average_precision_score(y_true_flipped, y_prob)

    print(f"AUCROC:        {auc_roc * 100:.4f}")
    print(f"AUCPR:         {auc_pr  * 100:.4f}")
    print(f"Accuracy:      {accuracy * 100:.4f}")
    print(f"MCC:           {mcc:.4f}")
    print(f"F1 score:      {f1:.4f}")
    print(f"PPV (Prec):    {ppv:.4f}")
    print(f"TPR (Recall):  {recall:.4f}")

    return {
        "AUCROC":     round(auc_roc  * 100, 4),
        "AUCPR":      round(auc_pr   * 100, 4),
        "Accuracy":   round(accuracy * 100, 4),
        "MCC":        round(mcc,            4),
        "F1 Score":   round(f1,             4),
        "Precision":  round(ppv,            4),
        "Recall":     round(recall,         4),
        "Threshold":  round(float(optimal_thr), 6),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_experiment(train_path: str, test_path: str,
                   n_clusters_list: list = None,
                   noise_pct: float = 1.0,
                   scaler_name: str = "unknown",
                   out_path: str = None,
                   logger: logging.Logger = None):
    """
    Full NFST pipeline with memory optimizations.
    Loops over each value of n_clusters (brute-force like the notebook).
    Measures peak RAM for each (n_cluster, dataset) combination.
    Logs every event to the provided logger (file + console).
    Returns a list of result dicts.
    """
    log = logger or logging.getLogger("nfst")

    if n_clusters_list is None:
        n_clusters_list = [1, 100, 139, 184, 301]

    log.info("=" * 60)
    log.info(f"Dataset  : {os.path.basename(train_path)}")
    log.info(f"Scaler   : {scaler_name}")
    log.info(f"Noise    : {noise_pct}%")
    log.info(f"nClusters: {len(n_clusters_list)} values ({n_clusters_list[0]}..{n_clusters_list[-1]})")
    log.info("=" * 60)

    # --- Giống notebook: merge train+test rồi re-split 80/20 ---
    # (notebook dùng test_size=0.2, random_state=42)
    # Lý do: các file CSV được tạo từ split cũ không đồng nhất,
    # cần normalize lại để đảm bảo tỉ lệ nhất quán.
    df_train = pd.read_csv(train_path).dropna()
    df_test  = pd.read_csv(test_path).dropna()

    df_full = pd.concat([df_train, df_test], ignore_index=True)

    log.info(f"df_full shape: {df_full.shape}")
    df_train_new, df_test_new = train_test_split(
        df_full, test_size=0.2, random_state=42
    )

    X_train, y_train, X_test, y_test = preprocess_data_noise(
        df_train_new, df_test_new, noise_pct
    )

    # Handle inf values once — fit ONLY on training data, transform test separately
    imputer = SimpleImputer(strategy="mean")
    X_train[np.isinf(X_train)] = np.nan
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test[np.isinf(X_test)] = np.nan
    X_test = imputer.transform(X_test).astype(np.float32)

    # Print dataset info AFTER imputation (values are final here)
    log.info(f"Training size : {len(X_train):,}  |  Testing size: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    results = []
    csv_header_written = os.path.exists(out_path) if out_path else False

    for n_clusters in n_clusters_list:
        # ---- Track peak RAM for this specific n_clusters run ----
        tracemalloc.start()
        try:
            # Cluster (k capped inside cluster_kmeans if n_clusters > n_train)
            X_clustered, y_clustered, _ = cluster_kmeans(X_train, n_clusters)
            actual_k = len(np.unique(y_clustered))

            # Train: compute NPD projection matrix W
            W, train_time = calculate_NPD_optimized(X_clustered, y_clustered)

            # Project training data into null space
            null_train = project_to_null(X_train, W)   # (n_train, L)

            # Score: FAISS-accelerated, matching notebook's learn() exactly
            t_test = time.time()
            null_test = project_to_null(X_test, W)     # (n_test, L)
            y_proba   = compute_scores(null_train, null_test)
            test_time = time.time() - t_test

        except Exception as e:
            log.error(f"[ERROR ncluster={n_clusters}]: {e}")
            tracemalloc.stop()
            continue

        finally:
            pass  # tracemalloc will be stopped below after get_traced_memory

        # Peak RAM at the highest point within this n_clusters run
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = round(peak_bytes / 1e6, 3)

        try:
            metrics = evaluate(y_test, y_proba)
        except Exception as e:
            log.error(f"[EVAL ERROR ncluster={n_clusters}]: {e}")
            continue

        auc_roc  = metrics["AUCROC"]
        auc_pr   = metrics["AUCPR"]

        result = {
            "scaler":              scaler_name,
            "Dataset":             os.path.basename(train_path),
            "nCluster":            actual_k,      # actual clusters (may be < requested)
            "nCluster_requested":  n_clusters,    # what was requested
            "noise_percentage":    noise_pct,
            "AUCROC":              metrics["AUCROC"],
            "AUCPR":               metrics["AUCPR"],
            "Accuracy":            metrics["Accuracy"],
            "MCC":                 metrics["MCC"],
            "F1 Score":            metrics["F1 Score"],
            "Precision":           metrics["Precision"],
            "Recall":              metrics["Recall"],
            "Threshold (Youden)":  metrics["Threshold"],
            "Time Train":          round(train_time, 4),
            "Time Test":           round(test_time, 4),
            "Peak RAM Train (MB)": peak_mb,
        }

        log.info(
            f"[ncluster={actual_k}/{n_clusters}] "
            f"AUC-ROC={auc_roc:.2f}%  AUC-PR={auc_pr:.2f}%  "
            f"F1={metrics['F1 Score']:.4f}  MCC={metrics['MCC']:.4f}  "
            f"Train={train_time:.3f}s  Test={test_time:.3f}s  "
            f"Peak RAM={peak_mb} MB"
        )

        # --- Ghi ngay lập tức vào CSV sau mỗi ncluster ---
        if out_path is not None:
            pd.DataFrame([result]).to_csv(
                out_path, mode='a',
                header=not csv_header_written,
                index=False
            )
            csv_header_written = True

        results.append(result)

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":

    # Adjust this path via DATA_DIR env variable for server deployment
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _default_data = os.path.normpath(
        os.path.join(_script_dir, '..', '..', 'Datascaled', 'NoiseOCData')
    )
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)

    # --- Match the notebook's configuration exactly ---
    DATASETS = ['data_CICIoT2023', 'data_ToNIoT', 'data_N_BaIoT', 'data_BoTIoT']

    # Notebook: scaler_names = ['StandardScaler','MinMaxScaler','Normalizer',
    #                           'QuantileTransformer','RobustScaler','Normalizer']
    SCALERS = ['StandardScaler', 'MinMaxScaler', 'Normalizer',
               'QuantileTransformer', 'RobustScaler']   # deduped

    # Notebook: for ncluster in range(1, 301, 3)
    N_CLUSTERS_LIST = list(range(1, 301, 3))            # [1, 4, 7, ..., 298, 301]

    # Notebook: for noise in [0, 1, 3, 5]  (adjust here as needed)
    NOISE_LIST = [0, 1, 3, 5]

    # ------------------------------------------------------------------
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    for prefix in DATASETS:
        outputs_dir = os.path.join(_script_dir, 'outputs/noise')
        logs_dir    = os.path.join(_script_dir, 'logs')
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # One CSV and one log file per dataset per run (timestamped)
        # [CHANGE 3] Suffix _anomaly_template để phân biệt với bản gốc
        out_path = os.path.join(outputs_dir, f"{prefix}_anomaly_template_{RUN_TIMESTAMP}.csv")
        log_path = os.path.join(logs_dir,    f"{prefix}_anomaly_template_{RUN_TIMESTAMP}.log")

        # Setup logger for this dataset
        ds_logger = setup_logger(log_path, name=f"nfst.{prefix}")
        ds_logger.info(f"Run started at {RUN_TIMESTAMP}")
        ds_logger.info(f"Dataset prefix : {prefix}")
        ds_logger.info(f"Scalers        : {SCALERS}")
        ds_logger.info(f"N_CLUSTERS     : range(1, 301, 3) — {len(N_CLUSTERS_LIST)} values")
        ds_logger.info(f"NOISE_LIST     : {NOISE_LIST}")
        ds_logger.info(f"CSV output     : {out_path}")
        ds_logger.info(f"Log file       : {log_path}")
        ds_logger.info("-" * 60)

        for scaler in SCALERS:
            ds_logger.info(f"\n>>> Processing {prefix} | scaler={scaler}")

            # Notebook uses: f'Train_{scaler}_{prefix}.csv'
            train_path = os.path.join(DATA_DIR, f"Train_{scaler}_{prefix}.csv")
            test_path  = os.path.join(DATA_DIR, f"Test_{scaler}_{prefix}.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                ds_logger.warning(f"[SKIP] Files not found: {train_path}")
                continue

            for noise in NOISE_LIST:
                ds_logger.info(f"--- noise={noise}% ---")

                run_experiment(
                    train_path, test_path,
                    n_clusters_list=N_CLUSTERS_LIST,
                    noise_pct=noise,
                    scaler_name=scaler,
                    out_path=out_path,
                    logger=ds_logger,
                )

        ds_logger.info(f"Results saved to: {out_path}")
        ds_logger.info(f"Log saved to    : {log_path}")

    print("\nAll done.")

