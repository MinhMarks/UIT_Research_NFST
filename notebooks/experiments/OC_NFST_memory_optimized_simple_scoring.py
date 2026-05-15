"""
LOC-NFST Memory-Optimized & Simple Scoring Version
====================================================

Optimizations and structural simplifications applied:

1. [float32 throughout]  All matrices use float32 instead of float64.
   → Cuts RAM consumption by ~50% for P_t, P_w, S_w, U, W.

2. [Single SVD call]  Previously SVD was called 3× on P_t/P_w.
   Now we compute it once and reuse the results.

3. [No sparse wrapper]  Removed the unnecessary `sp(X).dot(sp(npd)).toarray()`.
   Replaced with direct `X @ npd` matrix multiplication.

4. [Explicit del of temporaries]  P_t, P_w are deleted immediately after
   their scatter matrices are formed, releasing RAM early.

5. [Incremental S_w computation]  S_w is accumulated cluster-by-cluster
   instead of building the full P_w matrix (d × N). This reduces the
   peak memory from O(d × N) to O(d × max_cluster_size).

6. [Decoupled Inference RAM Optimization] Removed `X_train` from inference ops.
   Previously, scoring re-projected all training sets leading to heavily skewed 
   inference peak RAM measurements. Now, inference is fully autonomous.

7. [Explicit Null-Space Centroids (Simplified Scoring)] 
   Proximity distances are no longer mapped through backward original-space alignment. 
   Instead, representative centers (`null_centers`) are constructed by explicitly 
   projecting `X_train` to the null space and then taking the average for each class.
   For inference, the score relies on distance to the `null_centers` and is normalized 
   by `max_train` (the maximum self-distance encountered during Train phase). `max_train` 
   is passed down efficiently as an isolated float scaler.
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
from joblib import Parallel, delayed

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
    Load predefined Train (Normal) and Test sets.
    If noise > 0, extract anomalies from Test set, remove them to prevent leakage,
    and inject them into Train set.
    """
    X_train = train_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = train_data.iloc[:, -1].to_numpy()
    
    X_test = test_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_data.iloc[:, -1].to_numpy()
    
    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_percentage / 100))
    
    if noise_count > 0:
        anom_idx = np.where(y_test == 1)[0]
        if len(anom_idx) > 0:
            chosen_anom_idx = np.random.choice(anom_idx, size=min(noise_count, len(anom_idx)), replace=False)
            X_noise = X_test[chosen_anom_idx]
            
            # Remove chosen anomalies from test set to avoid data leakage
            mask = np.ones(len(y_test), dtype=bool)
            mask[chosen_anom_idx] = False
            X_test, y_test = X_test[mask], y_test[mask]
            
            # Inject to Train (labeled as 0 internally for OC clustering)
            X_train = np.vstack((X_train, X_noise))
            y_train = np.concatenate((y_train, np.zeros(len(X_noise))))
            
    return X_train, y_train, X_test, y_test


def drop_metadata_features(df: pd.DataFrame, logger: logging.Logger):
    """
    Drop known non-informative BoTIoT/IoT features if they are present.
    These features (like timestamps, IDs) can cause huge artificial distances
    or perfectly inverted rankings in one-class models.
    """
    meta_cols = [
        'pkSeqID', 'stime', 'ltime', 'seq', 'saddr', 'daddr', 'sport', 'dport',
        'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'state', 'flgs', 'proto'
    ]
    dropped = []
    for col in meta_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
    if dropped:
        logger.info(f"Dropped metadata features: {dropped}")
    return df


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


def compute_scores(X_test, W, null_centers, max_train):
    """
    Compute scores simply: Project test points to the null space, 
    and find the minimum distance from the point to the projected centers.
    """
    X_test  = np.ascontiguousarray(X_test,  dtype=np.float32)

    # 1. Project to null-space
    null_test  = project_to_null(X_test, W)

    # 2. Distance to nearest singular/center in null space
    if FAISS_AVAILABLE:
        y_score = faiss_test_score(null_test, null_centers)   # Dist from test to nearest projected center
    else:
        # Fallback numpy (slower)
        dists_test = np.linalg.norm(null_test[:, np.newaxis, :] - null_centers[np.newaxis, :, :], axis=2)
        y_score = np.min(dists_test, axis=1)

    # 3. Convert distance to an anomaly probability score [0, 1] using max_train normalization
    y_proba = np.zeros((len(y_score), 2), dtype=np.float32)
    y_proba[:, 1] = np.minimum(y_score / (max_train + 1e-10), 1.0)
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    
    return y_proba


# ============================================================================
# EVALUATION — exact replica of Model_evaluating() from the notebook:
#   y_true_flipped = 1 - y_true          (normal becomes the positive class)
#   y_prob = y_proba[:, 0]               (proximity-to-normal score)
#   Youden’s J  → optimal threshold     (from ROC curve)
#   Metrics: AUC-ROC, AUC-PR, Accuracy, MCC, F1, Precision, Recall
# ============================================================================
def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Evaluate model performance using the same logic as Model_evaluating()
    in OC-NSFT_old_noise_Kmean_threshold.ipynb.

    Returns a dict with all metrics and the optimal Youden threshold.
    """
    print("..............................Report Parameter...............................")

    y_true_flipped = (1 - y_true).astype(int)   # normal → 1, anomaly → 0
    y_prob = y_proba[:, 0]                        # proximity-to-normal score

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

    # --- Direct Feed Optimization ---
    # The new generate_oc_datasets.py guarantees that Train is 100% Normal 
    # (or near it) and Test contains anomalies. No more concatenating and resplitting!
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    log.info(f"Loaded Raw Shape -> Train: {df_train.shape}, Test: {df_test.shape}")

    # --- Feature cleaning specifically for BoTIoT/IoT metadata ---
    df_train = drop_metadata_features(df_train, log)
    df_test = drop_metadata_features(df_test, log)

    X_train, y_train, X_test, y_test = preprocess_data_noise(
        df_train, df_test, noise_pct
    )

    # Impute just in case (though generator usually handles this)
    X_train[np.isinf(X_train)] = np.nan
    X_test[np.isinf(X_test)] = np.nan
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    # Print dataset info AFTER imputation (values are final here)
    log.info(f"Training size : {len(X_train):,}  |  Testing size: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    csv_header_written = os.path.exists(out_path) if out_path else False

    def _process_one_cluster(n_clusters):
        # ---- Track peak RAM for TRAINING ----
        tracemalloc.start()
        try:
            X_clustered, y_clustered, centers = cluster_kmeans(X_train, n_clusters)
            actual_k = len(np.unique(y_clustered))

            # Train: compute NPD projection matrix W
            W, train_score_time_start = calculate_NPD_optimized(X_clustered, y_clustered)
            
            # Create null_centers: chiếu toàn bộ X_train sang null-space, sau đó lấy trung bình theo từng class
            t_center_start = time.time()
            null_train = project_to_null(X_clustered, W)
            null_centers = np.array([
                np.mean(null_train[y_clustered == cls], axis=0)
                for cls in np.unique(y_clustered)
            ], dtype=np.float32)
            
            # Tính max_train: Khoảng cách lớn nhất từ điểm train tới các trung bình trên
            if FAISS_AVAILABLE:
                train_score = faiss_test_score(null_train, null_centers)
            else:
                dists_train = np.linalg.norm(null_train[:, np.newaxis, :] - null_centers[np.newaxis, :, :], axis=2)
                train_score = np.min(dists_train, axis=1)
            max_train = float(np.max(train_score))

            train_time = train_score_time_start + (time.time() - t_center_start)
            
        except Exception as e:
            tracemalloc.stop()
            return {"error": f"[ERROR ncluster={n_clusters}] (train): {e}"}

        _, peak_bytes_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_train = round(peak_bytes_train / 1e6, 3)

        # ---- Track peak RAM for INFERENCE ----
        tracemalloc.start()
        try:
            t_test = time.time()
            y_proba = compute_scores(X_test, W, null_centers, max_train)
            test_time = time.time() - t_test
        except Exception as e:
            tracemalloc.stop()
            return {"error": f"[ERROR ncluster={n_clusters}] (test): {e}"}

        _, peak_bytes_test = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_test = round(peak_bytes_test / 1e6, 3)

        try:
            metrics = evaluate(y_test, y_proba)
        except Exception as e:
            return {"error": f"[EVAL ERROR ncluster={n_clusters}]: {e}"}

        auc_roc  = metrics["AUCROC"]
        auc_pr   = metrics["AUCPR"]

        result = {
            "scaler":               scaler_name,
            "Dataset":              os.path.basename(train_path),
            "nCluster":             actual_k,
            "nCluster_requested":   n_clusters,
            "noise_percentage":     noise_pct,
            "AUCROC":               metrics["AUCROC"],
            "AUCPR":                metrics["AUCPR"],
            "Accuracy":             metrics["Accuracy"],
            "MCC":                  metrics["MCC"],
            "F1 Score":             metrics["F1 Score"],
            "Precision":            metrics["Precision"],
            "Recall":               metrics["Recall"],
            "Threshold (Youden)":   metrics["Threshold"],
            "Time Train":           round(train_time, 4),
            "Time Test":            round(test_time, 4),
            "Peak RAM Train (MB)":  peak_mb_train,
            "Peak RAM Test (MB)":   peak_mb_test,
        }
        return {
            "result": result,
            "log_msg": (
                f"[ncluster={actual_k}/{n_clusters}] "
                f"AUC-ROC={auc_roc:.2f}%  AUC-PR={auc_pr:.2f}%  "
                f"F1={metrics['F1 Score']:.4f}  MCC={metrics['MCC']:.4f}  "
                f"Train={train_time:.3f}s  Test={test_time:.3f}s  "
                f"Peak RAM Train={peak_mb_train} MB  Peak RAM Test={peak_mb_test} MB"
            )
        }

    # Run in parallel using all available cores, maximizing RAM utilization for speed (-1 jobs)
    log.info(f"Firing up Parallel execution for {len(n_clusters_list)} cluster configs...")
    parallel_outputs = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_one_cluster)(nc) for nc in n_clusters_list
    )

    results = []
    for output in parallel_outputs:
        if "error" in output:
            log.error(output["error"])
        else:
            res = output["result"]
            log.info(output["log_msg"])
            
            # --- Ghi ngay lập tức vào CSV ---
            if out_path is not None:
                pd.DataFrame([res]).to_csv(
                    out_path, mode='a',
                    header=not csv_header_written,
                    index=False
                )
                csv_header_written = True
            results.append(res)

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":

    # Adjust this path via DATA_DIR env variable for server deployment
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _default_data = os.path.normpath(
        os.path.join(_script_dir, '..', '..', 'Datascaled', 'Official_OC_Data')
    )
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)

    # --- Match the notebook's configuration exactly ---
    # DATASETS = ['data_CICIoT2023', 'data_ToNIoT', 'data_N_BaIoT', 'data_BoTIoT']
    DATASETS = ['data_CICIoT2023', 'data_BoTIoT']
    
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
    experiment_name = f"Experiment_OC_NFST_MemOpt_{RUN_TIMESTAMP}"
    
    # Create the central experiment directory
    exp_dir = os.path.join(_script_dir, 'outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    for prefix in DATASETS:
        # One CSV and one log file per dataset per run (timestamped)
        out_path = os.path.join(exp_dir, f"{prefix}_memopt.csv")
        log_path = os.path.join(exp_dir, f"{prefix}_memopt.log")

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

