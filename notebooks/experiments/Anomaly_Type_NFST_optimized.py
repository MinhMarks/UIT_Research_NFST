"""
Anomaly Type NFST Memory-Optimized Version
==========================================

This script benchmarks LOC-NFST across different synthetic anomaly types
(Local, Cluster, Global) generated from the pre-scaled Official_OC_Data.
It leverages identical memory-optimization strategies (float32, minimal S_w scatter
matrices, single SVD projection) as `OC_NFST_memory_optimized.py`.
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
from joblib import Parallel, delayed

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not installed. Falling back to numpy scoring.")


# ============================================================================
# SEEDING
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# ============================================================================
# LOGGING
# ============================================================================
def setup_logger(log_path: str, name: str = "nfst") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def drop_metadata_features(df: pd.DataFrame, logger: logging.Logger):
    """Drop non-informative metadata features (e.g. timestamps, IPs)."""
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
    k = min(k, len(data))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    sorted_idx = np.argsort(labels)
    return data[sorted_idx], labels[sorted_idx], kmeans.cluster_centers_.astype(np.float32)

# ============================================================================
# MEMORY-OPTIMIZED NPD CALCULATION
# ============================================================================
def calculate_NPD_optimized(X: np.ndarray, y: np.ndarray, epsilon: float = 1e-6):
    t0 = time.time()

    X = np.ascontiguousarray(X.T, dtype=np.float32)
    d, N = X.shape
    classes = np.unique(y)

    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total

    U, s_t, _ = np.linalg.svd(P_t, full_matrices=False)
    rank_Pt = int(np.sum(s_t > epsilon))
    Q = U[:, :rank_Pt].astype(np.float32)

    del P_t, U, s_t

    S_w = np.zeros((d, d), dtype=np.float32)
    for cls in classes:
        mask = (y == cls)
        X_cls = X[:, mask]
        m_cls = np.mean(X_cls, axis=1, keepdims=True)
        diff = X_cls - m_cls
        S_w += diff @ diff.T
        del diff

    S_w /= N
    del X

    A = Q.T @ S_w @ Q
    del S_w

    B = null_space(A)
    del A

    W = (Q @ B).astype(np.float32)

    training_time = time.time() - t0
    return W, training_time

# ============================================================================
# SCORING
# ============================================================================
def compute_dist_to_centers(X, W, centers):
    if FAISS_AVAILABLE:
        d = centers.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(centers.astype('float32'))
        _, nearest_idx = index.search(X.astype('float32'), 1)
        nearest_centers = centers[nearest_idx.flatten()]
    else:
        nearest_centers = []
        for i in range(0, len(X), 2000):
            batch = X[i:i+2000]
            dists = np.linalg.norm(batch[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            idx = np.argmin(dists, axis=1)
            nearest_centers.append(centers[idx])
        nearest_centers = np.vstack(nearest_centers)

    diff = X - nearest_centers
    projections = diff @ W
    return np.sqrt(np.sum(projections**2, axis=1))


def compute_scores(X_train, X_test, W, centers):
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test  = np.ascontiguousarray(X_test,  dtype=np.float32)

    train_score = compute_dist_to_centers(X_train, W, centers)
    y_score     = compute_dist_to_centers(X_test, W, centers)

    max_train = np.max(train_score)
    y_proba = np.zeros((len(y_score), 2), dtype=np.float32)
    y_proba[:, 1] = np.minimum(y_score / (max_train + 1e-10), 1.0)
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    y_proba = np.nan_to_num(y_proba, nan=1.0)
    return y_proba


def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_true_flipped = (1 - y_true).astype(int)   # normal -> 1, anomaly -> 0
    y_prob = y_proba[:, 0]

    fpr, tpr, thresholds = roc_curve(y_true_flipped, y_prob)
    j_scores     = tpr - fpr
    optimal_idx  = np.argmax(j_scores)
    optimal_thr  = thresholds[optimal_idx]

    y_pred_opt = (y_prob >= optimal_thr).astype(int)

    mcc      = matthews_corrcoef(y_true_flipped, y_pred_opt)
    f1       = f1_score(y_true_flipped, y_pred_opt, zero_division=0)
    ppv      = precision_score(y_true_flipped, y_pred_opt, zero_division=0)
    recall   = recall_score(y_true_flipped, y_pred_opt, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_opt)
    auc_roc  = roc_auc_score(y_true_flipped, y_prob)
    auc_pr   = average_precision_score(y_true_flipped, y_prob)

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
                   n_clusters_list: list,
                   anomaly_mode: str,
                   scaler_name: str,
                   out_path: str,
                   logger: logging.Logger):
                   
    log = logger

    log.info("=" * 60)
    log.info(f"Dataset      : {os.path.basename(train_path)}")
    log.info(f"Anomaly Mode : {anomaly_mode.upper()}")
    log.info(f"Scaler       : {scaler_name}")
    log.info(f"nClusters    : {len(n_clusters_list)} values")
    log.info("=" * 60)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    log.info(f"Loaded Shape -> Train: {df_train.shape}, Test: {df_test.shape}")

    df_train = drop_metadata_features(df_train, log)
    df_test = drop_metadata_features(df_test, log)

    X_train = df_train.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = df_train.iloc[:, -1].to_numpy()
    X_test  = df_test.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test  = df_test.iloc[:, -1].to_numpy()

    X_train[np.isinf(X_train)] = np.nan
    X_test[np.isinf(X_test)] = np.nan
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    csv_header_written = os.path.exists(out_path)

    def _process_one_cluster(n_clusters):
        tracemalloc.start()
        try:
            X_clustered, y_clustered, centers = cluster_kmeans(X_train, n_clusters)
            actual_k = len(np.unique(y_clustered))

            W, train_time = calculate_NPD_optimized(X_clustered, y_clustered)

            t_test = time.time()
            y_proba = compute_scores(X_train, X_test, W, centers)
            test_time = time.time() - t_test

        except Exception as e:
            tracemalloc.stop()
            return {"error": f"[ERROR ncluster={n_clusters}]: {e}"}

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = round(peak_bytes / 1e6, 3)

        try:
            metrics = evaluate(y_test, y_proba)
        except Exception as e:
            return {"error": f"[EVAL ERROR ncluster={n_clusters}]: {e}"}

        result = {
            "Anomaly_Mode":        anomaly_mode,
            "scaler":              scaler_name,
            "Dataset":             os.path.basename(train_path).split('_data_')[-1].replace('.csv',''), # Extracts clean dataset name
            "nCluster":            actual_k,      
            "nCluster_requested":  n_clusters,    
            "noise_percentage":    0, # Forced 0 for pure anomaly evaluation
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
        return {"result": result, "log_msg": f"[{anomaly_mode.upper()} - ncluster={actual_k}] AUC-ROC={metrics['AUCROC']:.2f}%  Train={train_time:.3f}s  RAM={peak_mb} MB"}

    log.info(f"Parallel execution on {os.cpu_count()} cores for {len(n_clusters_list)} clusters...")
    parallel_outputs = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_one_cluster)(nc) for nc in n_clusters_list
    )

    results = []
    for output in parallel_outputs:
        if "error" in output:
            log.error(output["error"])
        else:
            res = output["result"]
            log.info(output["log_msg"])
            
            if out_path:
                pd.DataFrame([res]).to_csv(
                    out_path, mode='a',
                    header=not csv_header_written,
                    index=False
                )
                csv_header_written = True
            results.append(res)
    return results

if __name__ == "__main__":
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _default_data = os.path.normpath(os.path.join(_script_dir, '..', '..', 'Datascaled', 'Official_Anomaly_Data'))
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)

    # DATASETS = ['BoTIoT', 'CICIoT2023', 'ToNIoT', 'N_BaIoT']
    DATASETS = [ 'EdgeIIoTset', 'IoTID20']
    SCALERS = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'QuantileTransformer', 'RobustScaler']
    ANOMALY_MODES = ['local', 'cluster', 'global']
    
    # 1...301 step 3
    N_CLUSTERS_LIST = list(range(1, 301, 3))

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Anomaly_Type_NFST_MemOpt_{RUN_TIMESTAMP}"
    
    exp_dir = os.path.join(_script_dir, 'outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    for prefix in DATASETS:
        for mode in ANOMALY_MODES:
            # Output files segmented by Dataset AND Mode
            out_path = os.path.join(exp_dir, f"{prefix}_{mode}_memopt.csv")
            log_path = os.path.join(exp_dir, f"{prefix}_{mode}_memopt.log")

            ds_logger = setup_logger(log_path, name=f"nfst.{prefix}.{mode}")
            ds_logger.info(f"Run started at {RUN_TIMESTAMP}")
            
            for scaler in SCALERS:
                train_path = os.path.join(DATA_DIR, f"Train_{mode}_{scaler}_data_{prefix}.csv")
                test_path  = os.path.join(DATA_DIR, f"Test_{mode}_{scaler}_data_{prefix}.csv")

                if not os.path.exists(train_path) or not os.path.exists(test_path):
                    ds_logger.warning(f"[SKIP] Files not found: {os.path.basename(train_path)}")
                    continue

                run_experiment(
                    train_path, test_path,
                    n_clusters_list=N_CLUSTERS_LIST,
                    anomaly_mode=mode,
                    scaler_name=scaler,
                    out_path=out_path,
                    logger=ds_logger,
                )

    print("\nAll anomaly type experiments finished successfully.")
