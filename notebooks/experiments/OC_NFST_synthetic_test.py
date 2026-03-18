"""
OC-NFST Synthetic Sanity-Check
================================

Mục đích: Kiểm tra xem thuật toán có hoạt động đúng không bằng cách chạy
trên dataset tổng hợp CÂN BẰNG, có cấu trúc rõ ràng (make_blobs).

Khác biệt với các file khác:
  - Không đọc file CSV — tự sinh data bằng sklearn.datasets.make_blobs
  - Data cân bằng 50/50 normal vs anomaly (hoặc tuỳ chỉnh)
  - Normal và anomaly tạo thành 2 cum riêng biệt rõ ràng (class_sep nửa)
  - Mục tiêu: AUC-ROC > 70% mới có nghĩa là thuật toán hoạt động

Chạy file này để kiểm tra nhanh:
    python OC_NFST_synthetic_test.py
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
from sklearn.datasets import make_blobs

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
def generate_balanced_synthetic(
    n_normal: int = 2000,
    n_anomaly: int = 2000,
    n_features: int = 20,
    n_normal_clusters: int = 5,
    difficulty: str = 'easy',
    test_ratio: float = 0.2,
    noise_pct: float = 0.0,
    seed: int = 42,
):
    """
    Sinh dataset tổng hợp với 2 nhóm features độc lập (đảm bảo tồn tại Null Space)
    Các mức phân bố độ khó (difficulty):
      - 'easy'               : Anomaly nằm xa hoàn toàn khỏi Normal (class_sep = 10).
      - 'hard_overlap'       : Anomaly nằm đè một phần lên rìa của Normal (class_sep = 1).
      - 'hard_anisotropic'   : Cụm Normal hình bầu dục (phân tán không đều), Anomaly nằm giữa các khoảng trống.
      - 'extreme_interp'     : Anomaly được nội suy từ các Normal center (nằm lọt thỏm trong không gian vỏ bọc của Normal).
    """
    rng = np.random.default_rng(seed)

    n_informative = min(n_features // 2, n_normal_clusters)
    n_noise_feat  = n_features - n_informative

    # Cấu hình theo difficulty
    class_sep = 10.0
    cluster_std = 0.5
    if difficulty == 'hard_overlap':
        class_sep = 1.0
        cluster_std = 1.5
    elif difficulty == 'hard_anisotropic':
        class_sep = 4.0
        cluster_std = 1.0  # Sẽ bị bóp méo sau
    elif difficulty == 'extreme_interp':
        class_sep = 0.0    # Không dùng class_sep, anomaly sinh từ interpolation
        cluster_std = 0.8

    # Sinh các center ngẫu nhiên cho normal (informative dims)
    normal_centers = rng.uniform(
        low=-class_sep, high=class_sep,
        size=(n_normal_clusters, n_informative)
    ).astype(np.float32)

    samples_per_cluster = [n_normal // n_normal_clusters] * n_normal_clusters
    samples_per_cluster[-1] += n_normal - sum(samples_per_cluster)

    X_normal_list, y_normal_list = [], []
    for i, (center, n) in enumerate(zip(normal_centers, samples_per_cluster)):
        X_info = np.tile(center, (n, 1)).astype(np.float32)
        X_noise = rng.normal(0, cluster_std, size=(n, n_noise_feat)).astype(np.float32)
        
        if difficulty == 'hard_anisotropic':
            # Bóp méo tạo hình bầu dục bằng cách nhân scale ngẫu nhiên cho từng chiều noise
            scales = rng.uniform(0.1, 3.0, size=(1, n_noise_feat)).astype(np.float32)
            X_noise *= scales

        X_normal_list.append(np.hstack([X_info, X_noise]))
        y_normal_list.append(np.zeros(n, dtype=np.float32))

    X_normal = np.vstack(X_normal_list)

    # Sinh Anomaly
    if difficulty == 'extreme_interp':
        # Anomaly = Nội suy tuyến tính ngẫu nhiên giữa 2 Normal centers bất kỳ
        # (Nằm hoàn toàn bên MẶT PHẲNG của normal, rất khó phát hiện)
        idx1 = rng.choice(n_normal_clusters, size=n_anomaly)
        idx2 = rng.choice(n_normal_clusters, size=n_anomaly)
        alpha = rng.uniform(0.1, 0.9, size=(n_anomaly, 1)).astype(np.float32)
        X_anom_info = alpha * normal_centers[idx1] + (1 - alpha) * normal_centers[idx2]
        X_anom_noise = rng.normal(0, cluster_std * 0.5, size=(n_anomaly, n_noise_feat)).astype(np.float32)
        X_anomaly = np.hstack([X_anom_info, X_anom_noise])
    else:
        if difficulty == 'hard_anisotropic':
            # Nằm len lỏi giữa các tâm
            anomaly_center = np.mean(normal_centers, axis=0, keepdims=True) + class_sep/2
        else:
            anomaly_center = np.full((1, n_informative), class_sep * 1.5, dtype=np.float32)
            anomaly_center[0, 0] += class_sep
            
        X_anom_info = np.tile(anomaly_center, (n_anomaly, 1)).astype(np.float32)
        X_anom_noise = rng.normal(0, cluster_std, size=(n_anomaly, n_noise_feat)).astype(np.float32)
        X_anomaly = np.hstack([X_anom_info, X_anom_noise])

    X_all = np.vstack([X_normal, X_anomaly])
    y_all = np.concatenate([np.zeros(len(X_normal)), np.ones(n_anomaly)]).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=test_ratio, random_state=seed, stratify=y_all
    )

    X_train = X_tr[y_tr == 0]
    y_train = y_tr[y_tr == 0]

    if noise_pct > 0:
        X_anm_tr = X_tr[y_tr == 1]
        n_noise = int(len(X_train) * noise_pct / 100)
        if n_noise > 0 and len(X_anm_tr) > 0:
            idx = rng.choice(len(X_anm_tr), size=min(n_noise, len(X_anm_tr)), replace=False)
            X_train = np.vstack([X_train, X_anm_tr[idx]])
            y_train = np.concatenate([y_train, np.zeros(len(idx))])

    print(f"  [Synthetic] diff={difficulty}  normal_clusters={n_normal_clusters}  "
          f"features={n_features} (info={n_informative}, noise={n_noise_feat})")
    print(f"  [Synthetic] train_n={len(X_train):,}  test_n={int(np.sum(y_te==0)):,}  test_a={int(np.sum(y_te==1)):,}")

    return X_train, y_train, X_te, y_te




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
# EVALUATION
#   - Train data: NORMAL (label=0)
#   - NFST score: khoảng cách test point tới vùng normal
#   => y_proba[:,0] chứa điểm (probability) of being NORMAL (0).
#   - AUCPR mong muốn label dương = normal → (1 - y_true)
# ============================================================================
def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Evaluates scoring where distance is measured from normal clusters.
    y_proba[:, 0] = score of being normal (higher = closer to normal train).
    """
    print("..............................Report Parameter...............................")

    y_true_flipped = (1 - y_true).astype(int)   # normal → 1, anomaly → 0

    # LỖI CŨ: dùng [:, 1] là khoảng cách tới normal (thực chất là ANOMALY score).
    # SỬA LẠI: dùng [:, 0] là NORMAL SCORE (trung thành với việc train trên normal class).
    y_prob = y_proba[:, 0]

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
# SYNTHETIC PIPELINE (thay thế run_experiment đốc CSV)
# ============================================================================
def run_experiment_synthetic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clusters_list: list = None,
    label: str = "synthetic",
    out_path: str = None,
    logger: logging.Logger = None,
):
    """
    Chạy NFST pipeline trực tiếp trên arrays (không cần file CSV).
    X_train: chỉ chứa normal samples.
    y_test : nhãn gốc (normal=0, anomaly=1).
    """
    log = logger or logging.getLogger("nfst")

    if n_clusters_list is None:
        n_clusters_list = list(range(1, 21, 2))  # [1,3,5,...,19] — nhỏ để test nhanh

    # Imputer (nhưng synthetic data không có inf/nan)
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test  = imputer.transform(X_test).astype(np.float32)

    log.info(f"Training size : {len(X_train):,}  |  Testing size: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    results = []
    csv_header_written = os.path.exists(out_path) if out_path else False

    for n_clusters in n_clusters_list:
        tracemalloc.start()
        try:
            X_clustered, y_clustered, _ = cluster_kmeans(X_train, n_clusters)
            actual_k = len(np.unique(y_clustered))

            W, train_time = calculate_NPD_optimized(X_clustered, y_clustered)

            null_train = project_to_null(X_train, W)

            t_test = time.time()
            null_test = project_to_null(X_test, W)
            y_proba   = compute_scores(null_train, null_test)
            test_time = time.time() - t_test

        except Exception as e:
            log.error(f"[ERROR ncluster={n_clusters}]: {e}")
            tracemalloc.stop()
            continue

        finally:
            pass

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = round(peak_bytes / 1e6, 3)

        try:
            metrics = evaluate(y_test, y_proba)
        except Exception as e:
            log.error(f"[EVAL ERROR ncluster={n_clusters}]: {e}")
            continue

        result = {
            "label":               label,
            "nCluster":            actual_k,
            "nCluster_requested":  n_clusters,
            **metrics,
            "Time Train":          round(train_time, 4),
            "Time Test":           round(test_time, 4),
            "Peak RAM Train (MB)": peak_mb,
        }

        verdict = "✅ PASS" if metrics["AUCROC"] > 70 else "❌ FAIL"
        log.info(
            f"{verdict} [ncluster={actual_k}] "
            f"AUC-ROC={metrics['AUCROC']:.2f}%  AUC-PR={metrics['AUCPR']:.2f}%  "
            f"F1={metrics['F1 Score']:.4f}  MCC={metrics['MCC']:.4f}  "
            f"Train={train_time:.3f}s  Test={test_time:.3f}s"
        )

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
# ENTRY POINT — Sanity Check trên synthetic data
# ============================================================================
if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Experiment_OC_NFST_Synthetic_{RUN_TIMESTAMP}"
    exp_dir = os.path.join(_script_dir, 'outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    out_path = os.path.join(exp_dir, "synthetic_sanity.csv")
    log_path = os.path.join(exp_dir, "synthetic_sanity.log")

    log = setup_logger(log_path, name="nfst.synthetic")
    log.info("=" * 60)
    log.info("OC-NFST SYNTHETIC SANITY CHECK")
    log.info("=" * 60)

    # ==========================================================
    # Cấu hình — tùy chỉnh ở đây
    # ==========================================================
    SCENARIOS = [
        # (n_normal, n_anomaly, n_features, n_normal_clusters, difficulty, noise_pct, label)
        (2000, 2000, 20, 5, 'easy',             0.0, "1_EASY_separated"),   
        (2000, 2000, 20, 5, 'hard_overlap',     0.0, "2_HARD_overlap_edge"),  
        (2000, 2000, 20, 5, 'hard_anisotropic', 0.0, "3_HARD_anisotropic"),     
        (2000, 2000, 20, 5, 'extreme_interp',   0.0, "4_EXTREME_interpolated_anom"),
        (2000, 2000, 20, 5, 'extreme_interp',   3.0, "5_EXTREME_interp_with_3pct_noise"), 
    ]

    # n_clusters sweep nhỏ — chỉ để test nhanh
    N_CLUSTERS_LIST = list(range(1, 11, 1))   # [1..10]

    best_results = []

    for (n_normal, n_anomaly, n_features, n_normal_clusters, difficulty, noise_pct, label) in SCENARIOS:
        log.info("\n" + "-" * 60)
        log.info(f"SCENARIO: {label}")
        log.info("-" * 60)

        X_train, y_train, X_test, y_test = generate_balanced_synthetic(
            n_normal=n_normal, n_anomaly=n_anomaly,
            n_features=n_features,
            n_normal_clusters=n_normal_clusters,
            difficulty=difficulty,
            noise_pct=noise_pct, seed=42,
        )

        results = run_experiment_synthetic(
            X_train, y_train, X_test, y_test,
            n_clusters_list=N_CLUSTERS_LIST,
            label=label,
            out_path=out_path,
            logger=log,
        )

        if results:
            # Best n_cluster cho scenario này (theo AUC-ROC)
            best = max(results, key=lambda r: r["AUCROC"])
            best_results.append(best)
            verdict = "✅ PASS" if best["AUCROC"] > 70 else "❌ FAIL"
            log.info(f"{verdict}  Best AUC-ROC={best['AUCROC']:.2f}%  "
                     f"@ nCluster={best['nCluster']}")

    # ==========================================================
    # Tổng kết
    # ==========================================================
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for r in best_results:
        verdict = "✅ PASS" if r["AUCROC"] > 70 else "❌ FAIL"
        log.info(f"{verdict}  {r['label']:<35}  "
                 f"AUC-ROC={r['AUCROC']:.2f}%  AUC-PR={r['AUCPR']:.2f}%  "
                 f"F1={r['F1 Score']:.4f}  MCC={r['MCC']:.4f}")

    n_pass = sum(1 for r in best_results if r["AUCROC"] > 70)
    log.info(f"\n→ {n_pass}/{len(best_results)} scenarios PASSED")
    log.info(f"Results saved to: {out_path}")
    log.info(f"Log saved to    : {log_path}")


