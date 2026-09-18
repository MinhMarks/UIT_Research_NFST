"""
FL-LOC-NFST Centralized Baseline Runner
=========================================
Runs the centralized LOC-NFST (same algorithm, no partitioning)
for apples-to-apples comparison with FL results.

Uses the existing OC_NFST_memory_optimized_simple_scoring.py pipeline
with fixed K=5 clusters (matching FL config).

Usage:
    python run_centralized.py --dataset data_CICIoT2023 --scaler StandardScaler

    DATA_DIR=/path/to/data python run_centralized.py \
        --dataset data_CICIoT2023 --scaler StandardScaler --clusters 5
"""
import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import null_space, eigh
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ── Ensure parent experiments folder is on path ──────────────────────────────
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))

from fed_loc_nfst.config import (
    DATA_DIR, OUTPUT_DIR, K_CLUSTERS, NOISE_PCT, EPSILON_SVD,
    EPSILON_NEAR_NULL, L_MIN, SEED
)
from fed_loc_nfst.data_utils import load_dataset
from fed_loc_nfst.evaluate import compute_fl_scores, evaluate_scores


# ============================================================
# Centralized NPD Computation (mirrors OC_NFST_memory_optimized_simple_scoring)
# ============================================================

META_COLS = [
    'pkSeqID', 'stime', 'ltime', 'seq', 'saddr', 'daddr', 'sport', 'dport',
    'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'state', 'flgs', 'proto'
]


def compute_centralized_NPD(X_train: np.ndarray, K: int,
                             epsilon_svd: float = EPSILON_SVD,
                             epsilon_near_null: float = EPSILON_NEAR_NULL,
                             L_min: int = L_MIN):
    """
    Centralized NPD computation matching the federated version's
    adaptive spectral solve (for fair comparison).

    Uses eigh (symmetric eigen-decomp) instead of scipy null_space
    to enable near-null relaxation on the same code path.
    """
    N, d = X_train.shape
    K = min(K, N)

    # --- K-Means clustering ---
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    y = km.fit_predict(X_train)
    classes = np.unique(y)
    centroids = km.cluster_centers_.astype(np.float64)

    # --- S_w: within-class scatter ---
    S_w = np.zeros((d, d), dtype=np.float64)
    for cls in classes:
        X_cls = X_train[y == cls].astype(np.float64)
        mu_cls = centroids[cls]
        diff = X_cls - mu_cls
        S_w += diff.T @ diff
    S_w /= N

    # --- S_t: total scatter ---
    mu_total = np.mean(X_train, axis=0).astype(np.float64)
    diff_t = X_train.astype(np.float64) - mu_total
    S_t = (diff_t.T @ diff_t) / N

    # --- SVD rank detection on S_t ---
    eigvals_t, eigvecs_t = eigh(S_t)
    eigvals_t = np.maximum(eigvals_t, 0)
    rank_Pt = int(np.sum(eigvals_t > epsilon_svd))
    if rank_Pt == 0:
        rank_Pt = min(L_min, d)

    Q = eigvecs_t[:, -rank_Pt:].astype(np.float64)

    # --- Null space in projected subspace ---
    A = Q.T @ S_w @ Q

    B = null_space(A, rcond=epsilon_svd)
    L = B.shape[1]

    if L < L_min:
        # Near-null fallback
        eigvals_A, eigvecs_A = eigh(A)
        eigvals_A = np.maximum(eigvals_A, 0)
        near_null_mask = eigvals_A < epsilon_near_null
        if not np.any(near_null_mask):
            near_null_idx = np.argsort(eigvals_A)[:L_min]
        else:
            near_null_idx = np.where(near_null_mask)[0]
        B = eigvecs_A[:, near_null_idx]
        L = B.shape[1]

    W = (Q @ B).astype(np.float32)

    # --- Null centers ---
    null_train = (X_train.astype(np.float32)) @ W
    null_centers = np.array([
        np.mean(null_train[y == cls], axis=0)
        for cls in classes
    ], dtype=np.float32)

    # max_train
    from fed_loc_nfst.evaluate import min_dist_to_centers
    train_dists = min_dist_to_centers(null_train, null_centers)
    max_train = float(np.max(train_dists)) if len(train_dists) > 0 else 1.0
    if max_train < 1e-10:
        max_train = 1.0

    return W, null_centers, max_train, L, K


# ============================================================
# Centralized Experiment Runner
# ============================================================

def run_centralized_experiment(
    dataset_prefix: str,
    scaler: str,
    K: int = K_CLUSTERS,
    noise_pct: float = NOISE_PCT,
    logger: logging.Logger = None,
) -> dict:
    log = logger or logging.getLogger("centralized")

    train_path = os.path.join(DATA_DIR, f"Train_{scaler}_{dataset_prefix}.csv")
    test_path  = os.path.join(DATA_DIR, f"Test_{scaler}_{dataset_prefix}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log.warning(f"[SKIP] Files not found: {train_path}")
        return {"error": f"Not found: {train_path}"}

    log.info("=" * 60)
    log.info(f"Centralized Baseline: {dataset_prefix} | scaler={scaler} | K={K}")
    log.info("=" * 60)

    np.random.seed(SEED)
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path, noise_pct)
    d = X_train.shape[1]

    log.info(f"Dataset: N_train={len(X_train)}, N_test={len(X_test)}, d={d}")

    # --- Train ---
    t_train = time.time()
    W, null_centers, max_train, L, K_actual = compute_centralized_NPD(X_train, K)
    train_time = time.time() - t_train

    log.info(f"Centralized NPD: W={W.shape}, L={L}, K={K_actual}, "
             f"max_train={max_train:.6f}, time={train_time:.3f}s")

    # --- Evaluate ---
    t_infer = time.time()
    y_proba = compute_fl_scores(X_test, W, null_centers, max_train)
    infer_time = time.time() - t_infer

    metrics = evaluate_scores(y_test, y_proba)
    log.info(
        f"Centralized Metrics: F1={metrics['F1 Score']:.4f}, "
        f"AUC-ROC={metrics['AUCROC']:.2f}%, MCC={metrics['MCC']:.4f}"
    )

    return {
        "mode": "centralized",
        "dataset": dataset_prefix,
        "scaler": scaler,
        "K_clusters": K_actual,
        "noise_pct": noise_pct,
        "N_train": len(X_train),
        "N_test": len(X_test),
        "d_features": d,
        "L_null_dim": L,
        "train_time_s": round(train_time, 3),
        "infer_time_s": round(infer_time, 3),
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Centralized LOC-NFST Baseline")
    parser.add_argument('--dataset', type=str, default='data_CICIoT2023')
    parser.add_argument('--scaler', type=str, default='StandardScaler')
    parser.add_argument('--clusters', type=int, default=K_CLUSTERS)
    parser.add_argument('--noise', type=float, default=NOISE_PCT)
    parser.add_argument('--all-datasets', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Logger
    log_path = os.path.join(OUTPUT_DIR, f"centralized_{timestamp}.log")
    logger = logging.getLogger("centralized")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.addHandler(logging.FileHandler(log_path, encoding='utf-8'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(fmt)

    if args.all_datasets:
        from fed_loc_nfst.config import DATASETS, SCALERS
        datasets, scalers = DATASETS, SCALERS
    else:
        datasets, scalers = [args.dataset], [args.scaler]

    all_results = []
    for dataset in datasets:
        for scaler in scalers:
            result = run_centralized_experiment(
                dataset, scaler, K=args.clusters,
                noise_pct=args.noise, logger=logger
            )
            if "error" not in result:
                all_results.append(result)

    if all_results:
        out_csv = os.path.join(OUTPUT_DIR, f"centralized_results_{timestamp}.csv")
        pd.DataFrame(all_results).to_csv(out_csv, index=False)
        print(f"\n✓ Centralized results saved to: {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
