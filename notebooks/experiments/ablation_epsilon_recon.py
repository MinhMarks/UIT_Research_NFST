import os
import sys
import time
import tracemalloc
import random
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not installed. Falling back to numpy scoring.")

# ============================================================================
# Core Functions
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
set_seed(42)

def preprocess_data_noise(df_train, df_test, noise_percentage=1):
    X_train = df_train.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = df_train.iloc[:, -1].to_numpy()
    
    X_test = df_test.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = df_test.iloc[:, -1].to_numpy()
    
    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_percentage / 100))
    
    if noise_count > 0:
        anom_idx = np.where(y_test == 1)[0]
        if len(anom_idx) > 0:
            chosen_anom_idx = np.random.choice(anom_idx, size=min(noise_count, len(anom_idx)), replace=False)
            X_noise = X_test[chosen_anom_idx]
            
            # Khử nhiễu rò rỉ (leakage) ở tập Test
            mask = np.ones(len(y_test), dtype=bool)
            mask[chosen_anom_idx] = False
            X_test, y_test = X_test[mask], y_test[mask]
            
            # Inject vào Train (nhưng dán nhãn là 0 để NFST vẫn gom cụm mù)
            X_train = np.vstack((X_train, X_noise))
            y_train = np.concatenate((y_train, np.zeros(len(X_noise))))
            
    return X_train, y_train, X_test, y_test

def drop_metadata_features(df):
    meta_cols = ['pkSeqID', 'stime', 'ltime', 'seq', 'saddr', 'daddr', 'sport', 'dport', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'state', 'flgs', 'proto']
    for col in meta_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

def cluster_kmeans(data: np.ndarray, k: int):
    k = min(k, len(data))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    sorted_idx = np.argsort(labels)
    return data[sorted_idx], labels[sorted_idx], kmeans.cluster_centers_.astype(np.float32)

def calculate_NPD_optimized_ablation(X: np.ndarray, y: np.ndarray, epsilon: float):
    t0 = time.time()
    X = np.ascontiguousarray(X.T, dtype=np.float32)   # (d, N)
    d, N = X.shape
    classes = np.unique(y)

    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total

    U, s_t, _ = np.linalg.svd(P_t, full_matrices=False)
    
    # Ablation: apply epsilon
    rank_Pt = int(np.sum(s_t > epsilon))
    
    if rank_Pt == 0:
        return None, None, 0.0, 0
        
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
    
    # Returns both W (Null space) and Q (Normal subspace basis) for Recon error calculation
    return W, Q, training_time, rank_Pt

def compute_dist_to_centers(X, W, Q, centers, alpha=1.0):
    if FAISS_AVAILABLE:
        d = centers.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(centers.astype('float32'))
        _, nearest_idx = index.search(X.astype('float32'), 1)
        nearest_centers = centers[nearest_idx.flatten()]
    else:
        # Fallback to numpy block-search
        nearest_centers = []
        for i in range(0, len(X), 2000):
            batch = X[i:i+2000]
            dists = np.linalg.norm(batch[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            idx = np.argmin(dists, axis=1)
            nearest_centers.append(centers[idx])
        nearest_centers = np.vstack(nearest_centers)

    diff = X - nearest_centers
    
    # Distance IN the null space
    projections = diff @ W
    dist_null = np.sum(projections**2, axis=1)
    
    # Distance OUTSIDE the Normal Subspace (Reconstruction Error)
    recon = diff - (diff @ Q) @ Q.T
    dist_ortho = np.sum(recon**2, axis=1)
    
    return np.sqrt(dist_null + alpha * dist_ortho)

def compute_scores(X_train, X_test, W, Q, centers):
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test  = np.ascontiguousarray(X_test,  dtype=np.float32)

    # train_score is technically calculated but not used to mask test scores in recon!
    train_score = compute_dist_to_centers(X_train, W, Q, centers)
    y_score     = compute_dist_to_centers(X_test, W, Q, centers)

    score_max = np.max(y_score) + 1e-10
    
    y_proba = np.zeros((len(y_score), 2), dtype=np.float32)
    y_proba[:, 1] = y_score / score_max
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    y_proba = np.nan_to_num(y_proba, nan=1.0)
    return y_proba

def evaluate(y_true: np.ndarray, y_proba: np.ndarray):
    y_true_flipped = (1 - y_true).astype(int)
    y_prob = y_proba[:, 0]
    auc_roc  = roc_auc_score(y_true_flipped, y_prob)
    auc_pr   = average_precision_score(y_true_flipped, y_prob)
    
    fpr, tpr, thresholds = roc_curve(y_true_flipped, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thr = thresholds[optimal_idx]
    y_pred_opt = (y_prob >= optimal_thr).astype(int)
    f1 = f1_score(y_true_flipped, y_pred_opt, zero_division=0)
    
    return auc_roc * 100, auc_pr * 100, f1 * 100

# ============================================================================
# MAIN
# ============================================================================
def main():
    # User's configurations as requested
    DATASET_CONFIGS = {
        'data_CICIoT2023': 'StandardScaler',
        'data_ToNIoT': 'RobustScaler',
        'data_N_BaIoT': 'Normalizer',
        'data_BoTIoT': 'StandardScaler',
        'data_EdgeIIoTset': 'MinMaxScaler',
        'data_IoTID20': 'StandardScaler',
    }
    EPSILON_LIST = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 2e-6, 4e-6, 1e-8, 1e-10, 0.0]
    K = 120

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_data = os.path.normpath(
        os.path.join(_script_dir, '..', '..', 'Datascaled', 'Official_OC_Data')
    )
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)

    out_path = "outputs/ablation_epsilon_recon_results.csv"
    if os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs("outputs", exist_ok=True)
    
    results = []
    
    for prefix, scaler in DATASET_CONFIGS.items():
        print(f"\n==============================")
        print(f"Dataset: {prefix} (Scaler: {scaler})")
        print(f"==============================")
        
        train_path = os.path.join(DATA_DIR, f"Train_{scaler}_{prefix}.csv")
        test_path  = os.path.join(DATA_DIR, f"Test_{scaler}_{prefix}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"[SKIP] Files not found: {train_path}")
            continue

        df_train = pd.read_csv(train_path).dropna()
        df_test  = pd.read_csv(test_path).dropna()
        
        df_train = drop_metadata_features(df_train)
        df_test = drop_metadata_features(df_test)
        
        # Testing robustness strictly without noise contamination
        X_train, y_train, X_test, y_test = preprocess_data_noise(df_train, df_test, noise_percentage=0)
        
        # Impute
        X_train[np.isinf(X_train)] = np.nan
        X_test[np.isinf(X_test)] = np.nan
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train).astype(np.float32)
        X_test = imputer.transform(X_test).astype(np.float32)
        
        print(f"Clustering with K={K}...")
        X_clustered, y_clustered, centers = cluster_kmeans(X_train, K)
        
        for eps in EPSILON_LIST:
            print(f"  -> Testing epsilon = {eps}")
            try:
                # Track mem & train time inside calculating NPD
                tracemalloc.start()
                
                # Recon function natively exports Q matrix
                W, Q, train_time, rank_Pt = calculate_NPD_optimized_ablation(X_clustered, y_clustered, eps)
                
                _, peak_bytes_train = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                if W is None:
                    print("     [Error] Matrix collapsed (rank=0). Skipping.")
                    continue
                
                # compute_scores requires Q for Recon-error logic
                y_proba = compute_scores(X_train, X_test, W, Q, centers)
                auc_roc, auc_pr, f1 = evaluate(y_test, y_proba)
                
                res = {
                    "Dataset": prefix.replace("data_", "").replace(".csv", ""),
                    "Epsilon": eps,
                    "Rank_Pt": rank_Pt,
                    "AUC-ROC": round(auc_roc, 4),
                    "AUC-PR": round(auc_pr, 4),
                    "F1 Score": round(f1, 4),
                    "Train Time (s)": round(train_time, 4),
                    "Peak RAM Train (MB)": round(peak_bytes_train / 1e6, 3)
                }
                
                results.append(res)
                
                # Append to CSV instantly
                df_temp = pd.DataFrame([res])
                df_temp.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
                
            except Exception as e:
                print(f"     [Exception] Failed for epsilon {eps}: {e}")

    print(f"\nSaved all results to {out_path}")

if __name__ == "__main__":
    main()
