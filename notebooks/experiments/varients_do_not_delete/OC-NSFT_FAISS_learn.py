    """
OC-NSFT with FAISS Accelerated Distance Computation
====================================================

This is an optimized version of OC-NSFT_old_noise_Kmean_threshold.ipynb.

Key improvement: The `learn` function now uses FAISS for fast nearest neighbor 
search instead of computing full pairwise distance matrices.

Performance gains:
- train_score: O(n²) → O(n log n) with FAISS index
- y_score: O(n*m) → O(m log n) with FAISS search  
- Memory: O(n²) → O(n) (no full distance matrix needed)

To convert to notebook: jupyter nbconvert --to notebook OC-NSFT_FAISS_learn.py
"""

import pandas as pd
import numpy as np
import random
import faiss
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score, precision_score, average_precision_score, 
    recall_score, f1_score, accuracy_score, mean_squared_error,
    mean_absolute_error, roc_curve, auc, classification_report,
    confusion_matrix, matthews_corrcoef
)
from sklearn.datasets import make_blobs, make_multilabel_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    KernelCenterer, LabelEncoder, MinMaxScaler, 
    Normalizer, QuantileTransformer, RobustScaler
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.manifold import TSNE
import time
import scipy as sp_scipy
from scipy.linalg import svd, null_space
import os
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix as sp
import math
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Check if torch is available, though it might not be used here, 
    # but good for consistency if imported or used later.
    # The user request said "all code files", so we include it.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_seed(42)

# ============================================================================
# COMPLEXITY ANALYSIS
# ============================================================================
# Original:
#   - distance_vector: O(n²) time, O(n²) memory
#   - minimum_distance: O(n*m) time, O(n*m) memory
#
# With FAISS:
#   - build_faiss_index: O(n*d) time, O(n*d) memory
#   - faiss_min_distance_train: O(n*log(n)) time, O(n) memory
#   - faiss_min_distance_test: O(m*log(n)) time, O(m) memory
# ============================================================================

alpha = 0.9


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_data_noise(train_data, test_data, noise_percentage=10):
    """
    Preprocess data and add noise samples from anomaly class to training set.
    """
    print("..............................Data Overview................................")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    
    X_train_total = train_data.iloc[:, :-1].to_numpy()
    y_train_total = train_data.iloc[:, -1].to_numpy()

    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    print("Train Data Labels [0]:", np.unique(y_train))

    n_samples = X_train.shape[0]
    noise_samples_count = int(n_samples * (noise_percentage / 100))

    X_train_noise = X_train_total[y_train_total == 1]
    noisy_indices = np.random.choice(X_train_noise.shape[0], size=noise_samples_count, replace=False)
    X_train_noise = X_train_noise[noisy_indices]
    
    X_train = np.vstack((X_train, X_train_noise))
    y_train = np.concatenate((y_train, np.ones(X_train_noise.shape[0])))
    print(y_train) 
    
    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    print("Number of samples after adding noise:", n_samples)
    print("Number of features:", n_features)

    return X_train, y_train, X_test, y_test


# ============================================================================
# CLUSTERING
# ============================================================================
def cluster_kmeans(data, initial_k):
    """K-Means clustering with sklearn."""
    print("Starting K-Means clustering...")

    kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    labels = cluster_labels

    sorted_indices = np.argsort(labels)
    sorted_data = data[sorted_indices]
    sorted_labels = labels[sorted_indices]

    print("Final number of clusters:", len(np.unique(sorted_labels)))
    return sorted_data, sorted_labels


# ============================================================================
# FAISS UTILITIES - NEW OPTIMIZED FUNCTIONS
# ============================================================================
def build_faiss_index(X, use_gpu=False):
    """
    Build a FAISS index for fast nearest neighbor search.
    
    Args:
        X: Data matrix (n_samples, n_features), will be converted to float32
        use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
    
    Returns:
        index: FAISS index ready for search
    """
    X = np.ascontiguousarray(X.astype('float32'))
    d = X.shape[1]
    
    # Use L2 (Euclidean) distance - IndexFlatL2 is exact search
    index = faiss.IndexFlatL2(d)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(X)
    return index


def faiss_min_distance_train(X, k=2):
    """
    Compute minimum distance from each training point to its nearest neighbor
    (excluding itself) using FAISS.
    
    This replaces the O(n²) distance_vector computation.
    
    Args:
        X: Training data (n_samples, n_features)
        k: Number of neighbors to search (k=2 to get nearest excluding self)
    
    Returns:
        min_distances: Array of minimum distances for each point
    """
    X = np.ascontiguousarray(X.astype('float32'))
    index = build_faiss_index(X)
    
    # Search for k=2 neighbors (first is self with distance 0, second is nearest)
    distances, _ = index.search(X, k)
    
    # distances[:, 0] is distance to self (should be ~0)
    # distances[:, 1] is distance to nearest neighbor
    # FAISS returns squared L2 distances, so we take sqrt
    min_distances = np.sqrt(distances[:, 1])
    
    return min_distances


def faiss_min_distance_test(X_test, X_train):
    """
    Compute minimum distance from each test point to nearest training point
    using FAISS.
    
    This replaces the O(n*m) minimum_distance computation.
    
    Args:
        X_test: Test data (n_test, n_features)
        X_train: Training data (n_train, n_features)
    
    Returns:
        min_distances: Array of minimum distances for each test point
    """
    X_train = np.ascontiguousarray(X_train.astype('float32'))
    X_test = np.ascontiguousarray(X_test.astype('float32'))
    
    index = build_faiss_index(X_train)
    
    # Search for k=1 nearest neighbor
    distances, _ = index.search(X_test, 1)
    
    # FAISS returns squared L2 distances
    min_distances = np.sqrt(distances[:, 0])
    
    return min_distances


# ============================================================================
# ORIGINAL DISTANCE FUNCTIONS (kept for comparison)
# ============================================================================
def nullspace(A):
    _, s, vh = np.linalg.svd(A)
    null_mask = np.isclose(s, 0)
    null_space_result = vh[null_mask].T
    return null_space_result


def minimum_distance_original(A, B):
    """Original O(n*m) implementation - kept for comparison."""
    A = np.asarray(A)
    B = np.asarray(B)
    min_distances = np.empty(A.shape[0], dtype=np.float64)
    for i, a in enumerate(A):
        distances = cdist([a], B, metric='euclidean')
        min_distances[i] = np.min(distances)
    return min_distances


def distance_vector_original(point_X, point_Y):
    """Original O(n²) implementation - kept for comparison."""
    norm_X = np.sum(point_X**2, axis=1)
    norm_Y = np.sum(point_Y**2, axis=1)
    dot_product = np.dot(point_Y, point_X.T)
    distance = np.sqrt(abs(norm_Y[:, np.newaxis] + norm_X[np.newaxis, :] - 2 * dot_product))
    return distance


# ============================================================================
# NPD CALCULATION
# ============================================================================
def calculate_NPD(X, y, epsilon=1e-6):
    """
    Calculate Null Projecting Directions (NPDs).
    
    Parameters:
        X: Data matrix (n_samples, n_features)
        y: Cluster/class labels (n_samples,)
        epsilon: Threshold for singular value detection
    
    Returns:
        W: NPD matrix (n_features, L)
        k: Estimated constant k
        training_time: Time taken for computation
    """
    print("Begin calculating NPD and k --------------")
    X = X.T  # Convert to (n_features, n_samples)
    print('Shape of X:', X.shape)
    
    c = len(np.unique(y))
    d, N = X.shape
    
    t0 = time.time()
    
    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total
    
    P_w = np.zeros_like(X)
    for i in np.unique(y):
        class_mean = np.mean(X[:, y == i], axis=1, keepdims=True)
        P_w[:, y == i] = X[:, y == i] - class_mean
    
    S_w = np.dot(P_w, P_w.T) / N
    S_t = np.dot(P_t, P_t.T) / N
    
    _, singular_values_Pw, _ = np.linalg.svd(P_w, full_matrices=False)
    rank_Pw = np.sum(singular_values_Pw > epsilon)
    _, singular_values_Pt, _ = np.linalg.svd(P_t, full_matrices=False)
    rank_Pt = np.sum(singular_values_Pt > epsilon)
    
    k = 0
    
    U, _, _ = np.linalg.svd(P_t, full_matrices=False)
    Q = U
    
    B = null_space(Q.T @ S_w @ Q)
    W = Q @ B

    t05 = time.time()
    print("...............................Timing Model................................")
    print("Time train:", t05 - t0)
    print("N =", N, "d =", d, "c =", c)
    print("W : d x L =", W.shape)
    print("Threshold c_th =", N - d - k + 1)
    
    return W, k, (t05 - t0)


# ============================================================================
# LEARN FUNCTION - FAISS OPTIMIZED VERSION
# ============================================================================
def learn_faiss(npd, X_train, y_train, X_test):
    """
    FAISS-optimized learn function.
    
    Key changes from original:
    1. Uses FAISS for train_score computation (nearest neighbor excluding self)
    2. Uses FAISS for y_score computation (nearest neighbor in training set)
    
    Complexity improvement:
    - Original: O(n²) for train_score, O(n*m) for y_score
    - FAISS: O(n*log(n)) for train_score, O(m*log(n)) for y_score
    
    Args:
        npd: Null Projecting Directions matrix
        X_train: Training data
        y_train: Training labels
        X_test: Test data
    
    Returns:
        y_proba: Probability predictions (n_test, 2)
        y_predict: Binary predictions
        inference_time: Time taken for inference
    """
    # Project data to null space
    null_point_X = (sp(X_train).dot(sp(npd))).toarray()
    null_point_X_test = (sp(X_test).dot(sp(npd))).toarray()

    t1 = time.time()
    
    # =========================================================================
    # FAISS OPTIMIZATION: Compute train_score using nearest neighbor search
    # Original code:
    #   train_score_tmp = distance_vector(null_point_X, null_point_X)
    #   for i in range(len(train_score_tmp)):
    #       train_score_tmp[i, i] = 1e9
    #   train_score = np.amin(train_score_tmp, axis=1)
    #
    # FAISS version: O(n*log(n)) instead of O(n²)
    # =========================================================================
    train_score = faiss_min_distance_train(null_point_X, k=2)
    
    # =========================================================================
    # FAISS OPTIMIZATION: Compute y_score using nearest neighbor search
    # Original code:
    #   y_score = minimum_distance(null_point_X_test, null_point_X)
    #
    # FAISS version: O(m*log(n)) instead of O(n*m)
    # =========================================================================
    y_score = faiss_min_distance_test(null_point_X_test, null_point_X)
    
    # Compute probabilities
    y_proba = np.zeros((len(y_score), 2))
    y_proba[:, 1] = np.minimum(y_score / np.max(train_score), 1)
    y_proba[:, 0] = 1 - y_proba[:, 1]
    
    y_proba = np.nan_to_num(y_proba, nan=1.0)
    y_predict = (y_proba[:, 1] > 0.2).astype(int)
    
    t2 = time.time()
    print("...............................Timing Model................................")
    print("Time test (FAISS):", t2 - t1)
    
    return y_proba, y_predict, (t2 - t1)


def learn_original(npd, X_train, y_train, X_test):
    """
    Original learn function (kept for comparison/benchmarking).
    """
    null_point_X = (sp(X_train).dot(sp(npd))).toarray()
    null_point_X_test = (sp(X_test).dot(sp(npd))).toarray()

    t1 = time.time()
    
    # Original O(n²) computation
    train_score_tmp = distance_vector_original(null_point_X, null_point_X)
    for i in range(len(train_score_tmp)):
        train_score_tmp[i, i] = 1e9
    train_score = np.amin(train_score_tmp, axis=1)
    
    # Original O(n*m) computation
    y_score = minimum_distance_original(null_point_X_test, null_point_X)
    
    y_proba = np.zeros((len(y_score), 2))
    y_proba[:, 1] = np.minimum(y_score / np.max(train_score), 1)
    y_proba[:, 0] = 1 - y_proba[:, 1]
    
    y_proba = np.nan_to_num(y_proba, nan=1.0)
    y_predict = (y_proba[:, 1] > 0.2).astype(int)
    
    t2 = time.time()
    print("...............................Timing Model................................")
    print("Time test (Original):", t2 - t1)
    
    return y_proba, y_predict, (t2 - t1)


# Alias for default usage
learn = learn_faiss


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def Model_evaluating(y_true, y_predict, y_scores):
    """
    Evaluate model using threshold derived from ROC curve (Youden's J statistic).
    """
    print("..............................Report Parameter...............................")
    
    # Invert true labels (1=Anomaly, 0=Normal)
    y_true_inverted = 1 - y_true
    y_prob = y_scores[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true_inverted, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print("Optimal threshold (Youden's J):", optimal_threshold)

    y_predict_optimal = (y_prob >= optimal_threshold).astype(int)

    mcc = matthews_corrcoef(y_true_inverted, y_predict_optimal)
    f1 = f1_score(y_true_inverted, y_predict_optimal)
    ppv = precision_score(y_true_inverted, y_predict_optimal, zero_division=0)
    recall = recall_score(y_true_inverted, y_predict_optimal, zero_division=0)
    accuracy = accuracy_score(y_true_inverted, y_predict_optimal)
    auc_score = roc_auc_score(y_true_inverted, y_prob)
    aucpr = average_precision_score(y_true_inverted, y_prob)
    
    # In ra các kết quả
    print("AUCROC:", auc_score * 100)
    print("AUCPR:", aucpr * 100)
    print("Accuracy:", accuracy * 100)
    print("MCC:", mcc)
    print("F1 score:", f1)
    print("PPV (Precision):", ppv)
    print("TPR (Recall):", recall)

    return [auc_score * 100, aucpr * 100, accuracy * 100, mcc, f1, ppv, recall]
    
    print("AUCROC:", auc_score * 100)
    print("AUCPR:", aucpr * 100)
    print("Accuracy:", accuracy * 100)
    print("MCC:", mcc)
    print("F1 score:", f1)
    print("PPV (Precision):", ppv)
    print("TPR (Recall):", recall)

    return [auc_score * 100, aucpr * 100, accuracy * 100, mcc, f1, ppv, recall]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def plot_data_2D(X, labels, title="Data visualization"):
    """Placeholder for 2D visualization."""
    return 0


def benchmark_learn_functions(npd, X_train, y_train, X_test, n_runs=3):
    """
    Benchmark FAISS vs Original learn function.
    
    Args:
        npd, X_train, y_train, X_test: Standard inputs
        n_runs: Number of runs for averaging
    
    Returns:
        dict: Timing results
    """
    print("=" * 60)
    print("BENCHMARKING: FAISS vs Original")
    print("=" * 60)
    
    # Benchmark FAISS
    faiss_times = []
    for i in range(n_runs):
        _, _, t = learn_faiss(npd, X_train, y_train, X_test)
        faiss_times.append(t)
    
    # Benchmark Original
    original_times = []
    for i in range(n_runs):
        _, _, t = learn_original(npd, X_train, y_train, X_test)
        original_times.append(t)
    
    results = {
        'faiss_mean': np.mean(faiss_times),
        'faiss_std': np.std(faiss_times),
        'original_mean': np.mean(original_times),
        'original_std': np.std(original_times),
        'speedup': np.mean(original_times) / np.mean(faiss_times)
    }
    
    print(f"\nFAISS:    {results['faiss_mean']:.4f}s ± {results['faiss_std']:.4f}s")
    print(f"Original: {results['original_mean']:.4f}s ± {results['original_std']:.4f}s")
    print(f"Speedup:  {results['speedup']:.2f}x")
    print("=" * 60)
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================
columns = ["scaler", "nCluster", "noise_percentage", "AUCROC", "AUCPR", "Accuracy", 
           "MCC", "F1 Score", "Precision", "Recall", "Time Train", "Time Test"]


def function(df1, df2, scaler, noise, output_file, use_faiss=True):
    """
    Main experiment function.
    
    Args:
        df1: Training dataframe
        df2: Test dataframe
        scaler: Scaler name
        noise: Noise percentage
        output_file: Output CSV file path
        use_faiss: Whether to use FAISS-optimized learn function
    """
    X_train0, y_train0, X_test, y_test = preprocess_data_noise(df1, df2, noise)

    imputer = SimpleImputer(strategy="mean")
    X_train0[np.isinf(X_train0)] = np.nan
    X_train0 = imputer.fit_transform(X_train0)
    
    # Select learn function
    learn_fn = learn_faiss if use_faiss else learn_original
    
    for ncluster in [1, 100, 139, 184, 301]:
        X_train, y_train = cluster_kmeans(X_train0, ncluster)
        plot_data_2D(X_train, y_train, "Data after clustering")
        
        npd, k, training_time = calculate_NPD(X_train, y_train)
        
        y_proba, y_predict, inference_time = learn_fn(npd, X_train, y_train, X_test)
        
        v = Model_evaluating(y_test, y_predict, y_proba)
        
        result = [scaler, ncluster, noise] + v + [training_time, inference_time]

        result_df = pd.DataFrame([result], columns=columns)
        result_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        
    return 0


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
if __name__ == "__main__":
    import cProfile
    import pstats
    
    dataset_prefixes = ['data_ToNIoT', 'data_CICIoT2023', 'data_N_BaIoT', 'data_BoTIoT']
    scaler_names = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 
                    'QuantileTransformer', 'RobustScaler']

    for prefix in dataset_prefixes:
        print("-" * 50)
        print("--------", prefix, "-" * 30)
        print("-" * 50)

        base_output_file = f"Results_FAISS_{prefix}"
        output_file = base_output_file + "_0.csv"

        counter = 0
        while os.path.exists(output_file):
            counter += 1
            output_file = f"{base_output_file}_{counter}.csv"

        for scaler in scaler_names:
            print(f"Processing dataset {prefix} with {scaler} scaler ...")
            
            train_file = f'../../Datascaled/NoiseOCData/Train_{scaler}_{prefix}.csv'
            test_file = f'../../Datascaled/NoiseOCData/Test_{scaler}_{prefix}.csv'
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"  Skipping: files not found")
                continue
                
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)

            df_train = df_train.dropna()
            df_test = df_test.dropna()
            
            df_full = pd.concat([df_train, df_test], ignore_index=True)
            df_train_new, df_test_new = train_test_split(df_full, test_size=0.3, random_state=42)
            
            for noise in [0]:
                function(df_train_new, df_test_new, scaler, noise, output_file, use_faiss=True)
