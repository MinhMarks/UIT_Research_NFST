import sys
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score, 
                             recall_score, accuracy_score, roc_auc_score, 
                             average_precision_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from scipy.linalg import null_space

def set_seed(seed=42):
    np.random.seed(seed)
set_seed()

# ----------------- Core Methods -----------------
def preprocess_data_noise(df_train, df_test, noise_percentage=1):
    X_train_total = df_train.iloc[:, :-1].to_numpy()
    y_train_total = df_train.iloc[:, -1].to_numpy()
    
    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]
    
    n_samples = X_train.shape[0]
    noise_samples_count = int(n_samples * (noise_percentage / 100))
    X_train_noise = X_train_total[y_train_total == 1]
    
    if noise_samples_count > 0 and len(X_train_noise) > 0:
        noisy_indices = np.random.choice(X_train_noise.shape[0], size=min(noise_samples_count, len(X_train_noise)), replace=False)
        X_train_noise = X_train_noise[noisy_indices]
        X_train = np.vstack((X_train, X_train_noise))
        y_train = np.concatenate((y_train, np.ones(X_train_noise.shape[0])))
        
    X_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()
    return X_train, y_train, X_test, y_test

def cluster_kmeans(data, initial_k):
    kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    sorted_indices = np.argsort(labels)
    return data[sorted_indices], labels[sorted_indices], kmeans.cluster_centers_

def calculate_NPD_svd(X, y, epsilon=1e-6):
    """The new proposed NPD calculation using Singular Value Decomposition for Stability."""
    X = X.T  # d x N
    d, N = X.shape
    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total
    
    P_w = np.zeros_like(X)
    for i in np.unique(y):
        class_mean = np.mean(X[:, y == i], axis=1, keepdims=True)
        P_w[:, y == i] = X[:, y == i] - class_mean
        
    S_w = np.dot(P_w, P_w.T) / N
    
    # --- SVD APPROACH ---
    U, _, _ = np.linalg.svd(P_t, full_matrices=False)
    Q = U
    
    B = null_space(Q.T @ S_w @ Q)
    W = Q @ B
    return W

def gram_schmidt_basis(X, tol=1e-6):
    """Classical Gram-Schmidt Orthogonalization (Simulating older variant)"""
    basis = []
    for i in range(X.shape[1]):
        v = X[:, i]
        for b in basis:
            v = v - np.dot(b, v) * b
        norm = np.linalg.norm(v)
        if norm > tol:
            basis.append(v / norm)
            if len(basis) == X.shape[0]: 
                break
    if len(basis) == 0:
        return np.zeros((X.shape[0], 1))
    return np.column_stack(basis)

def calculate_NPD_gs(X, y, epsilon=1e-6):
    """The old NPD calculation using Gram-Schmidt Orthogonalization."""
    X = X.T  # d x N
    d, N = X.shape
    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total
    
    P_w = np.zeros_like(X)
    for i in np.unique(y):
        class_mean = np.mean(X[:, y == i], axis=1, keepdims=True)
        P_w[:, y == i] = X[:, y == i] - class_mean
        
    S_w = np.dot(P_w, P_w.T) / N
    
    # --- GRAM-SCHMIDT APPROACH ---
    # Due to full matrix processing, this is slower and less stable to rank-deficiency
    Q = gram_schmidt_basis(P_t)
    
    try:
        B = null_space(Q.T @ S_w @ Q)
        W = Q @ B
    except:
        # Fallback if null space fails due to GS instability
        W = Q
    return W

def minimum_distance(A, B):
    A, B = np.asarray(A), np.asarray(B)
    min_distances = np.empty(A.shape[0], dtype=np.float64)
    for i, a in enumerate(A):
        distances = cdist([a], B, metric='euclidean')
        min_distances[i] = np.min(distances)
    return min_distances

def evaluate_predictions(y_true, y_prob):
    # Imbalanced datasets evaluating Normal class -> flip labels
    y_true_flipped = 1 - y_true
    auc_roc = roc_auc_score(y_true_flipped, 1 - y_prob)  # Since y_prob is distance to normal
    auc_pr = average_precision_score(y_true_flipped, 1 - y_prob)
    return auc_roc * 100, auc_pr * 100

def plot_2d_projections(dataset_name, projected_data, y_test):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (method, X_proj) in zip(axes, projected_data.items()):
        if X_proj.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_proj)
        elif X_proj.shape[1] == 2:
            X_2d = X_proj
        elif X_proj.shape[1] == 1:
            X_2d = np.hstack([X_proj, np.zeros_like(X_proj)])
        else:
            X_2d = np.zeros((X_proj.shape[0], 2))
            
        # Sample points if too large to avoid cluttered plots
        n_samples = X_2d.shape[0]
        if n_samples > 10000:
            idx = np.random.choice(n_samples, 10000, replace=False)
            X_plot = X_2d[idx]
            y_plot = y_test[idx]
        else:
            X_plot = X_2d
            y_plot = y_test
            
        # Rename labels for legend (0 -> Normal, 1 -> Anomaly)
        labels = np.where(y_plot == 0, 'Normal (0)', 'Anomaly (1)')
        palette = {'Normal (0)': 'dodgerblue', 'Anomaly (1)': 'salmon'}
        
        sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=labels, palette=palette, ax=ax, alpha=0.5, s=20, edgecolor=None)
        ax.set_title(f"{method} Projection (PCA 2D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
    plt.suptitle(f"Projected Space Distribution: {dataset_name}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/Projection_2D_{dataset_name}.png", dpi=300)
    plt.close()

# ----------------- Pipeline -----------------
def run_comparison():
    datasets = ['data_ToNIoT.csv', 'data_N_BaIoT.csv', 'data_CICIoT2023.csv', 'data_BoTIoT.csv']
    scaler = 'MinMaxScaler'
    # Data directory can be set via the DATA_DIR environment variable for server deployments.
    # Example: export DATA_DIR=/data/GMM-nfst/Datascaled/NoiseOCData
    # Locally, it falls back to a relative path from this file.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_data_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', 'Datascaled', 'NoiseOCData'))
    base_dir = os.environ.get('DATA_DIR', _default_data_dir)
    results = []

    for ds in datasets:
        print(f"\nEvaluating Dataset: {ds}")
        train_path = os.path.join(base_dir, f"Train_{scaler}_{ds}")
        test_path = os.path.join(base_dir, f"Test_{scaler}_{ds}")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Skipping {ds} - File not found.")
            continue
            
        df_train = pd.read_csv(train_path).dropna()
        df_test = pd.read_csv(test_path).dropna()
        
        X_train, y_train, X_test, y_test = preprocess_data_noise(df_train, df_test, noise_percentage=1)
        
        imputer = SimpleImputer(strategy="mean")
        X_train[np.isinf(X_train)] = np.nan
        X_train = imputer.fit_transform(X_train)
        X_test[np.isinf(X_test)] = np.nan
        X_test = imputer.transform(X_test)
        
        # We process with 2 clusters for speed/consistency across both
        n_clusters = 2
        X_train_clustered, y_train_clustered, cluster_centers = cluster_kmeans(X_train, n_clusters)
        
        methods = {"SVD": calculate_NPD_svd, "Gram-Schmidt": calculate_NPD_gs}
        projected_data_for_plot = {}
        
        for method_name, npd_func in methods.items():
            print(f"  --> Running {method_name} Variant...")
            
            # Record Train Time & Memory
            tracemalloc.start()
            start_train = time.time()
            try:
                W = npd_func(X_train_clustered, y_train_clustered)
            except Exception as e:
                print(f"      Method {method_name} failed: {e}")
                tracemalloc.stop()
                continue
                
            null_point_X = np.dot(X_train, W)
            null_point_centers = np.dot(cluster_centers, W)
            train_time = time.time() - start_train
            current_mem, peak_train = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Record Test Time & Memory
            tracemalloc.start()
            start_test = time.time()
            null_point_X_test = np.dot(X_test, W)
            y_prob = minimum_distance(null_point_X_test, null_point_centers)
            test_time = time.time() - start_test
            current_mem, peak_test = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Evaluate Accuracy via Flipped Labels
            # Invert distances to represent anomaly score
            y_prob_standardized = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
            auc_roc, auc_pr = evaluate_predictions(y_test, y_prob_standardized)
            
            projected_data_for_plot[method_name] = null_point_X_test
            
            results.append({
                "Dataset": ds.replace('data_', '').replace('.csv', ''),
                "Method": method_name,
                "AUCROC": auc_roc,
                "AUCPR": auc_pr,
                "Train Time (s)": train_time,
                "Test Time (s)": test_time,
                "Peak RAM Train (MB)": peak_train / 10**6,
                "Peak RAM Test (MB)": peak_test / 10**6
            })
            
        ds_name = ds.replace('data_', '').replace('.csv', '')
        plot_2d_projections(ds_name, projected_data_for_plot, y_test)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv("svd_vs_gs_results.csv", index=False)
    print("\nResults exported to svd_vs_gs_results.csv")
    return df_results

# ----------------- Plotting -----------------
def generate_plots(df):
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 1. Performance (AUCPR & AUCROC)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df, x='Dataset', y='AUCPR', hue='Method', ax=axes[0], palette=['dodgerblue', 'salmon'])
    axes[0].set_title('AUCPR Comparison: SVD vs Gram-Schmidt')
    axes[0].set_ylabel('AUC-PR (%)')
    
    sns.barplot(data=df, x='Dataset', y='AUCROC', hue='Method', ax=axes[1], palette=['dodgerblue', 'salmon'])
    axes[1].set_title('AUCROC Comparison: SVD vs Gram-Schmidt')
    axes[1].set_ylabel('AUC-ROC (%)')
    
    plt.tight_layout()
    plt.savefig('plots/SVD_vs_GS_Performance.png', dpi=300)
    plt.close()

    # 2. Memory Footprint
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Dataset', y='Peak RAM Train (MB)', hue='Method', palette=['dodgerblue', 'salmon'], ax=ax)
    ax.set_title('Peak RAM Usage (Training Phase)')
    ax.set_ylabel('Memory (MB) - Lower is Better')
    plt.tight_layout()
    plt.savefig('plots/SVD_vs_GS_Memory.png', dpi=300)
    plt.close()

    # 3. Time Complexity
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df, x='Dataset', y='Train Time (s)', hue='Method', ax=axes[0], palette=['dodgerblue', 'salmon'])
    axes[0].set_title('Training Time Complexity')
    axes[0].set_ylabel('Seconds - Lower is Better')
    axes[0].set_yscale('log')
    
    sns.barplot(data=df, x='Dataset', y='Test Time (s)', hue='Method', ax=axes[1], palette=['dodgerblue', 'salmon'])
    axes[1].set_title('Inference Time Complexity')
    axes[1].set_ylabel('Seconds - Lower is Better')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/SVD_vs_GS_Time.png', dpi=300)
    plt.close()
    
    print("All comparison plots saved to the 'plots/' directory.")

if __name__ == "__main__":
    df = run_comparison()
    generate_plots(df)
