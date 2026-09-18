"""
FL-LOC-NFST Data Utilities
============================
Dataset loading, partitioning (IID and Dirichlet non-IID), and preprocessing
for simulated Federated Learning experiments.

Partition strategies:
- IID:     Shuffle-then-split evenly across M clients.
- Non-IID: Dirichlet distribution (α parameter) — simulates data heterogeneity
           common in real-world IoT deployments.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ============================================================
# Dataset Loading
# ============================================================

META_COLS = [
    'pkSeqID', 'stime', 'ltime', 'seq', 'saddr', 'daddr', 'sport', 'dport',
    'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'state', 'flgs', 'proto'
]


def load_dataset(train_path: str, test_path: str, noise_pct: float = 1.0):
    """
    Load a pre-scaled train/test CSV pair and preprocess.
    Drops metadata features, imputes NaN/inf, optionally injects noise.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray (float32)
    """
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Drop non-informative metadata columns
    for col in META_COLS:
        if col in df_train.columns:
            df_train.drop(columns=[col], inplace=True)
        if col in df_test.columns:
            df_test.drop(columns=[col], inplace=True)

    X_train = df_train.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = df_train.iloc[:, -1].to_numpy()
    X_test  = df_test.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test  = df_test.iloc[:, -1].to_numpy()

    # Fix inf/nan
    X_train[np.isinf(X_train)] = np.nan
    X_test[np.isinf(X_test)]   = np.nan
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test  = imputer.transform(X_test).astype(np.float32)

    # Optional noise injection from test anomalies into train
    if noise_pct > 0:
        X_train, y_train, X_test, y_test = _inject_noise(
            X_train, y_train, X_test, y_test, noise_pct
        )

    logger.info(
        f"Loaded: Train={X_train.shape}, Test={X_test.shape}, "
        f"Features={X_train.shape[1]}, Noise={noise_pct}%"
    )
    return X_train, y_train, X_test, y_test


def _inject_noise(X_train, y_train, X_test, y_test, noise_pct):
    """Inject anomaly samples from test into train (same logic as centralized baseline)."""
    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_pct / 100))
    if noise_count > 0:
        anom_idx = np.where(y_test == 1)[0]
        if len(anom_idx) > 0:
            chosen = np.random.choice(anom_idx, size=min(noise_count, len(anom_idx)), replace=False)
            X_noise = X_test[chosen]
            mask = np.ones(len(y_test), dtype=bool)
            mask[chosen] = False
            X_test, y_test = X_test[mask], y_test[mask]
            X_train = np.vstack((X_train, X_noise))
            y_train = np.concatenate((y_train, np.zeros(len(X_noise))))
    return X_train, y_train, X_test, y_test


# ============================================================
# IID Partition
# ============================================================

def partition_iid(X_train: np.ndarray, num_clients: int, seed: int = 42):
    """
    Shuffle and split training data evenly across M clients.

    Returns
    -------
    List of (X_client,) tuples, one per client.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    splits = np.array_split(idx, num_clients)
    partitions = [X_train[s] for s in splits]
    for i, p in enumerate(partitions):
        logger.info(f"Client {i} (IID): {len(p)} samples")
    return partitions


# ============================================================
# Non-IID Partition (Dirichlet)
# ============================================================

def partition_dirichlet(X_train: np.ndarray, X_test: np.ndarray,
                         y_test: np.ndarray, num_clients: int,
                         alpha: float = 0.5, seed: int = 42):
    """
    Dirichlet-based non-IID partition. Since train data is 100% normal (OC setting),
    we simulate heterogeneity by distributing based on feature-space clusters
    (K-Means soft pseudo-labels on normal data).

    Approach:
    1. K-Means on X_train with K = num_clients * 3
    2. Dirichlet(alpha) distribution over cluster assignments per client
    3. Sample indices proportional to Dirichlet weights

    Returns
    -------
    List of X_client arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    K = num_clients * 3  # over-cluster for more heterogeneity control

    # Cluster training data into pseudo-classes
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    pseudo_labels = km.fit_predict(X_train)

    # Dirichlet distribution: q[k,m] = probability that cluster k goes to client m
    q = rng.dirichlet(alpha=np.ones(num_clients) * alpha, size=K)  # (K, M)

    client_indices = [[] for _ in range(num_clients)]
    for k in range(K):
        cluster_idx = np.where(pseudo_labels == k)[0]
        rng.shuffle(cluster_idx)
        # Assign proportionally
        proportions = (q[k] * len(cluster_idx)).astype(int)
        # Fix rounding: ensure total = len(cluster_idx)
        proportions[-1] = len(cluster_idx) - proportions[:-1].sum()
        ptr = 0
        for m, cnt in enumerate(proportions):
            client_indices[m].extend(cluster_idx[ptr:ptr+cnt].tolist())
            ptr += cnt

    partitions = [X_train[idx] for idx in client_indices]
    for i, p in enumerate(partitions):
        logger.info(f"Client {i} (Dirichlet α={alpha}): {len(p)} samples")
    return partitions


# ============================================================
# Test Data Sharding (for local evaluation)
# ============================================================

def shard_test_data(X_test: np.ndarray, y_test: np.ndarray, num_clients: int, seed: int = 42):
    """
    Distribute test data evenly for per-client local evaluation.
    In FL, each client has its own test set reflecting its local deployment.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_test))
    splits = np.array_split(idx, num_clients)
    return [(X_test[s], y_test[s]) for s in splits]
