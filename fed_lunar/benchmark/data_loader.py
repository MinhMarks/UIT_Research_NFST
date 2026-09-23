"""
Benchmark Data Loader & Non-IID Dirichlet Partitioner.

Supports canonical IoT intrusion detection datasets:
- BoTIoT
- EdgeIIoTset
- CICIoT2023
- N_BaIoT

Implements one-class federated learning data pipelines:
- Train sets contain strictly benign telemetry (y = 0).
- Test sets evaluate binary discrimination between normal (0) and attack (1).
- Non-IID partitioning via Dirichlet distribution across client nodes.

References:
    - Hsu et al., "Measuring the Effects of Non-Identical Distributions on Federated Visual Classification", arXiv 2019.
    - Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    - Goodge et al., "LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks", AAAI 2022.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class DirichletPartitioner:
    """
    Non-IID Dirichlet Partitioner for Federated Edge Clients.

    Partitions samples across M clients such that clients experience non-identical
    distributional shifts. Supports:
    1. Cluster-based Dirichlet Partitioning: Partitions data using Dirichlet
       concentrations over latent feature clusters (Hsu et al., 2019), creating realistic
       subspace manifold heterogeneity (different clients observe different device clusters).
    2. Direct Dirichlet Sample Partitioning: Standard sample proportion allocation.

    Args:
        num_clients: Number of participating federated clients (default: 3).
        alpha: Dirichlet concentration parameter (lower = higher Non-IID skew) (default: 0.5).
        num_clusters: Number of latent manifold clusters for heterogeneous assignment (default: 5).
        use_clustering: Whether to partition via latent clusters to induce manifold shifts (default: True).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_clients: int = 3,
        alpha: float = 0.5,
        num_clusters: int = 5,
        use_clustering: bool = True,
        seed: int = 42,
    ):
        if num_clients < 1:
            raise ValueError(f"num_clients must be at least 1, got {num_clients}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.num_clients = num_clients
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.use_clustering = use_clustering
        self.seed = seed

    def partition(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Partition dataset X across clients.

        Args:
            X: Input samples of shape (N, D).
            labels: Optional cluster/class labels of shape (N,). If None and use_clustering
                    is True, latent clusters are fitted via MiniBatchKMeans.

        Returns:
            List of M numpy arrays, where each array contains the client's local samples.
        """
        rng = np.random.RandomState(self.seed)
        N = len(X)
        if N == 0:
            return [np.empty((0, X.shape[1]), dtype=X.dtype) for _ in range(self.num_clients)]

        if self.num_clients == 1:
            return [X.copy()]

        # Generate latent cluster labels if clustering is requested and no labels provided
        if labels is None and self.use_clustering and N >= self.num_clusters * 2:
            n_clusters = min(self.num_clusters, N // 2)
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.seed,
                batch_size=min(1024, N),
                n_init=3,
            )
            cat_labels = kmeans.fit_predict(X)
        elif labels is not None:
            cat_labels = np.asarray(labels)
        else:
            cat_labels = None

        if cat_labels is not None and len(np.unique(cat_labels)) > 1:
            # Cluster-based Non-IID Dirichlet allocation
            classes = np.unique(cat_labels)
            client_indices: List[List[int]] = [[] for _ in range(self.num_clients)]

            for c in classes:
                cls_idx = np.where(cat_labels == c)[0]
                rng.shuffle(cls_idx)
                n_cls = len(cls_idx)

                # Sample Dirichlet proportions for class c
                proportions = rng.dirichlet([self.alpha] * self.num_clients)
                counts = (proportions * n_cls).astype(int)

                # Ensure all samples are assigned
                diff = n_cls - np.sum(counts)
                if diff > 0:
                    add_indices = rng.choice(self.num_clients, size=diff, replace=True)
                    for idx in add_indices:
                        counts[idx] += 1

                curr = 0
                for client_id in range(self.num_clients):
                    c_count = counts[client_id]
                    if c_count > 0:
                        client_indices[client_id].extend(cls_idx[curr : curr + c_count])
                        curr += c_count

            # Ensure every client receives at least a few samples if possible
            result: List[np.ndarray] = []
            for client_id in range(self.num_clients):
                idx = np.array(client_indices[client_id], dtype=int)
                if len(idx) == 0:
                    # Fallback: borrow random samples if a client got 0
                    idx = rng.choice(N, size=min(10, N), replace=False)
                result.append(X[idx])

            return result

        else:
            # Direct sample proportion Dirichlet allocation
            proportions = rng.dirichlet([self.alpha] * self.num_clients)
            proportions = proportions / proportions.sum()
            counts = (proportions * N).astype(int)
            counts[-1] = N - np.sum(counts[:-1])

            perm = rng.permutation(N)
            splits: List[np.ndarray] = []
            curr = 0
            for c in counts:
                splits.append(X[perm[curr : curr + c]])
                curr += c
            return splits


class OneClassDatasetLoader:
    """
    Standardized Loader for One-Class IoT Telemetry Benchmark Datasets.

    Loads and prepares pre-scaled datasets from Official_OC_Data format.
    Ensures zero data leakage:
    - Training data contains strictly benign telemetry (y = 0).
    - Testing data evaluates binary detection on benign (y = 0) vs attacks (y = 1).
    """

    SUPPORTED_DATASETS = ["BoTIoT", "EdgeIIoTset", "CICIoT2023", "N_BaIoT", "ToNIoT", "IoTID20"]

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing pre-scaled Official_OC_Data CSV files.
        """
        self.data_dir = data_dir

    def _resolve_file_path(self, dataset: str, split: str, scaler: str = "StandardScaler") -> str:
        """Find CSV path corresponding to dataset, split ('Train' or 'Test'), and scaler."""
        # Standard pattern: Train_StandardScaler_data_BoTIoT.csv
        filename = f"{split}_{scaler}_data_{dataset}.csv"
        candidate = os.path.join(self.data_dir, filename)
        if os.path.exists(candidate):
            return candidate

        # Try case-insensitive or wildcards
        pattern = os.path.join(self.data_dir, f"*{split}*{scaler}*{dataset}*.csv")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

        # Fallback to any scaler for this split and dataset
        fallback_pattern = os.path.join(self.data_dir, f"*{split}*{dataset}*.csv")
        fallback_matches = glob.glob(fallback_pattern)
        if fallback_matches:
            return fallback_matches[0]

        raise FileNotFoundError(
            f"Could not find dataset file for dataset='{dataset}', split='{split}', scaler='{scaler}' in {self.data_dir}"
        )

    def load_dataset(
        self,
        dataset: str,
        scaler: str = "StandardScaler",
        max_train_samples: Optional[int] = 50000,
        max_test_samples: Optional[int] = 20000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test sets for a benchmark dataset.

        Args:
            dataset: Name of the dataset ('BoTIoT', 'EdgeIIoTset', 'CICIoT2023', 'N_BaIoT').
            scaler: Feature scaler name ('StandardScaler', 'MinMaxScaler', etc.).
            max_train_samples: Subsample cap for training to control memory/runtime (default: 50,000).
            max_test_samples: Subsample cap for testing (default: 20,000).
            seed: Random seed for sampling.

        Returns:
            Tuple of (X_train_normal, y_train_normal, X_test, y_test).
        """
        rng = np.random.RandomState(seed)

        train_path = self._resolve_file_path(dataset, "Train", scaler)
        test_path = self._resolve_file_path(dataset, "Test", scaler)

        # 1. Load Train Set
        df_train = pd.read_csv(train_path)
        # Identify label column (usually 'label' or 'Label' or last column)
        label_col = None
        for col in ["label", "Label", "y", "target"]:
            if col in df_train.columns:
                label_col = col
                break
        if label_col is None:
            label_col = df_train.columns[-1]

        # Extract features and labels
        y_tr = df_train[label_col].values.astype(int)
        feature_cols = [c for c in df_train.columns if c != label_col]
        X_tr = df_train[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)

        # In one-class training, filter strictly to normal samples (label == 0)
        norm_mask = (y_tr == 0)
        X_train_normal = X_tr[norm_mask]
        y_train_normal = y_tr[norm_mask]

        if max_train_samples is not None and len(X_train_normal) > max_train_samples:
            sub_idx = rng.choice(len(X_train_normal), size=max_train_samples, replace=False)
            X_train_normal = X_train_normal[sub_idx]
            y_train_normal = y_train_normal[sub_idx]

        # 2. Load Test Set
        df_test = pd.read_csv(test_path)
        y_te = df_test[label_col].values.astype(int)
        X_te = df_test[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)

        if max_test_samples is not None and len(X_te) > max_test_samples:
            # Stratified subsampling to preserve anomaly ratio
            classes = np.unique(y_te)
            test_sub_indices = []
            for cls in classes:
                cls_indices = np.where(y_te == cls)[0]
                n_cls_sub = max(1, int(len(cls_indices) * max_test_samples / len(y_te)))
                n_cls_sub = min(n_cls_sub, len(cls_indices))
                chosen = rng.choice(cls_indices, size=n_cls_sub, replace=False)
                test_sub_indices.extend(chosen)
            rng.shuffle(test_sub_indices)
            X_test = X_te[test_sub_indices]
            y_test = y_te[test_sub_indices]
        else:
            X_test = X_te
            y_test = y_te

        return X_train_normal, y_train_normal, X_test, y_test


def partition_and_prepare_dataset(
    dataset_name: str,
    data_dir: str,
    num_clients: int = 3,
    alpha: float = 0.5,
    scaler: str = "StandardScaler",
    max_train_samples: Optional[int] = 50000,
    max_test_samples: Optional[int] = 20000,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    End-to-end pipeline: Loads canonical dataset and partitions normal data across clients.

    Args:
        dataset_name: Dataset name ('BoTIoT', 'EdgeIIoTset', 'CICIoT2023', 'N_BaIoT').
        data_dir: Path to directory containing pre-scaled datasets.
        num_clients: Number of federated clients.
        alpha: Non-IID Dirichlet concentration.
        scaler: Feature scaler prefix ('StandardScaler').
        max_train_samples: Subsample cap for training.
        max_test_samples: Subsample cap for testing.
        seed: Random seed.

    Returns:
        Tuple of:
            - client_train_data: List of M numpy arrays (client normal datasets).
            - X_test: Testing features (N_test, D).
            - y_test: Testing binary labels (N_test,).
            - metadata: Dataset summary dict (feature_dim, client_sample_counts, test_attack_ratio).
    """
    loader = OneClassDatasetLoader(data_dir=data_dir)
    X_train_norm, y_train_norm, X_test, y_test = loader.load_dataset(
        dataset=dataset_name,
        scaler=scaler,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
    )

    partitioner = DirichletPartitioner(
        num_clients=num_clients,
        alpha=alpha,
        use_clustering=True,
        seed=seed,
    )
    client_train_data = partitioner.partition(X_train_norm)

    metadata = {
        "dataset": dataset_name,
        "scaler": scaler,
        "input_dim": X_train_norm.shape[1],
        "total_train_normal": len(X_train_norm),
        "client_sizes": [len(d) for d in client_train_data],
        "test_total": len(y_test),
        "test_normal_count": int(np.sum(y_test == 0)),
        "test_attack_count": int(np.sum(y_test == 1)),
        "test_attack_ratio": float(np.mean(y_test == 1)),
        "num_clients": num_clients,
        "dirichlet_alpha": alpha,
    }

    return client_train_data, X_test, y_test, metadata
