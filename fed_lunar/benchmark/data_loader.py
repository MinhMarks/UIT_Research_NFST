"""
Benchmark Data Loader & Non-IID Dirichlet Partitioner with Offline Synthetic Fallback.

Supports canonical IoT intrusion detection datasets:
- BoTIoT (35 features - Smart Home Gateway Botnet DDoS & Recon)
- EdgeIIoTset (42 features - Industrial SCADA Multi-protocol Telemetry)
- CICIoT2023 (46 features - Large-scale Smart City Volumetric Floods)
- N_BaIoT (115 features - Commercial IoT Hardware Botnet)

Key Capabilities:
1. One-Class FL Data Pipeline:
   - Training sets contain strictly benign telemetry (y = 0).
   - Test evaluation stream enforces realistic anomaly contamination bound (<= 5%).
2. Non-IID Dirichlet Partitioner:
   - Latent cluster-guided Dirichlet partitioning to induce subspace manifold heterogeneity.
   - Strict sample conservation: sum(n_m) == N with non-empty client guarantees.
3. Dual-Mode Operation:
   - Seamlessly loads real pre-scaled CSV files from Official_OC_Data when available.
   - Transparently activates high-fidelity synthetic fallback when running offline/locally.

References:
    - Hsu et al., "Measuring the Effects of Non-Identical Distributions on Federated Visual Classification", arXiv 2019.
    - Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    - Goodge et al., "LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks", AAAI 2022.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import glob
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger("FedLUNAR.DataLoader")

IOT_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "BoTIoT": {
        "dim": 35,
        "num_clusters": 5,
        "domain": "Smart Home Gateway Botnet DDoS & Recon",
        "description": "35-feature network flow telemetry from smart home IoT devices",
    },
    "EdgeIIoTset": {
        "dim": 42,
        "num_clusters": 6,
        "domain": "Industrial SCADA & Multi-protocol Telemetry",
        "description": "42-feature telemetry across Modbus, MQTT, and industrial sensors",
    },
    "CICIoT2023": {
        "dim": 46,
        "num_clusters": 8,
        "domain": "Smart City Volumetric Floods (105 devices)",
        "description": "46-feature high-throughput volumetric flood attack telemetry",
    },
    "N_BaIoT": {
        "dim": 115,
        "num_clusters": 9,
        "domain": "Commercial IoT Hardware Botnet (9 devices)",
        "description": "115-feature hardware botnet statistics over 5 time windows",
    },
    "ToNIoT": {
        "dim": 44,
        "num_clusters": 6,
        "domain": "Heterogeneous IoT & IIoT Network",
        "description": "44-feature network and host telemetry dataset",
    },
    "IoTID20": {
        "dim": 83,
        "num_clusters": 7,
        "domain": "Smart Home IoT Intrusion Detection",
        "description": "83-feature network flow dataset for IoT devices",
    },
}


class SyntheticIoTGenerator:
    """Deterministic, high-fidelity offline synthetic fallback generator for IoT datasets."""

    @staticmethod
    def generate(
        dataset_name: str,
        n_train_normal: int = 2000,
        n_test_normal: int = 950,
        n_test_anomaly: int = 50,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes multimodal normal manifolds and separated anomaly samples.
        Ensures test stream contamination <= 5% (default: 50 / (950 + 50) = 5.0%).

        Args:
            dataset_name: Canonical dataset name to retrieve target feature dimension.
            n_train_normal: Number of benign normal training samples (y=0).
            n_test_normal: Number of benign normal test samples (y=0).
            n_test_anomaly: Number of anomaly test samples (y=1).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (X_train_norm, y_train_norm, X_test, y_test).
        """
        rng = np.random.RandomState(seed)
        config = IOT_DATASET_CONFIGS.get(dataset_name, {"dim": 35, "num_clusters": 5})
        dim = config["dim"]
        n_clusters = config["num_clusters"]

        # 1. Establish latent cluster centers for normal manifold in ambient space
        cluster_centers = rng.uniform(low=-2.5, high=2.5, size=(n_clusters, dim)).astype(np.float32)

        def sample_normal_stream(n_samples: int) -> np.ndarray:
            if n_samples <= 0:
                return np.empty((0, dim), dtype=np.float32)
            cluster_assignments = rng.choice(n_clusters, size=n_samples)
            samples = np.zeros((n_samples, dim), dtype=np.float32)
            for c in range(n_clusters):
                idx = np.where(cluster_assignments == c)[0]
                if len(idx) > 0:
                    cov_scale = rng.uniform(0.1, 0.3)
                    samples[idx] = cluster_centers[c] + rng.normal(
                        loc=0.0, scale=cov_scale, size=(len(idx), dim)
                    ).astype(np.float32)
            return samples

        X_train_norm = sample_normal_stream(n_train_normal)
        y_train_norm = np.zeros(n_train_normal, dtype=int)

        X_test_norm = sample_normal_stream(n_test_normal)
        y_test_norm = np.zeros(n_test_normal, dtype=int)

        # 2. Synthesize displaced anomalies (outliers in exterior subspaces)
        if n_test_anomaly > 0:
            anom_origins = rng.choice(n_clusters, size=n_test_anomaly)
            X_test_anom = np.zeros((n_test_anomaly, dim), dtype=np.float32)
            for i, origin_cluster in enumerate(anom_origins):
                direction = rng.randn(dim).astype(np.float32)
                direction /= (np.linalg.norm(direction) + 1e-8)
                displacement = rng.uniform(3.5, 6.0)
                X_test_anom[i] = cluster_centers[origin_cluster] + direction * displacement
            y_test_anom = np.ones(n_test_anomaly, dtype=int)
        else:
            X_test_anom = np.empty((0, dim), dtype=np.float32)
            y_test_anom = np.empty((0,), dtype=int)

        # Combine and shuffle test set
        if len(X_test_norm) > 0 and len(X_test_anom) > 0:
            X_test = np.vstack([X_test_norm, X_test_anom])
            y_test = np.concatenate([y_test_norm, y_test_anom])
            perm = rng.permutation(len(y_test))
            X_test = X_test[perm]
            y_test = y_test[perm]
        elif len(X_test_norm) > 0:
            X_test = X_test_norm
            y_test = y_test_norm
        else:
            X_test = X_test_anom
            y_test = y_test_anom

        return X_train_norm, y_train_norm, X_test, y_test


class DirichletPartitioner:
    """
    Non-IID Dirichlet Partitioner for Federated Edge Clients.

    Partitions samples across M clients such that clients experience non-identical
    distributional shifts. Supports:
    1. 2-Stage Latent Cluster Dirichlet Partitioning: Partitions data using Dirichlet
       concentrations over latent feature clusters (Hsu et al., 2019), creating realistic
       subspace manifold heterogeneity (different clients observe different device clusters).
    2. Direct Dirichlet Sample Partitioning: Standard sample proportion allocation.

    Ensures strict sample conservation (sum(n_m) == N) and guarantees non-empty client partitions.

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
        self.num_clusters = max(num_clusters, 2 * num_clients)
        self.use_clustering = use_clustering
        self.seed = seed

    def partition(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Partition dataset X across clients with strict sample conservation sum(n_m) == N.

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
            n_clusters = min(self.num_clusters, max(2, N // 10))
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

                # Sample Dirichlet proportions for cluster c
                proportions = rng.dirichlet([self.alpha] * self.num_clients)
                counts = (proportions * n_cls).astype(int)

                # Ensure all samples in cluster c are assigned
                diff = n_cls - int(np.sum(counts))
                if diff > 0:
                    add_indices = rng.choice(self.num_clients, size=diff, replace=True)
                    for idx in add_indices:
                        counts[idx] += 1

                curr = 0
                for client_id in range(self.num_clients):
                    c_count = counts[client_id]
                    if c_count > 0:
                        client_indices[client_id].extend(cls_idx[curr : curr + c_count].tolist())
                        curr += c_count

            # Enforce non-empty partitions while preserving strict sum(n_m) == N (exact conservation)
            for client_id in range(self.num_clients):
                if len(client_indices[client_id]) == 0 and N >= self.num_clients:
                    largest_client = int(np.argmax([len(idx_list) for idx_list in client_indices]))
                    if len(client_indices[largest_client]) > 1:
                        transferred = client_indices[largest_client].pop()
                        client_indices[client_id].append(transferred)

            return [X[np.array(idx, dtype=int)] for idx in client_indices]

        else:
            # Direct sample proportion Dirichlet allocation
            proportions = rng.dirichlet([self.alpha] * self.num_clients)
            proportions = proportions / proportions.sum()
            counts = (proportions * N).astype(int)
            diff = N - int(np.sum(counts))
            if diff > 0:
                add_indices = rng.choice(self.num_clients, size=diff, replace=True)
                for idx in add_indices:
                    counts[idx] += 1
            elif diff < 0:
                sub_indices = rng.choice(self.num_clients, size=abs(diff), replace=True)
                for idx in sub_indices:
                    if counts[idx] > 0:
                        counts[idx] -= 1

            # Ensure all counts are >= 1 if N >= num_clients
            if N >= self.num_clients:
                for client_id in range(self.num_clients):
                    if counts[client_id] == 0:
                        donor = int(np.argmax(counts))
                        if counts[donor] > 1:
                            counts[donor] -= 1
                            counts[client_id] += 1

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
    - Testing data evaluates binary detection on benign (y = 0) vs attacks (y = 1)
      with contamination bound <= 5%.
    - If real CSV files are absent, seamlessly activates SyntheticIoTGenerator fallback.
    """

    SUPPORTED_DATASETS = ["BoTIoT", "EdgeIIoTset", "CICIoT2023", "N_BaIoT", "ToNIoT", "IoTID20"]

    def __init__(self, data_dir: str, allow_synthetic_fallback: bool = True):
        """
        Args:
            data_dir: Directory containing pre-scaled Official_OC_Data CSV files.
            allow_synthetic_fallback: Whether to use SyntheticIoTGenerator when files are missing.
        """
        self.data_dir = data_dir
        self.allow_synthetic_fallback = allow_synthetic_fallback

    def _resolve_file_path(self, dataset: str, split: str, scaler: str = "StandardScaler") -> Optional[str]:
        """Find CSV path corresponding to dataset, split ('Train' or 'Test'), and scaler."""
        if not os.path.exists(self.data_dir):
            return None

        # Standard patterns
        patterns = [
            f"{split}_{scaler}_data_{dataset}.csv",
            f"{split}_{scaler}_{dataset}.csv",
            f"*{split}*{scaler}*{dataset}*.csv",
            f"*{split}*{dataset}*.csv",
        ]
        for pat in patterns:
            candidate = os.path.join(self.data_dir, pat)
            if "*" in pat:
                matches = glob.glob(candidate)
                if matches:
                    return matches[0]
            elif os.path.exists(candidate):
                return candidate

        return None

    def load_dataset(
        self,
        dataset: str,
        scaler: str = "StandardScaler",
        max_train_samples: Optional[int] = 50000,
        max_test_samples: Optional[int] = 20000,
        test_contamination: float = 0.05,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test sets for a benchmark dataset.

        Args:
            dataset: Name of the dataset ('BoTIoT', 'EdgeIIoTset', 'CICIoT2023', 'N_BaIoT').
            scaler: Feature scaler name ('StandardScaler', 'QuantileTransformer', etc.).
            max_train_samples: Subsample cap for training to control memory/runtime (default: 50,000).
            max_test_samples: Subsample cap for testing (default: 20,000).
            test_contamination: Maximum anomaly contamination ratio in test stream (default: 0.05).
            seed: Random seed for sampling.

        Returns:
            Tuple of (X_train_normal, y_train_normal, X_test, y_test).
        """
        rng = np.random.RandomState(seed)

        train_path = self._resolve_file_path(dataset, "Train", scaler)
        test_path = self._resolve_file_path(dataset, "Test", scaler)

        if train_path is None or test_path is None:
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"Could not find dataset files for dataset='{dataset}', scaler='{scaler}' in {self.data_dir}"
                )
            dim = IOT_DATASET_CONFIGS.get(dataset, {}).get("dim", 35)
            logger.warning(
                f"[SYNTHETIC_FALLBACK] Data files for '{dataset}' not found in {self.data_dir}. "
                f"Generating synthetic offline dataset with D={dim}."
            )
            n_tr = min(max_train_samples or 3000, 3000)
            n_te_total = min(max_test_samples or 1000, 1000)
            if test_contamination <= 0.0:
                n_te_norm = n_te_total
                n_te_anom = 0
            elif test_contamination >= 1.0:
                n_te_norm = 0
                n_te_anom = n_te_total
            else:
                n_te_anom = max(1, int(round(n_te_total * test_contamination)))
                n_te_norm = max(1, n_te_total - n_te_anom)

            return SyntheticIoTGenerator.generate(
                dataset_name=dataset,
                n_train_normal=n_tr,
                n_test_normal=n_te_norm,
                n_test_anomaly=n_te_anom,
                seed=seed,
            )

        # 1. Load Train Set (Strict Normal Only)
        df_train = pd.read_csv(train_path)
        label_col = None
        for col in ["label", "Label", "y", "target"]:
            if col in df_train.columns:
                label_col = col
                break
        if label_col is None:
            label_col = df_train.columns[-1]

        y_tr = df_train[label_col].values.astype(int)
        feature_cols = [c for c in df_train.columns if c != label_col]
        X_tr = df_train[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)

        # Filter strictly to normal samples (label == 0)
        norm_mask = (y_tr == 0)
        X_train_normal = X_tr[norm_mask]
        y_train_normal = y_tr[norm_mask]

        if max_train_samples is not None and len(X_train_normal) > max_train_samples:
            sub_idx = rng.choice(len(X_train_normal), size=max_train_samples, replace=False)
            X_train_normal = X_train_normal[sub_idx]
            y_train_normal = y_train_normal[sub_idx]

        # 2. Load Test Set & Enforce Contamination Bound <= 5%
        df_test = pd.read_csv(test_path)
        label_col_te = None
        for col in ["label", "Label", "y", "target"]:
            if col in df_test.columns:
                label_col_te = col
                break
        if label_col_te is None:
            label_col_te = df_test.columns[-1]

        y_te = df_test[label_col_te].values.astype(int)
        X_te = df_test[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)

        idx_norm = np.where(y_te == 0)[0]
        idx_anom = np.where(y_te == 1)[0]

        if test_contamination is not None and 0.0 < test_contamination < 1.0:
            target_anom_count = int(len(idx_norm) * (test_contamination / (1.0 - test_contamination)))
            target_anom_count = max(1, min(len(idx_anom), target_anom_count))
            chosen_anom = rng.choice(idx_anom, size=target_anom_count, replace=False)
            selected_idx = np.concatenate([idx_norm, chosen_anom])
            rng.shuffle(selected_idx)
            X_te = X_te[selected_idx]
            y_te = y_te[selected_idx]

        if max_test_samples is not None and len(X_te) > max_test_samples:
            sub_te = rng.choice(len(X_te), size=max_test_samples, replace=False)
            X_te = X_te[sub_te]
            y_te = y_te[sub_te]

        return X_train_normal, y_train_normal, X_te, y_te


def partition_and_prepare_dataset(
    dataset_name: str,
    data_dir: str,
    num_clients: int = 3,
    alpha: float = 0.5,
    scaler: str = "StandardScaler",
    max_train_samples: Optional[int] = 50000,
    max_test_samples: Optional[int] = 20000,
    test_contamination: float = 0.05,
    allow_synthetic_fallback: bool = True,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    End-to-end pipeline: Loads canonical dataset and partitions normal data across clients.

    Args:
        dataset_name: Dataset name ('BoTIoT', 'EdgeIIoTset', 'CICIoT2023', 'N_BaIoT').
        data_dir: Path to directory containing pre-scaled datasets.
        num_clients: Number of federated clients.
        alpha: Non-IID Dirichlet concentration.
        scaler: Feature scaler prefix ('StandardScaler', 'QuantileTransformer').
        max_train_samples: Subsample cap for training.
        max_test_samples: Subsample cap for testing.
        test_contamination: Maximum anomaly contamination ratio in test stream (default: 0.05).
        allow_synthetic_fallback: Whether to use SyntheticIoTGenerator when files are missing.
        seed: Random seed.

    Returns:
        Tuple of:
            - client_train_data: List of M numpy arrays (client normal datasets).
            - X_test: Testing features (N_test, D).
            - y_test: Testing binary labels (N_test,).
            - metadata: Dataset summary dict.
    """
    loader = OneClassDatasetLoader(data_dir=data_dir, allow_synthetic_fallback=allow_synthetic_fallback)
    X_train_norm, y_train_norm, X_test, y_test = loader.load_dataset(
        dataset=dataset_name,
        scaler=scaler,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        test_contamination=test_contamination,
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
