"""
FL-LOC-NFST Client
===================
Implements the Federated LOC-NFST client (Protocol A - One-Shot).

Each client:
1. Receives global cluster anchors from server (round 0 broadcast).
2. Runs local K-Means to form pseudo-classes.
3. Computes local within-class scatter S_w_m via Welford accumulation.
4. Sends (S_w_m, local_centroids_m, counts_m) to server.
5. Receives aggregated W, null_centers, max_train from server.
6. Performs local inference and evaluation.

Algorithm reference:
  Algorithm 1 — FedLOC_Client_Streaming_Welford
  from FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md
"""
import time
import logging
import numpy as np
import flwr as fl

from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional

from .config import (
    K_CLUSTERS, EPSILON_SVD, SEED, EXPECTED_PAYLOAD_KB
)
from .evaluate import compute_fl_scores, evaluate_scores

logger = logging.getLogger(__name__)


# ============================================================
# Serialization Helpers
# ============================================================

def scatter_to_parameters(
    S_w_m: np.ndarray,
    centroids_m: np.ndarray,
    counts_m: np.ndarray
) -> List[np.ndarray]:
    """
    Serialize client scatter statistics into a flat list of numpy arrays
    suitable for Flower's NDArrays parameter format.

    Layout:
        [0] S_w_m   (d×d float32, flattened)
        [1] centroids_m (K×d float32, flattened)
        [2] counts_m  (K, int32)
    """
    return [
        S_w_m.flatten().astype(np.float32),
        centroids_m.flatten().astype(np.float32),
        counts_m.astype(np.float32),  # use float32 for Flower compatibility
    ]


def parameters_to_model(
    parameters: List[np.ndarray],
    d: int, L: int, K_global: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Deserialize server broadcast parameters into model components.

    Layout (from server):
        [0] W           (d×L float32, flattened)
        [1] null_centers (K_global×L float32, flattened)
        [2] [max_train]  (scalar)
    """
    W = parameters[0].reshape(d, L).astype(np.float32)
    null_centers = parameters[1].reshape(K_global, L).astype(np.float32)
    max_train = float(parameters[2][0])
    return W, null_centers, max_train


# ============================================================
# Local Scatter Computation (Welford-style accumulation)
# ============================================================

def compute_local_scatter(X_local: np.ndarray, K: int = K_CLUSTERS):
    """
    Compute local within-class scatter matrix S_w_m using incremental
    accumulation (Welford-style — O(K·d²) RAM, O(N·d²/K) time per client).

    Protocol A: Each client performs its own K-Means independently.
    The scatter-shift correction (for centroid misalignment between clients)
    is handled by the SERVER using the per-client centroids.

    Parameters
    ----------
    X_local : (N_m, d) float32 — local training data (normal traffic only)
    K : int — number of pseudo-classes

    Returns
    -------
    S_w_m        : (d, d) float32 — within-class scatter matrix
    centroids_m  : (K_actual, d) float32 — local cluster centroids
    counts_m     : (K_actual,) int32 — samples per cluster
    y_clustered  : (N_m,) — cluster labels
    training_time: float
    """
    t0 = time.time()
    N_m, d = X_local.shape
    K = min(K, N_m)  # cap K to avoid KMeans crash

    # Step 1: K-Means clustering on local data
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    y_clustered = km.fit_predict(X_local)
    centroids_m = km.cluster_centers_.astype(np.float32)  # (K, d)

    classes = np.unique(y_clustered)
    K_actual = len(classes)

    # Step 2: Incremental within-class scatter accumulation
    # S_w_m = (1/N_m) Σ_k Σ_{x ∈ C_k} (x - μ_k)(x - μ_k)^T
    S_w_m = np.zeros((d, d), dtype=np.float32)
    counts_m = np.zeros(K_actual, dtype=np.int32)

    for i, cls in enumerate(classes):
        mask = (y_clustered == cls)
        X_cls = X_local[mask]                           # (n_k, d)
        mu_cls = centroids_m[cls]                        # (d,)
        diff = (X_cls - mu_cls).astype(np.float32)      # (n_k, d)
        S_w_m += diff.T @ diff                           # accumulate (d, d)
        counts_m[i] = len(X_cls)
        del diff

    S_w_m /= N_m  # normalize by total local samples

    training_time = time.time() - t0

    # Log payload size estimate
    payload_bytes = (S_w_m.nbytes + centroids_m.nbytes + counts_m.nbytes)
    logger.info(
        f"[Client] Local scatter computed: d={d}, K_actual={K_actual}, "
        f"N_m={N_m}, S_w_m={S_w_m.shape}, "
        f"payload={payload_bytes/1024:.2f} KB, time={training_time:.3f}s"
    )

    return S_w_m, centroids_m, counts_m, y_clustered, training_time


# ============================================================
# Flower Client
# ============================================================

class FedLOCClient(fl.client.NumPyClient):
    """
    Federated LOC-NFST Client (Protocol A — One-Shot).

    fit():      Compute local scatter statistics and send to server.
    evaluate(): Score local test data using server-broadcast W matrix.
    """

    def __init__(
        self,
        client_id: int,
        X_local: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        K: int = K_CLUSTERS,
    ):
        self.client_id = client_id
        self.X_local = X_local.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_test = y_test
        self.K = K
        self.d = X_local.shape[1]

        # Will be set after server broadcasts model
        self.W = None
        self.null_centers = None
        self.max_train = None
        self.L = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Not used in Protocol A (server initiates aggregation)."""
        # Return zeros as placeholder
        return [np.zeros(1, dtype=np.float32)]

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Round 1: Compute local scatter and return to server.
        """
        logger.info(f"[Client {self.client_id}] Starting local scatter computation...")

        S_w_m, centroids_m, counts_m, y_clustered, train_time = compute_local_scatter(
            self.X_local, K=self.K
        )

        # Serialize for Flower NDArrays
        params_out = scatter_to_parameters(S_w_m, centroids_m, counts_m)

        payload_bytes = sum(p.nbytes for p in params_out)
        metrics = {
            "client_id": self.client_id,
            "train_time_s": round(train_time, 4),
            "n_samples": len(self.X_local),
            "payload_bytes": payload_bytes,
            "payload_kb": round(payload_bytes / 1024, 2),
            "n_clusters": len(counts_m),
            "d": self.d,
        }
        logger.info(f"[Client {self.client_id}] fit() complete: {metrics}")

        return params_out, len(self.X_local), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        After server broadcasts W: score local test set and return metrics.
        """
        if len(parameters) < 3:
            logger.warning(f"[Client {self.client_id}] No model received yet, skipping evaluate.")
            return 0.0, len(self.X_test), {}

        K_global = config.get("K_global", self.K)
        L = config.get("L", 1)

        try:
            W, null_centers, max_train = parameters_to_model(parameters, self.d, L, K_global)
        except Exception as e:
            logger.error(f"[Client {self.client_id}] Model deserialization failed: {e}")
            return 0.0, len(self.X_test), {"error": str(e)}

        self.W = W
        self.null_centers = null_centers
        self.max_train = max_train
        self.L = L

        # Local inference
        t_infer = time.time()
        y_proba = compute_fl_scores(self.X_test, W, null_centers, max_train)
        infer_time = time.time() - t_infer

        metrics = evaluate_scores(self.y_test, y_proba)
        metrics["client_id"] = self.client_id
        metrics["infer_time_s"] = round(infer_time, 4)
        metrics["n_test"] = len(self.X_test)

        logger.info(
            f"[Client {self.client_id}] evaluate(): "
            f"F1={metrics['F1 Score']:.4f}, AUC-ROC={metrics['AUCROC']:.2f}%, "
            f"infer={infer_time:.3f}s"
        )

        # Loss = 1 - F1 (for Flower's aggregation)
        loss = 1.0 - metrics["F1 Score"]
        return loss, len(self.X_test), metrics


# ============================================================
# Client Factory
# ============================================================

def create_client_fn(
    client_partitions: List[np.ndarray],
    test_shards: List[Tuple[np.ndarray, np.ndarray]],
    K: int = K_CLUSTERS,
):
    """
    Returns a Flower client_fn compatible with fl.simulation.start_simulation().

    Parameters
    ----------
    client_partitions : List of X_local per client (training data)
    test_shards       : List of (X_test, y_test) per client
    """
    def client_fn(cid: str) -> FedLOCClient:
        cid_int = int(cid)
        X_local = client_partitions[cid_int]
        X_test, y_test = test_shards[cid_int]
        return FedLOCClient(
            client_id=cid_int,
            X_local=X_local,
            X_test=X_test,
            y_test=y_test,
            K=K,
        )
    return client_fn
