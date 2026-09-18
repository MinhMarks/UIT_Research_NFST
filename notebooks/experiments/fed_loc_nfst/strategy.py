"""
FL-LOC-NFST Server & Custom Strategy
======================================
Implements the server-side aggregation for Protocol A (One-Shot).

Core algorithm (Algorithm 2 — FedLOC_Server_Adaptive_Spectral_Solve):
1. Collect (S_w_m, centroids_m, counts_m) from all M clients.
2. Compute global centroid anchors μ_k via Hierarchical K-Means on all centroids.
3. Apply scatter-shift correction:
   S_w_global = Σ_m S_w_m + Σ_k Σ_m N_mk (μ_mk - μ_k)(μ_mk - μ_k)^T
4. Compute S_t = S_w + S_b (from global centroids).
5. Adaptive spectral solve:
   - SVD of deviation matrix P_t → Q, rank_Pt
   - A = Q^T S_w Q
   - If null_space(A) is empty (L=0), use near-null relaxation:
     select eigenvectors with eigenvalue < ε_near_null
6. Broadcast W, null_centers, max_train to all clients.

Mathematical basis (Theorem 1 from report):
  S_t ≡ S_w + S_b  (algebraically exact — no approximation needed)
  ⟹  T_m = S_t - S_w = S_b  (bandwidth reduction: skip sending S_t)
"""
import time
import logging
import numpy as np
from scipy.linalg import null_space, eigh
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes, EvaluateRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    NDArrays, FitIns, EvaluateIns
)
from flwr.server.client_proxy import ClientProxy

from .config import (
    K_CLUSTERS, EPSILON_SVD, EPSILON_NEAR_NULL, L_MIN, SEED
)
from .evaluate import project_to_null, min_dist_to_centers

logger = logging.getLogger(__name__)


# ============================================================
# Scatter Aggregation
# ============================================================

def aggregate_scatter_matrices(
    S_w_list: List[np.ndarray],
    centroids_list: List[np.ndarray],
    counts_list: List[np.ndarray],
    K_global: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lossless aggregation of per-client scatter matrices with
    scatter-shift correction for centroid misalignment.

    Parameters
    ----------
    S_w_list     : List of (d, d) within-class scatter matrices, one per client
    centroids_list: List of (K_m, d) local centroids
    counts_list  : List of (K_m,) sample counts per cluster
    K_global     : Number of global anchor clusters

    Returns
    -------
    S_w_global   : (d, d) Corrected global within-class scatter matrix
    S_t_global   : (d, d) Global total scatter matrix
    global_anchors: (K_global, d) Global cluster centroids
    N_per_anchor : (K_global,) Counts per global anchor
    """
    d = S_w_list[0].shape[0]
    M = len(S_w_list)

    # --- Step 1: Compute global centroid anchors via K-Means on all local centroids ---
    all_centroids = np.vstack(centroids_list)           # (Σ K_m, d)
    all_weights   = np.concatenate(counts_list).astype(float)

    km = KMeans(n_clusters=K_global, random_state=SEED, n_init=10)
    anchor_labels = km.fit_predict(all_centroids)        # (Σ K_m,) cluster assignment
    global_anchors = km.cluster_centers_.astype(np.float32)  # (K_global, d)

    # Weighted count per global anchor
    N_per_anchor = np.zeros(K_global, dtype=np.float64)
    for i, label in enumerate(anchor_labels):
        N_per_anchor[label] += all_weights[i]

    # --- Step 2: Sum raw client scatter matrices ---
    S_w_sum = np.zeros((d, d), dtype=np.float64)
    for S_w_m in S_w_list:
        S_w_sum += S_w_m.astype(np.float64)

    # --- Step 3: Scatter-shift correction ---
    # Correct for each client's centroid deviation from global anchors:
    #   correction += N_mk (μ_mk - μ_k)(μ_mk - μ_k)^T
    correction = np.zeros((d, d), dtype=np.float64)
    ptr = 0  # pointer into all_centroids / all_weights
    for m in range(M):
        K_m = len(centroids_list[m])
        for j in range(K_m):
            mu_mk = all_centroids[ptr + j]           # local centroid
            k_global = anchor_labels[ptr + j]         # nearest global anchor
            mu_k = global_anchors[k_global]           # global anchor
            N_mk = all_weights[ptr + j]
            diff = (mu_mk - mu_k).astype(np.float64)
            correction += N_mk * np.outer(diff, diff)
        ptr += K_m

    # Normalize by total N (consistent with per-client 1/N_m normalization)
    N_total = float(np.sum(all_weights))
    S_w_global = (S_w_sum + correction / N_total).astype(np.float32)

    # --- Step 4: Compute S_t = S_w + S_b ---
    # S_b = (1/N) Σ_k N_k (μ_k - μ_global)(μ_k - μ_global)^T
    mu_global = np.average(global_anchors, weights=N_per_anchor, axis=0)
    S_b = np.zeros((d, d), dtype=np.float64)
    for k in range(K_global):
        diff_b = (global_anchors[k] - mu_global).astype(np.float64)
        S_b += N_per_anchor[k] * np.outer(diff_b, diff_b)
    S_b /= N_total
    S_t_global = (S_w_global.astype(np.float64) + S_b).astype(np.float32)

    logger.info(
        f"[Server] Scatter aggregation: d={d}, M={M}, K_global={K_global}, "
        f"N_total={N_total:.0f}, "
        f"S_w_norm={np.linalg.norm(S_w_global):.4f}, "
        f"S_t_norm={np.linalg.norm(S_t_global):.4f}"
    )

    return S_w_global, S_t_global, global_anchors, N_per_anchor


# ============================================================
# Adaptive Spectral Solve (Algorithm 2)
# ============================================================

def adaptive_spectral_solve(
    S_w_global: np.ndarray,
    S_t_global: np.ndarray,
    epsilon_svd: float = EPSILON_SVD,
    epsilon_near_null: float = EPSILON_NEAR_NULL,
    L_min: int = L_MIN,
) -> Tuple[np.ndarray, int]:
    """
    Solve for the global NPD matrix W using adaptive dual-mode spectral solve.

    Mode 1 (Exact Null Space, L > 0):
        Q ← SVD basis of S_t spanning top rank_Pt directions
        A = Q^T S_w Q
        W = Q @ null_space(A)

    Mode 2 (Near-Null Relaxation, L = 0 fallback):
        When null_space(A) is empty (real data: S_w is full-rank),
        select L_min eigenvectors of A with smallest eigenvalues < ε_near_null.

    Parameters
    ----------
    S_w_global : (d, d) global within-class scatter
    S_t_global : (d, d) global total scatter
    epsilon_svd : threshold for rank detection of S_t
    epsilon_near_null : threshold for near-null relaxation
    L_min : minimum null-space dimensions for fallback

    Returns
    -------
    W    : (d, L) projection matrix
    L    : null-space dimension
    """
    d = S_w_global.shape[0]

    # Work in float64 for numerical stability in SVD
    S_w = S_w_global.astype(np.float64)
    S_t = S_t_global.astype(np.float64)

    # --- Step 1: Total scatter basis via SVD ---
    # Compute deviation matrix representation for SVD rank detection
    # Note: SVD of S_t directly (not P_t) — equivalent for symmetric PSD matrix
    eigvals_t, eigvecs_t = eigh(S_t)  # eigenvalues ascending
    eigvals_t = np.maximum(eigvals_t, 0)  # numerical non-negativity
    rank_Pt = int(np.sum(eigvals_t > epsilon_svd))

    if rank_Pt == 0:
        logger.warning("[Server] S_t is rank-0! Using identity fallback.")
        rank_Pt = min(L_min, d)

    Q = eigvecs_t[:, -rank_Pt:].astype(np.float64)  # top rank_Pt eigenvectors (d, rank_Pt)

    # --- Step 2: Project S_w into Q's subspace ---
    A = Q.T @ S_w @ Q  # (rank_Pt, rank_Pt)

    # --- Step 3: Null space computation ---
    B = null_space(A, rcond=epsilon_svd)  # (rank_Pt, L)
    L = B.shape[1]

    logger.info(f"[Server] rank_Pt={rank_Pt}, L={L} (exact null-space)")

    # --- Step 4: Near-null relaxation fallback if L = 0 ---
    if L < L_min:
        logger.warning(
            f"[Server] L={L} < L_min={L_min}. Applying near-null relaxation "
            f"(ε_near_null={epsilon_near_null})"
        )
        # Eigendecomposition of A (symmetric), ascending order
        eigvals_A, eigvecs_A = eigh(A)
        eigvals_A = np.maximum(eigvals_A, 0)

        lambda_max = float(eigvals_A[-1]) if len(eigvals_A) > 0 else 1.0

        # Adaptive threshold: use relative gap OR absolute epsilon (whichever is larger)
        # Relative threshold: eigvals < 1% of max eigenvalue  — handles both small and large scale
        relative_threshold = max(epsilon_near_null, lambda_max * 1e-2)
        near_null_mask = eigvals_A < relative_threshold

        logger.info(
            f"[Server] Near-null: lambda_max={lambda_max:.6f}, "
            f"relative_thr={relative_threshold:.6f}, "
            f"n_candidates={np.sum(near_null_mask)}"
        )

        if not np.any(near_null_mask):
            # Absolute fallback: take L_min smallest
            logger.warning(
                f"[Server] No eigenvalues < threshold={relative_threshold:.6f}. "
                f"Taking {L_min} smallest."
            )
            near_null_idx = np.argsort(eigvals_A)[:L_min]
        else:
            near_null_idx = np.where(near_null_mask)[0]

        B = eigvecs_A[:, near_null_idx]  # (rank_Pt, L_near)
        L = B.shape[1]
        logger.info(f"[Server] Near-null fallback: L={L}, "
                    f"eigenvalues={eigvals_A[near_null_idx]}")

    # --- Step 5: Final W matrix ---
    W = (Q @ B).astype(np.float32)  # (d, L)
    logger.info(f"[Server] W computed: shape={W.shape}")

    return W, L


# ============================================================
# Compute Global Null Centers & Max Train
# ============================================================

def compute_null_centers(
    global_anchors: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Project global cluster anchors into null space to get reference centers.
    Compute max_train as max distance from any anchor to any other anchor.

    Parameters
    ----------
    global_anchors : (K_global, d)
    W              : (d, L)

    Returns
    -------
    null_centers : (K_global, L)
    max_train    : float
    """
    null_anchors = project_to_null(global_anchors, W)  # (K_global, L)

    # max_train = max distance from any anchor to its nearest other anchor
    # (mirrors centralized: max distance from each train point to nearest center)
    dists = min_dist_to_centers(null_anchors, null_anchors)
    # Exclude self-distance (0) by using second smallest
    # For anchors, self dist ≈ 0 so max of non-zero should work
    max_train = float(np.max(dists)) if len(dists) > 1 else 1.0
    if max_train < 1e-10:
        max_train = 1.0  # safety: avoid division by zero

    return null_anchors.astype(np.float32), max_train


# ============================================================
# Model Serialization (Server → Clients)
# ============================================================

def model_to_parameters(
    W: np.ndarray,
    null_centers: np.ndarray,
    max_train: float,
) -> NDArrays:
    """
    Serialize W, null_centers, max_train into Flower NDArrays.

    Layout:
        [0] W           (d×L, float32, flattened)
        [1] null_centers (K×L, float32, flattened)
        [2] [max_train]  (scalar, shape (1,))
    """
    return [
        W.flatten().astype(np.float32),
        null_centers.flatten().astype(np.float32),
        np.array([max_train], dtype=np.float32),
    ]


# ============================================================
# Custom Flower Strategy — FedLOC (Protocol A)
# ============================================================

class FedLOCStrategy(fl.server.strategy.Strategy):
    """
    One-Shot Federated LOC-NFST Strategy (Protocol A).

    Communication pattern:
      Round 1: Server sends dummy parameters → Clients return scatter matrices.
      Server aggregates → computes W → broadcasts to all clients for evaluation.
    """

    def __init__(
        self,
        num_clusters: int = K_CLUSTERS,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.K = num_clusters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Aggregated model components (set after aggregate_fit)
        self.W: Optional[np.ndarray] = None
        self.null_centers: Optional[np.ndarray] = None
        self.max_train: Optional[float] = None
        self.L: int = 0
        self.K_global: int = num_clusters

        # Tracking
        self.aggregation_time: float = 0.0
        self.fit_metrics: List[Dict] = []

        super().__init__()

    # --- Flower Strategy Interface ---

    def initialize_parameters(
        self, client_manager: fl.server.ClientManager
    ) -> Optional[Parameters]:
        """Return empty init parameters (server initiates with dummy)."""
        return ndarrays_to_parameters([np.zeros(1, dtype=np.float32)])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send fit instructions to all available clients."""
        config = {"round": server_round, "K": self.K}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=max(
                self.min_fit_clients,
                int(self.fraction_fit * client_manager.num_available())
            ),
            min_num_clients=self.min_fit_clients,
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Core aggregation: collect scatter matrices, compute global W.
        """
        if not results:
            logger.error("[Server] No results received from clients!")
            return None, {}

        t0 = time.time()
        logger.info(f"[Server] Round {server_round}: Aggregating {len(results)} clients")

        # --- Deserialize client results ---
        S_w_list, centroids_list, counts_list = [], [], []
        d_ref = None

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if len(ndarrays) < 3:
                logger.warning(f"[Server] Malformed result from client, skipping.")
                continue

            client_metrics = fit_res.metrics
            N_m = fit_res.num_examples
            K_m = client_metrics.get("n_clusters", self.K)
            d = client_metrics.get("d", None)

            if d is None:
                # Infer d from S_w shape
                d = int(np.sqrt(len(ndarrays[0])))
            if d_ref is None:
                d_ref = d

            S_w_m = ndarrays[0].reshape(d, d).astype(np.float32)
            centroids_m = ndarrays[1].reshape(K_m, d).astype(np.float32)
            counts_m = ndarrays[2].astype(np.float32)

            S_w_list.append(S_w_m)
            centroids_list.append(centroids_m)
            counts_list.append(counts_m)

            self.fit_metrics.append(client_metrics)
            logger.info(
                f"[Server] Client {client_metrics.get('client_id', '?')}: "
                f"N_m={N_m}, K_m={K_m}, "
                f"payload={client_metrics.get('payload_kb', '?')} KB"
            )

        if not S_w_list:
            return None, {}

        # --- Aggregate scatter matrices ---
        K_global = min(self.K, sum(len(c) for c in centroids_list))
        self.K_global = K_global

        S_w_global, S_t_global, global_anchors, N_per_anchor = aggregate_scatter_matrices(
            S_w_list, centroids_list, counts_list, K_global
        )

        # --- Adaptive spectral solve ---
        self.W, self.L = adaptive_spectral_solve(S_w_global, S_t_global)

        # --- Compute null-space centers for inference ---
        self.null_centers, self.max_train = compute_null_centers(global_anchors, self.W)

        self.aggregation_time = time.time() - t0
        logger.info(
            f"[Server] Aggregation done: W={self.W.shape}, L={self.L}, "
            f"K_global={K_global}, max_train={self.max_train:.6f}, "
            f"time={self.aggregation_time:.3f}s"
        )

        # Serialize model for broadcast to clients
        model_ndarrays = model_to_parameters(self.W, self.null_centers, self.max_train)
        return ndarrays_to_parameters(model_ndarrays), {
            "aggregation_time_s": round(self.aggregation_time, 4),
            "L": self.L,
            "K_global": K_global,
        }

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Send model to clients for evaluation."""
        if self.W is None:
            return []

        config = {
            "round": server_round,
            "L": self.L,
            "K_global": self.K_global,
        }
        eval_ins = EvaluateIns(parameters, config)
        clients = client_manager.sample(
            num_clients=max(
                self.min_evaluate_clients,
                int(self.fraction_evaluate * client_manager.num_available())
            ),
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics across clients (weighted average)."""
        if not results:
            return None, {}

        total_samples = sum(r.num_examples for _, r in results)
        weighted_f1 = sum(
            r.metrics.get("F1 Score", 0.0) * r.num_examples for _, r in results
        ) / total_samples
        weighted_auc = sum(
            r.metrics.get("AUCROC", 0.0) * r.num_examples for _, r in results
        ) / total_samples
        weighted_mcc = sum(
            r.metrics.get("MCC", 0.0) * r.num_examples for _, r in results
        ) / total_samples

        aggregated = {
            "FL_F1_weighted": round(weighted_f1, 4),
            "FL_AUCROC_weighted": round(weighted_auc, 4),
            "FL_MCC_weighted": round(weighted_mcc, 4),
            "round": server_round,
            "num_clients": len(results),
            "total_test_samples": total_samples,
        }
        logger.info(f"[Server] Aggregated eval: {aggregated}")

        # Return weighted loss (1 - F1) and metrics
        return 1.0 - weighted_f1, aggregated

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (not used in Protocol A — evaluation done by clients)."""
        return None
