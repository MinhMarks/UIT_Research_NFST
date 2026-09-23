"""
Tier 3 Baseline: LOC-NFST Closed-Form Null-Space Theoretical Bound (Feature F8).

Closed-form Null-Space analytical baseline (T=1 communication round) serving as
the theoretical performance upper bound for Federated One-Class Intrusion Detection.
Integrates the exact Null-Space projection method from notebooks/experiments/OC_NFST_memory_optimized.py
and notebooks/experiments/fed_loc_nfst/strategy.py.

Reference:
    - Bodesheim et al., "Kernel Null Space Methods for Obstacle Detection", CVPR 2013.
    - Rahimzadeh Arashloo, "Multiple Kernel Fisher Null-Space", 2020.
    - Milestone M2 Specification (Feature F8).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans


class LOC_NFST_Bound:
    """
    Tier 3 Baseline: LOC-NFST Closed-Form Null-Space Bound.

    Calculates the exact Null-Space Projection Directions (NPDs) W in R^{D x L} and
    null-space projection operator P_N = W W^T on benign training telemetry.
    The anomaly score is the squared Euclidean norm of the projected residual:
        score(x) = ||P_N (x - m^*)||_2^2 = ||W^T (x - m^*)||_2^2
    where m^* is the nearest benign cluster anchor centroid.

    For benign samples, within-class scatter in the null space satisfies S_w = 0,
    collapsing scores to approximately 0, while unseen anomalous traffic projects
    strongly into the null space.

    Args:
        n_clusters: Number of local/global benign clusters K (default: 3).
        epsilon_svd: Singular value cutoff threshold for S_t rank detection (default: 1e-6).
        epsilon_near_null: Spectral relaxation threshold if exact null-space is empty (default: 1e-4).
        L_min: Minimum null-space dimension fallback (default: 3).
        seed: Random seed for KMeans clustering.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        epsilon_svd: float = 1e-6,
        epsilon_near_null: float = 1e-4,
        L_min: int = 3,
        seed: int = 42,
        **kwargs,
    ):
        self.n_clusters = kwargs.get("n_components", n_clusters)
        self.epsilon_svd = kwargs.get("tol", epsilon_svd)
        self.epsilon_near_null = epsilon_near_null
        self.L_min = L_min
        self.seed = seed

        self.W: Optional[np.ndarray] = None  # (D, L) projection matrix
        self.P_N: Optional[np.ndarray] = None  # (D, D) null-space projector W @ W.T
        self.centers: Optional[np.ndarray] = None  # (K, D) cluster centroids
        self.mean_total: Optional[np.ndarray] = None  # (D,) global centroid
        self.L: int = 0
        self.max_train_score: float = 1.0

    def _format_client_data(
        self,
        client_train_data: Union[List[np.ndarray], Dict[Any, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Format input client training data into a contiguous float64 array."""
        if isinstance(client_train_data, np.ndarray):
            return np.ascontiguousarray(client_train_data, dtype=np.float64)
        elif isinstance(client_train_data, (list, tuple)):
            arrays = [np.asarray(d, dtype=np.float64) for d in client_train_data]
            return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float64)
        elif isinstance(client_train_data, dict):
            arrays = [np.asarray(d, dtype=np.float64) for d in client_train_data.values()]
            return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float64)
        else:
            raise TypeError(
                f"Unsupported client_train_data type: {type(client_train_data)}. Expected ndarray, list, or dict."
            )

    @property
    def null_basis(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for projection matrix W."""
        return self.W

    @property
    def threshold(self) -> float:
        """Backward-compatible alias for max_train_score."""
        return self.max_train_score

    def score(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Backward-compatible alias for decision_function(X)."""
        return self.decision_function(X)

    def fit(
        self,
        client_train_data: Union[List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray], np.ndarray],
        y_clusters: Optional[np.ndarray] = None,
        verbose: bool = False,
        tol: Optional[float] = None,
        rounds: Optional[int] = None,
        **kwargs: Any,
    ) -> "LOC_NFST_Bound":
        """
        Compute the closed-form null-space projection matrix W and projector P_N.

        Args:
            client_train_data: Normal training data from single or multiple clients.
            y_clusters: Optional pre-assigned cluster labels for normal data.
            verbose: Whether to log spectral decomposition details.
            tol: Optional spectral tolerance overriding epsilon_svd and epsilon_near_null.
            rounds: Optional rounds parameter for unified baseline fit API.
            **kwargs: Additional parameters for interface compatibility.

        Returns:
            self (fitted instance).
        """
        if tol is not None:
            self.epsilon_svd = float(tol)
            self.epsilon_near_null = float(tol)

        X = self._format_client_data(client_train_data)
        N, D = X.shape

        if N < 2:
            raise ValueError(f"Need at least 2 training samples, got {N}")

        # 1. Total scatter mean and deviation matrix
        self.mean_total = np.mean(X, axis=0)  # (D,)
        P_t = (X - self.mean_total).T  # (D, N)

        # 2. Total scatter basis via SVD: P_t = U S V^T with full matrices
        U, s_t, _ = np.linalg.svd(P_t, full_matrices=True)
        rank_Pt = int(np.sum(s_t > self.epsilon_svd))
        rank_Pt = max(1, min(rank_Pt, min(D, N)))
        Q = U[:, :rank_Pt]  # (D, rank_Pt)
        W_null_t = U[:, rank_Pt:] if rank_Pt < D else None  # (D, D - rank_Pt)

        # 3. Cluster centroids and Within-class scatter S_w
        K = min(self.n_clusters, max(1, N // 2))
        if y_clusters is None:
            if K > 1 and N >= K:
                kmeans = KMeans(n_clusters=K, random_state=self.seed, n_init=10)
                labels = kmeans.fit_predict(X)
                self.centers = kmeans.cluster_centers_  # (K, D)
            else:
                labels = np.zeros(N, dtype=int)
                self.centers = self.mean_total[np.newaxis, :]  # (1, D)
        else:
            labels = np.asarray(y_clusters, dtype=int)
            unique_labels = np.unique(labels)
            self.centers = np.array([np.mean(X[labels == c], axis=0) for c in unique_labels])

        # Compute within-class scatter incrementally: S_w = 1/N sum_k sum_{x in C_k} (x - m_k)(x - m_k)^T
        S_w = np.zeros((D, D), dtype=np.float64)
        for c_idx in range(len(self.centers)):
            mask = (labels == c_idx)
            if np.any(mask):
                diff = X[mask] - self.centers[c_idx]
                S_w += diff.T @ diff
        S_w /= float(N)

        # 4. Project S_w into Q's subspace: A = Q^T S_w Q (rank_Pt x rank_Pt)
        A = Q.T @ S_w @ Q

        # 5. Null space computation with near-null fallback
        B = scipy.linalg.null_space(A, rcond=self.epsilon_svd)
        L_range = B.shape[1] if B.size > 0 else 0

        # If orthogonal complement exists (data manifold rank is deficient in ambient space),
        # W_null_t is an exact null space of the training manifold: (X - mu) @ W_null_t == 0.
        if W_null_t is not None and W_null_t.shape[1] > 0:
            if L_range > 0:
                W_candidate = np.hstack([W_null_t, Q @ B])
            else:
                W_candidate = W_null_t
            self.W = np.ascontiguousarray(W_candidate, dtype=np.float64)
            self.L = self.W.shape[1]
        else:
            # Full manifold rank: standard near-null fallback if exact null-space is empty
            if L_range < self.L_min:
                eigvals_A, eigvecs_A = scipy.linalg.eigh(A)
                eigvals_A = np.maximum(eigvals_A, 0.0)
                lambda_max = float(eigvals_A[-1]) if len(eigvals_A) > 0 else 1.0

                rel_thr = max(self.epsilon_near_null, lambda_max * 1e-2)
                candidates = np.where(eigvals_A < rel_thr)[0]

                if len(candidates) >= self.L_min:
                    chosen_idx = candidates[: self.L_min]
                else:
                    chosen_idx = np.argsort(eigvals_A)[: min(self.L_min, rank_Pt)]

                B = eigvecs_A[:, chosen_idx]
                L_range = B.shape[1]

            self.L = L_range
            self.W = np.ascontiguousarray(Q @ B, dtype=np.float64)  # (D, L)

        # 7. Null-space projection operator P_N = W @ W^T in R^{D x D}
        self.P_N = np.ascontiguousarray(self.W @ self.W.T, dtype=np.float64)

        if verbose:
            print(f"[LOC_NFST_Bound] Fitted W shape: {self.W.shape}, L={self.L}, rank_Pt={rank_Pt}")

        # Compute calibration score on training data for probability normalization
        train_scores = self.decision_function(X)
        self.max_train_score = float(np.percentile(train_scores, 99.0))
        if self.max_train_score <= 1e-12:
            self.max_train_score = float(np.max(train_scores)) + 1e-6

        return self

    def decision_function(self, X: Union[np.ndarray, Any], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Compute anomaly score for test samples as squared null-space projection distance.

        Args:
            X: Input samples of shape (N, D).
            batch_size: Optional batch size (ignored, null-space projection is vectorized).

        Returns:
            1D numpy array of shape (N,) containing anomaly scores.
        """
        if self.W is None or self.centers is None:
            raise RuntimeError("Model must be fitted before calling decision_function")

        X_np = np.asarray(X, dtype=np.float64)
        N, D = X_np.shape

        # Find nearest cluster center for each query sample
        if len(self.centers) == 1:
            diff = X_np - self.centers[0]  # (N, D)
        else:
            # Pairwise distances to cluster centers: (N, K)
            # Efficient vectorized distance computation
            dists = np.linalg.norm(X_np[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
            nearest_idx = np.argmin(dists, axis=1)
            diff = X_np - self.centers[nearest_idx]  # (N, D)

        # Project into null space: z = diff @ W in R^{N x L}
        z = diff @ self.W  # (N, L)
        # Anomaly score is ||z||_2^2 = sum_l z_l^2 = ||P_N (x - m^*)||_2^2
        scores = np.sum(z ** 2, axis=1)

        return scores.astype(np.float32)

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Predict class probabilities [P(normal), P(anomaly)] using calibrated null-space residuals.

        Args:
            X: Input samples of shape (N, D).

        Returns:
            2D numpy array of shape (N, 2).
        """
        scores = self.decision_function(X)
        norm_scores = np.clip(scores / (self.max_train_score + 1e-12), 0.0, 1.0)
        proba = np.zeros((len(norm_scores), 2), dtype=np.float32)
        proba[:, 1] = norm_scores
        proba[:, 0] = 1.0 - norm_scores
        return proba

    def predict(self, X: Union[np.ndarray, Any], threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels (0 = normal, 1 = anomaly).
        """
        scores = self.decision_function(X)
        thresh = self.max_train_score if threshold is None else threshold
        return (scores >= thresh).astype(int)

    score = decision_function
