"""
Federated Subspace Density Sketches (FSDS).

Enables privacy-preserving exchange of low-rank manifold geometry certificates
(centroid, top-r principal subspace, eigenvalues, and null-space envelope radius)
to empower Cross-Manifold Negative Purging (CMNP).

Reference:
    Explorer 3 Report (Mathematical Foundations), Algorithm 1 & Theorem 2.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import scipy.stats
import torch


class FSDSSketch:
    """
    Federated Subspace Density Sketch S_c = {mu_c, Lambda_c, U_c, r_c^max}.

    Attributes:
        client_id: Unique client identifier.
        mu: Centroid vector of shape (D,).
        Lambda: Top-r eigenvalues of shape (r,).
        U: Top-r orthonormal eigenvectors of shape (D, r).
        r_max: Null-space tubular envelope radius.
        r: Subspace rank.
        n_samples: Number of training samples summarized.
        ambient_dim: Ambient dimension D.
    """

    def __init__(
        self,
        client_id: Union[int, str],
        mu: np.ndarray,
        Lambda: np.ndarray,
        U: np.ndarray,
        r_max: float,
        r: int,
        n_samples: int,
    ):
        self.client_id = client_id
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.Lambda = np.asarray(Lambda, dtype=np.float64).ravel()
        self.U = np.asarray(U, dtype=np.float64)
        self.r_max = float(r_max)
        self.r = int(r)
        self.n_samples = int(n_samples)
        self.ambient_dim = self.mu.shape[0]

        if self.U.shape != (self.ambient_dim, self.r):
            raise ValueError(
                f"U shape {self.U.shape} does not match (ambient_dim={self.ambient_dim}, r={self.r})"
            )
        if self.Lambda.shape[0] != self.r:
            raise ValueError(
                f"Lambda length {self.Lambda.shape[0]} does not match r={self.r}"
            )

    def null_space_distance(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute null-space distance || P_c^perp (x - mu_c) ||_2 for points X.
        
        P_c^perp (x - mu_c) = (x - mu_c) - U_c (U_c^T (x - mu_c))

        Args:
            X: Array of shape (N, D) or (D,).

        Returns:
            1D array of Euclidean distances of length N.
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        diff = X_np - self.mu  # (N, D)
        proj = diff @ self.U    # (N, r)
        recon = proj @ self.U.T # (N, D)
        residuals = diff - recon  # (N, D)
        d_null = np.linalg.norm(residuals, axis=1)
        return d_null

    def subspace_mahalanobis_distance(self, X: Union[np.ndarray, torch.Tensor], eps: float = 1e-7) -> np.ndarray:
        """
        Compute in-subspace Mahalanobis distance:
        sqrt( (x - mu_c)^T U_c Lambda_c^{-1} U_c^T (x - mu_c) )

        Args:
            X: Array of shape (N, D) or (D,).
            eps: Epsilon floor for eigenvalues to avoid division by zero.

        Returns:
            1D array of Mahalanobis distances of length N.
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        diff = X_np - self.mu  # (N, D)
        proj = diff @ self.U   # (N, r)
        lambda_safe = np.clip(self.Lambda, eps, None)
        quad_form = np.sum((proj ** 2) / lambda_safe, axis=1)
        d_sub = np.sqrt(np.maximum(0.0, quad_form))
        return d_sub

    def is_intruding(
        self,
        X: Union[np.ndarray, torch.Tensor],
        tau_null: float = 1.0,
        alpha: float = 0.01,
    ) -> np.ndarray:
        """
        Evaluate Hard Intrusion Indicator:
        I_intrude(x) = (dist_null <= tau_null * r_max) AND (dist_sub^2 <= chi2_r(1 - alpha))

        Args:
            X: Candidate points of shape (N, D) or (D,).
            tau_null: Null-space radius multiplier (default: 1.0).
            alpha: Chi-squared significance level (default: 0.01).

        Returns:
            Boolean numpy array of length N (True if intruding into this manifold).
        """
        d_null = self.null_space_distance(X)
        d_sub = self.subspace_mahalanobis_distance(X)

        chi2_thresh = scipy.stats.chi2.ppf(1.0 - alpha, df=max(1, self.r))
        d_sub_sq = d_sub ** 2

        null_condition = d_null <= (tau_null * self.r_max)
        sub_condition = d_sub_sq <= chi2_thresh

        return null_condition & sub_condition

    def continuous_intrusion_weight(
        self,
        X: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Compute continuous soft intrusion density:
        phi(x) = exp( -0.5 * ( (dist_null / r_max)^2 + dist_sub^2 ) )

        Returns:
            Array of values in [0, 1] of length N.
        """
        d_null = self.null_space_distance(X)
        d_sub = self.subspace_mahalanobis_distance(X, eps=eps)
        r_scale = max(self.r_max, eps)

        exponent = -0.5 * ((d_null / r_scale) ** 2 + d_sub ** 2)
        return np.exp(np.clip(exponent, -50.0, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sketch to dictionary for federated transmission."""
        return {
            "client_id": self.client_id,
            "mu": self.mu.tolist(),
            "Lambda": self.Lambda.tolist(),
            "U": self.U.tolist(),
            "r_max": self.r_max,
            "r": self.r,
            "n_samples": self.n_samples,
            "ambient_dim": self.ambient_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FSDSSketch":
        """Deserialize sketch from dictionary."""
        return cls(
            client_id=data["client_id"],
            mu=np.array(data["mu"], dtype=np.float64),
            Lambda=np.array(data["Lambda"], dtype=np.float64),
            U=np.array(data["U"], dtype=np.float64),
            r_max=float(data["r_max"]),
            r=int(data["r"]),
            n_samples=int(data["n_samples"]),
        )


def compute_fsds_sketch(
    X: Union[np.ndarray, torch.Tensor],
    client_id: Union[int, str] = 0,
    rank: int = 10,
    beta: float = 2.0,
    eps: float = 1e-7,
) -> FSDSSketch:
    """
    Construct a Federated Subspace Density Sketch from client normal dataset D_c.

    Args:
        X: Normal training samples of shape (N, D).
        client_id: Client identifier.
        rank: Target subspace rank r (clipped to min(N - 1, D - 1)).
        beta: Envelope multiplier for tubular bound (default: 2.0).
        eps: Epsilon floor for numerical stability.

    Returns:
        FSDSSketch containing {mu_c, Lambda_c, U_c, r_c^max}.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)

    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_np.shape}")

    n_samples, ambient_dim = X_np.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to compute sketch, got {n_samples}")

    # Determine practical rank
    effective_rank = min(rank, ambient_dim, n_samples - 1)
    effective_rank = max(1, effective_rank)

    # 1. Centroid
    mu = np.mean(X_np, axis=0)
    centered = X_np - mu

    # 2. Economy SVD of centered data
    # centered = V @ diag(S) @ W^T
    # Covariance Sigma = (1/N) * centered^T @ centered = W @ (S^2 / N) @ W^T
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Eigenvalues of covariance
    eigenvalues = (S ** 2) / float(n_samples)

    # Top-r principal eigenvectors: rows of Vt correspond to columns of W
    U = Vt[:effective_rank, :].T  # Shape: (D, r)
    Lambda = eigenvalues[:effective_rank]  # Shape: (r,)

    # 3. Compute null-space residuals
    proj = centered @ U    # (N, r)
    recon = proj @ U.T     # (N, D)
    residuals = centered - recon  # (N, D)
    null_norms = np.linalg.norm(residuals, axis=1)

    # Residual covariance trace: sum of remaining eigenvalues
    tr_res = np.sum(eigenvalues[effective_rank:]) if len(eigenvalues) > effective_rank else 0.0
    tr_res = max(0.0, float(tr_res))

    # Envelope radius: max observed null residual + beta * dispersion
    dispersion = max(float(np.std(null_norms)), np.sqrt(tr_res / max(1, ambient_dim - effective_rank)))
    r_max = float(np.max(null_norms) + beta * dispersion)
    r_max = max(r_max, eps)

    return FSDSSketch(
        client_id=client_id,
        mu=mu,
        Lambda=Lambda,
        U=U,
        r_max=r_max,
        r=effective_rank,
        n_samples=n_samples,
    )
