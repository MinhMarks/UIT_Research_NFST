"""
Subspace Pseudo-Negative Generator with Cross-Manifold Negative Purging (CMNP).

Synthesizes pseudo-negatives via local subspace perturbation and actively purges
or debiases candidates that intrude upon peer clients' normal manifolds using
Federated Subspace Density Sketches (FSDS).

Reference:
    Explorer 3 Report (Mathematical Foundations), Algorithms 1 & 3, Theorem 2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from fed_lunar.federated.sketches import FSDSSketch


class CMNPFilter:
    """
    Cross-Manifold Negative Purging (CMNP) Filter.

    Evaluates candidate pseudo-negatives against peer manifold sketches {S_j}_{j != c}
    and discards or downweights candidates that intrude into foreign benign manifolds.

    Args:
        peer_sketches: Dictionary of peer client sketches {client_id: FSDSSketch} or list of sketches.
        tau_null: Null-space intrusion threshold multiplier (default: 1.0).
        alpha: Significance level for Chi-squared in-subspace intrusion boundary (default: 0.01).
        gamma: Soft debiasing sensitivity multiplier (default: 1.0).
        mode: Purging mode ('hard' geometric rejection or 'soft' debiased weighting).
    """

    def __init__(
        self,
        peer_sketches: Optional[Union[Dict[Union[int, str], FSDSSketch], List[FSDSSketch]]] = None,
        tau_null: float = 1.0,
        alpha: float = 0.01,
        gamma: float = 1.0,
        mode: str = "hard",
    ):
        self.tau_null = tau_null
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode.lower()

        self.peer_sketches: Dict[Union[int, str], FSDSSketch] = {}
        if peer_sketches is not None:
            self.set_peer_sketches(peer_sketches)

        # Cumulative statistics
        self.total_candidates_evaluated: int = 0
        self.total_candidates_rejected: int = 0

    def set_peer_sketches(
        self,
        peer_sketches: Union[Dict[Union[int, str], FSDSSketch], List[FSDSSketch]],
    ) -> None:
        """Update or register peer sketches."""
        if isinstance(peer_sketches, list):
            self.peer_sketches = {s.client_id: s for s in peer_sketches}
        elif isinstance(peer_sketches, dict):
            self.peer_sketches = dict(peer_sketches)
        else:
            raise TypeError("peer_sketches must be a list or dict of FSDSSketch objects")

    def check_intrusion(
        self,
        candidates: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Check if candidates intrude into ANY peer manifold.

        Args:
            candidates: Candidate points of shape (N, D).

        Returns:
            Boolean array of length N (True = candidate intrudes into at least one peer manifold).
        """
        if isinstance(candidates, torch.Tensor):
            cand_np = candidates.detach().cpu().numpy()
        else:
            cand_np = np.asarray(candidates, dtype=np.float64)

        num_candidates = cand_np.shape[0]
        if num_candidates == 0 or len(self.peer_sketches) == 0:
            return np.zeros(num_candidates, dtype=bool)

        # Logical OR across all peer sketches
        is_intruding = np.zeros(num_candidates, dtype=bool)
        for sketch in self.peer_sketches.values():
            intruding_on_peer = sketch.is_intruding(
                cand_np, tau_null=self.tau_null, alpha=self.alpha
            )
            is_intruding |= intruding_on_peer

        return is_intruding

    def filter_candidates(
        self,
        candidates: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply hard purging to remove intruding candidates.

        Args:
            candidates: Candidate points of shape (N, D).

        Returns:
            Tuple of (purged_candidates, stats_dict).
        """
        is_torch = isinstance(candidates, torch.Tensor)
        device = candidates.device if is_torch else None

        if is_torch:
            cand_np = candidates.detach().cpu().numpy()
        else:
            cand_np = np.asarray(candidates, dtype=np.float64)

        n_cand = cand_np.shape[0]
        if n_cand == 0:
            return candidates, {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "rejection_rate": 0.0,
            }

        intruding_mask = self.check_intrusion(cand_np)
        accepted_mask = ~intruding_mask

        n_rejected = int(np.sum(intruding_mask))
        n_accepted = int(np.sum(accepted_mask))
        rejection_rate = (n_rejected / float(n_cand)) * 100.0

        self.total_candidates_evaluated += n_cand
        self.total_candidates_rejected += n_rejected

        purged_np = cand_np[accepted_mask]

        if is_torch:
            purged = torch.from_numpy(purged_np).to(device=device, dtype=candidates.dtype)
        else:
            purged = purged_np

        stats = {
            "total": n_cand,
            "accepted": n_accepted,
            "rejected": n_rejected,
            "rejection_rate": rejection_rate,
        }
        return purged, stats

    def compute_soft_weights(
        self,
        candidates: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute continuous debiasing weights w(x_tilde) in [0, 1].
        w(x) = max(0, 1 - gamma * sum_{j != c} phi_j(x))

        Args:
            candidates: Candidate points of shape (N, D).

        Returns:
            Weight array/tensor of shape (N,).
        """
        is_torch = isinstance(candidates, torch.Tensor)
        device = candidates.device if is_torch else None

        if is_torch:
            cand_np = candidates.detach().cpu().numpy()
        else:
            cand_np = np.asarray(candidates, dtype=np.float64)

        n_cand = cand_np.shape[0]
        if n_cand == 0 or len(self.peer_sketches) == 0:
            weights_np = np.ones(n_cand, dtype=np.float64)
        else:
            sum_phi = np.zeros(n_cand, dtype=np.float64)
            for sketch in self.peer_sketches.values():
                phi = sketch.continuous_intrusion_weight(cand_np)
                sum_phi += phi

            weights_np = np.maximum(0.0, 1.0 - self.gamma * sum_phi)

        if is_torch:
            return torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)
        return weights_np

    def get_cumulative_rejection_rate(self) -> float:
        """Return cumulative percentage of candidates rejected so far."""
        if self.total_candidates_evaluated == 0:
            return 0.0
        return (self.total_candidates_rejected / float(self.total_candidates_evaluated)) * 100.0


class SubspaceNegativeGenerator:
    """
    Subspace Pseudo-Negative Generator for One-Class LUNAR.

    Generates pseudo-anomalies around normal anchors via orthogonal/subspace perturbations,
    and coordinates with CMNPFilter to purge cross-manifold intrusive points.

    Args:
        negative_ratio: Ratio of pseudo-negatives to generate per normal sample (default: 1.0).
        sigma_pert: Perturbation standard deviation / radius (default: 0.1).
        sigma_parallel: In-subspace perturbation standard deviation (default: 0.01).
        cmnp_filter: Optional CMNPFilter instance to filter candidates.
        subspace_U: Optional top-r principal subspace matrix of shape (D, r).
        mode: Generation mode ('subspace', 'isotropic', or 'hypersphere').
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        negative_ratio: float = 1.0,
        sigma_pert: float = 0.1,
        sigma_parallel: float = 0.01,
        cmnp_filter: Optional[CMNPFilter] = None,
        subspace_U: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: str = "subspace",
        multi_scale: bool = False,
        scales: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        self.negative_ratio = max(0.1, float(negative_ratio))
        self.sigma_pert = float(sigma_pert)
        self.sigma_parallel = float(sigma_parallel)
        self.cmnp_filter = cmnp_filter
        self.mode = mode.lower()
        self.multi_scale = multi_scale
        self.scales = scales if scales is not None else [0.2, 0.5, 1.5, 3.0, 6.0]
        self.rng = np.random.default_rng(seed)

        self.subspace_U: Optional[np.ndarray] = None
        if subspace_U is not None:
            self.set_subspace(subspace_U)

    def set_subspace(self, U: Union[np.ndarray, torch.Tensor]) -> None:
        """Set local principal subspace basis U of shape (D, r)."""
        if isinstance(U, torch.Tensor):
            self.subspace_U = U.detach().cpu().numpy().astype(np.float64)
        else:
            self.subspace_U = np.asarray(U, dtype=np.float64)

    def set_cmnp_filter(self, cmnp_filter: CMNPFilter) -> None:
        """Set or update CMNP filter."""
        self.cmnp_filter = cmnp_filter

    def _sample_perturbations(self, n_samples: int, ambient_dim: int) -> np.ndarray:
        """Sample perturbation noise delta in R^{n_samples x ambient_dim}."""
        if self.multi_scale:
            scale_factors = self.rng.choice(self.scales, size=(n_samples, 1)).astype(np.float64)
        else:
            scale_factors = self.sigma_pert

        if self.mode == "subspace" and self.subspace_U is not None:
            # delta = delta_perp + delta_parallel
            # where delta_perp in null space (I - U U^T), delta_parallel in span(U)
            r = self.subspace_U.shape[1]
            xi = self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, ambient_dim))
            # In-subspace projection
            proj = xi @ self.subspace_U  # (n_samples, r)
            recon = proj @ self.subspace_U.T  # (n_samples, ambient_dim)
            perp = xi - recon  # (n_samples, ambient_dim)

            delta = scale_factors * perp + self.sigma_parallel * recon
            return delta

        elif self.mode == "hypersphere":
            # Uniform direction on unit sphere, scaled to radius
            xi = self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, ambient_dim))
            norms = np.linalg.norm(xi, axis=1, keepdims=True) + 1e-12
            unit_dirs = xi / norms
            if self.multi_scale:
                radii = scale_factors
            else:
                radii = self.rng.uniform(0.5 * self.sigma_pert, 1.5 * self.sigma_pert, size=(n_samples, 1))
            return unit_dirs * radii

        else:
            # Isotropic Gaussian perturbation (Standard Naive baseline)
            if self.multi_scale:
                xi = self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, ambient_dim))
                return scale_factors * xi
            else:
                delta = self.rng.normal(loc=0.0, scale=self.sigma_pert, size=(n_samples, ambient_dim))
                return delta

    def _fallback_boundary_noise(self, X_norm: np.ndarray, count: int) -> np.ndarray:
        """Generate high-radius boundary noise when all candidates get purged."""
        ambient_dim = X_norm.shape[1]
        n_norm = X_norm.shape[0]
        if count == n_norm:
            anchors = X_norm
        elif count < n_norm:
            anchors = X_norm[:count]
        else:
            indices = np.tile(np.arange(n_norm), int(np.ceil(count / n_norm)))[:count]
            anchors = X_norm[indices]

        # Use large-scale perturbation (2.5x sigma) to escape dense regions
        directions = self.rng.normal(loc=0.0, scale=1.0, size=(count, ambient_dim))
        norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        unit_dirs = directions / norms
        radii = self.rng.uniform(2.0 * self.sigma_pert, 3.5 * self.sigma_pert, size=(count, 1))

        return anchors + unit_dirs * radii

    def generate(
        self,
        X_normal: Union[np.ndarray, torch.Tensor],
        negative_ratio: Optional[float] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Generate pseudo-negatives from normal anchor samples.

        Args:
            X_normal: Normal samples of shape (N, D).
            negative_ratio: Optional override for negative generation ratio.

        Returns:
            Tuple of (X_neg, stats_dict).
        """
        is_torch = isinstance(X_normal, torch.Tensor)
        device = X_normal.device if is_torch else None

        if is_torch:
            X_norm_np = X_normal.detach().cpu().numpy()
        else:
            X_norm_np = np.asarray(X_normal, dtype=np.float64)

        n_normal, ambient_dim = X_norm_np.shape
        ratio = self.negative_ratio if negative_ratio is None else float(negative_ratio)
        target_count = max(1, int(round(n_normal * ratio)))

        # Subsample anchors if target_count != n_normal
        if target_count == n_normal:
            anchors = X_norm_np
        else:
            anchor_indices = self.rng.choice(n_normal, size=target_count, replace=(target_count > n_normal))
            anchors = X_norm_np[anchor_indices]

        # Sample perturbations
        perturbations = self._sample_perturbations(target_count, ambient_dim)
        candidates = anchors + perturbations

        # Apply CMNP Filter if active
        if self.cmnp_filter is not None:
            purged_negatives, stats = self.cmnp_filter.filter_candidates(candidates)
            if isinstance(purged_negatives, torch.Tensor):
                purged_neg_np = purged_negatives.detach().cpu().numpy()
            else:
                purged_neg_np = purged_negatives

            # Fallback if all candidates are purged
            if purged_neg_np.shape[0] == 0:
                fallback_neg_np = self._fallback_boundary_noise(X_norm_np, target_count)
                stats["fallback_used"] = True
                stats["final_count"] = fallback_neg_np.shape[0]
                result_np = fallback_neg_np
            else:
                stats["fallback_used"] = False
                stats["final_count"] = purged_neg_np.shape[0]
                result_np = purged_neg_np
        else:
            result_np = candidates
            stats = {
                "total": target_count,
                "accepted": target_count,
                "rejected": 0,
                "rejection_rate": 0.0,
                "fallback_used": False,
                "final_count": target_count,
            }

        if is_torch:
            result_tensor = torch.from_numpy(result_np).to(device=device, dtype=X_normal.dtype)
            return result_tensor, stats

        return result_np, stats

    def __call__(
        self,
        X_normal: Union[np.ndarray, torch.Tensor],
        negative_ratio: Optional[float] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        return self.generate(X_normal, negative_ratio=negative_ratio)


# Backwards-compatible alias matching contract in PROJECT.md:
NegativeGenerator = SubspaceNegativeGenerator
