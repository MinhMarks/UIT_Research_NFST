"""
Distance-Ranking Orthogonal Gradient Alignment (DROGA).

Implements server-side gradient surgery strategies for Federated LUNAR:
1. DR-PCGrad: Pairwise orthogonal projection of conflicting client gradients.
2. DR-CAGrad: Conflict-Averse Gradient descent solving the dual simplex QP.
3. Gradient Conflict Metrics: Exact computation of pairwise cosine similarities and GCR.

Reference:
    - Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020 (PCGrad).
    - Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning", NeurIPS 2021 (CAGrad).
    - Explorer 3 Report (Mathematical Foundations), Section 5 & Algorithm 2.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn


def flatten_tensor_list(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of parameter tensors into a single 1D tensor."""
    return torch.cat([t.view(-1) for t in tensors])


def unflatten_to_tensor_list(
    flat: torch.Tensor,
    reference_shapes: List[torch.Size],
) -> List[torch.Tensor]:
    """Unflatten a 1D tensor back into a list of tensors matching reference shapes."""
    result: List[torch.Tensor] = []
    offset = 0
    for shape in reference_shapes:
        numel = shape.numel()
        result.append(flat[offset : offset + numel].view(shape))
        offset += numel
    return result


def compute_gradient_conflict_metrics(
    gradients: List[Union[torch.Tensor, np.ndarray]],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute pairwise cosine similarities and exact Gradient Conflict Ratio (GCR).

    Args:
        gradients: List of M client gradient vectors in R^P.
        eps: Small constant to prevent division by zero.

    Returns:
        Dictionary containing:
            - 'cosine_matrix': M x M matrix of pairwise cosine similarities in [-1, 1].
            - 'gcr': Gradient conflict ratio in [0, 1] (% pairs with cos < 0).
            - 'gcr_percent': GCR expressed as percentage in [0, 100].
            - 'conflicting_pairs': Count of pairs (i < j) with cos < 0.
            - 'total_pairs': Total client pairs M * (M - 1) / 2.
            - 'mean_cosine': Average pairwise cosine similarity (i < j).
            - 'min_cosine': Minimum pairwise cosine similarity.
            - 'gram_matrix': M x M inner product matrix G_ij = <g_i, g_j>.
    """
    M = len(gradients)
    if M == 0:
        raise ValueError("Gradients list cannot be empty")

    # Convert to 2D numpy array of shape (M, P)
    grads_list = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            grads_list.append(g.detach().cpu().float().numpy().ravel())
        else:
            grads_list.append(np.asarray(g, dtype=np.float64).ravel())
    G_mat = np.stack(grads_list, axis=0)  # (M, P)

    # Compute Gram matrix G_ij = <g_i, g_j>
    gram = G_mat @ G_mat.T  # (M, M)

    # Norms
    norms = np.sqrt(np.maximum(0.0, np.diag(gram)))  # (M,)
    norm_outer = np.outer(norms, norms) + eps

    # Cosine matrix
    cosine_matrix = np.clip(gram / norm_outer, -1.0, 1.0)
    np.fill_diagonal(cosine_matrix, 1.0)

    if M < 2:
        return {
            "cosine_matrix": cosine_matrix,
            "gcr": 0.0,
            "gcr_percent": 0.0,
            "conflicting_pairs": 0,
            "total_pairs": 0,
            "mean_cosine": 1.0,
            "min_cosine": 1.0,
            "gram_matrix": gram,
        }

    # Extract upper triangular indices (i < j)
    triu_indices = np.triu_indices(M, k=1)
    pair_cosines = cosine_matrix[triu_indices]
    total_pairs = len(pair_cosines)

    conflicting_pairs = int(np.sum(pair_cosines < 0.0))
    gcr = float(conflicting_pairs) / float(total_pairs)
    gcr_percent = gcr * 100.0

    return {
        "cosine_matrix": cosine_matrix,
        "gcr": gcr,
        "gcr_percent": gcr_percent,
        "conflicting_pairs": conflicting_pairs,
        "total_pairs": total_pairs,
        "mean_cosine": float(np.mean(pair_cosines)),
        "min_cosine": float(np.min(pair_cosines)),
        "gram_matrix": gram,
    }


def dr_pcgrad(
    gradients: List[Union[torch.Tensor, np.ndarray]],
    weights: Optional[List[float]] = None,
    seed: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Distance-Ranking PCGrad (DR-PCGrad).

    Projects conflicting client gradients onto orthogonal half-spaces:
    If <g_i, g_j> < 0:
        g_i = g_i - (<g_i, g_j> / ||g_j||^2) * g_j

    Args:
        gradients: List of M client gradient vectors in R^P.
        weights: Optional aggregation weights for clients (default: uniform 1/M).
        seed: Random seed for peer client permutations.
        eps: Small constant to avoid zero division.

    Returns:
        Aligned aggregated gradient vector in R^P as torch.Tensor.
    """
    M = len(gradients)
    if M == 0:
        raise ValueError("Gradients list cannot be empty")
    if M == 1:
        g = gradients[0]
        return g if isinstance(g, torch.Tensor) else torch.from_numpy(g).float()

    # Convert to torch float tensors
    torch_grads: List[torch.Tensor] = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            torch_grads.append(g.detach().float().clone().view(-1))
        else:
            torch_grads.append(torch.from_numpy(np.asarray(g, dtype=np.float32)).view(-1))

    # Normalized weights
    if weights is None:
        w = [1.0 / float(M)] * M
    else:
        w_sum = sum(weights)
        w = [float(val) / w_sum for val in weights]

    rng = np.random.default_rng(seed)

    # Unit-norm gradient scaling \tilde{g}_i = g_i / (||g_i|| + eps) to resolve scale-disparity distortion
    raw_norms = [float(torch.norm(g).item()) for g in torch_grads]
    torch_grads_unit = [
        torch_grads[i] / (raw_norms[i] + eps) if raw_norms[i] > eps else torch.zeros_like(torch_grads[i])
        for i in range(M)
    ]
    mean_norm = float(np.sum(np.array(w) * np.array(raw_norms)))

    # Initialize projected gradients from unit-norm gradients
    projected_grads = [g.clone() for g in torch_grads_unit]

    for i in range(M):
        # Random permutation of peer clients
        peers = [j for j in range(M) if j != i]
        rng.shuffle(peers)

        for j in peers:
            inner_prod = torch.dot(projected_grads[i], torch_grads_unit[j])
            if inner_prod < 0.0:
                norm_sq = torch.dot(torch_grads_unit[j], torch_grads_unit[j]) + eps
                projected_grads[i] = projected_grads[i] - (inner_prod / norm_sq) * torch_grads_unit[j]

    # Weighted sum of projected unit gradients
    g_aligned_unit = torch.zeros_like(projected_grads[0])
    for i in range(M):
        g_aligned_unit += w[i] * projected_grads[i]

    return g_aligned_unit * mean_norm


def dr_cagrad(
    gradients: List[Union[torch.Tensor, np.ndarray]],
    c_param: float = 0.4,
    weights: Optional[List[float]] = None,
    mode: str = "dual_simplex_qp",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Distance-Ranking CAGrad (DR-CAGrad).

    Solves the dual simplex QP:
        min_alpha 1/2 ||g_0 + sum_i alpha_i g_i||^2
        subject to alpha >= 0, sum_i alpha_i = c * ||g_0|| / max_i ||g_i||

    Incorporates unit-norm gradient scaling \\tilde{g}_i = g_i / (||g_i|| + eps)
    prior to Gram matrix computation to resolve scale-disparity distortion.

    Args:
        gradients: List of M client gradient vectors in R^P.
        c_param: Conflict-aversion coefficient c in [0, 1) (default: 0.4).
        weights: Client sample weights for base average g_0 (default: uniform 1/M).
        mode: Optimization mode ('dual_simplex_qp' per prompt spec, or 'minimax_simplex').
        eps: Small constant to avoid zero division.

    Returns:
        Aligned conflict-averse gradient vector in R^P as torch.Tensor.
    """
    M = len(gradients)
    if M == 0:
        raise ValueError("Gradients list cannot be empty")
    if M == 1:
        g = gradients[0]
        return g if isinstance(g, torch.Tensor) else torch.from_numpy(g).float()

    # Convert to torch float tensors
    torch_grads: List[torch.Tensor] = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            torch_grads.append(g.detach().float().clone().view(-1))
        else:
            torch_grads.append(torch.from_numpy(np.asarray(g, dtype=np.float32)).view(-1))

    # Base weights w0
    if weights is None:
        w0 = np.array([1.0 / float(M)] * M, dtype=np.float64)
    else:
        w_sum = sum(weights)
        w0 = np.array([float(val) / w_sum for val in weights], dtype=np.float64)

    # Base average gradient g_0
    g_0 = torch.zeros_like(torch_grads[0])
    for i in range(M):
        g_0 += float(w0[i]) * torch_grads[i]

    # If c_param is 0, CAGrad recovers standard weighted average g_0
    if c_param <= 0.0:
        return g_0

    # Unit-norm gradient scaling \tilde{g}_i = g_i / (||g_i|| + eps) prior to Gram matrix computation
    raw_norms = [float(torch.norm(g).item()) for g in torch_grads]
    torch_grads_unit = [
        torch_grads[i] / (raw_norms[i] + eps) if raw_norms[i] > eps else torch.zeros_like(torch_grads[i])
        for i in range(M)
    ]
    mean_norm = float(np.sum(w0 * np.array(raw_norms)))

    # Base normalized average gradient \tilde{g}_0
    g_0_unit = torch.zeros_like(torch_grads[0])
    for i in range(M):
        g_0_unit += float(w0[i]) * torch_grads_unit[i]

    # Normalized Gram matrix \tilde{G}_ij = <\tilde{g}_i, \tilde{g}_j>
    G_mat_unit = np.stack([g.cpu().numpy().astype(np.float64) for g in torch_grads_unit], axis=0)  # (M, P)
    gram = G_mat_unit @ G_mat_unit.T  # (M, M)

    # Norm of g_0_unit and client normalized gradients
    norm_g0 = float(np.sqrt(max(0.0, float(w0.T @ gram @ w0))))
    client_norms = np.sqrt(np.maximum(0.0, np.diag(gram)))
    max_norm = float(np.max(client_norms))

    if norm_g0 < eps or max_norm < eps:
        return g_0

    if mode == "dual_simplex_qp":
        # Prompt specification on unit-norm gradients:
        # min_alpha 1/2 ||\tilde{g}_0 + sum_i alpha_i \tilde{g}_i||^2
        # s.t. alpha >= 0, sum_i alpha_i = phi = c * ||\tilde{g}_0|| / max_i ||\tilde{g}_i||
        phi = float(c_param * (norm_g0 / (max_norm + eps)))

        # Expanding 1/2 ||\tilde{g}_0 + G_mat_unit^T alpha||^2:
        # = 1/2 ||\tilde{g}_0||^2 + alpha^T (gram @ w0) + 1/2 alpha^T gram alpha
        G_w0 = gram @ w0  # (M,)

        def objective(alpha: np.ndarray) -> Tuple[float, np.ndarray]:
            val = float(0.5 * (alpha.T @ gram @ alpha) + alpha.T @ G_w0 + 0.5 * (norm_g0 ** 2))
            grad = (gram @ alpha + G_w0).astype(np.float64)
            return val, grad

        # Constraints and bounds
        bounds = [(0.0, phi) for _ in range(M)]
        constraints = [{"type": "eq", "fun": lambda a: np.sum(a) - phi, "jac": lambda a: np.ones(M)}]
        init_alpha = np.ones(M, dtype=np.float64) * (phi / float(M))

        res = scipy.optimize.minimize(
            objective,
            init_alpha,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )

        alpha_opt = res.x if res.success else init_alpha
        # Form aligned normalized direction \tilde{g}_aligned = \tilde{g}_0 + sum_i alpha_i * \tilde{g}_i
        g_aligned_unit = g_0_unit.clone()
        for i in range(M):
            g_aligned_unit += float(alpha_opt[i]) * torch_grads_unit[i]

        # Rescale by mean gradient norm to preserve the physical optimization step size
        return g_aligned_unit * mean_norm

    else:
        # Explorer 3 Dual formulation (Liu et al. NeurIPS 2021 Algorithm 2):
        # min_{w in Delta^M} (1/M) w^T G 1_M + c * norm_g0 * sqrt(w^T G w)
        def dual_objective(w: np.ndarray) -> Tuple[float, np.ndarray]:
            quad = float(w.T @ gram @ w)
            sqrt_quad = np.sqrt(max(quad, eps))
            lin = float(w.T @ (gram @ w0))
            val = lin + c_param * norm_g0 * sqrt_quad
            grad = (gram @ w0) + c_param * norm_g0 * (gram @ w) / sqrt_quad
            return val, grad

        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(M)}]
        init_w = np.ones(M, dtype=np.float64) / float(M)

        res = scipy.optimize.minimize(
            dual_objective,
            init_w,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        w_opt = res.x if res.success else init_w
        norm_gw = float(np.sqrt(max(eps, w_opt.T @ gram @ w_opt)))

        scale = (c_param * norm_g0) / norm_gw
        g_aligned_unit = g_0_unit.clone()
        for i in range(M):
            g_aligned_unit += float(scale * w_opt[i]) * torch_grads_unit[i]

        return g_aligned_unit * mean_norm


class DROGAStrategy:
    """
    Distance-Ranking Orthogonal Gradient Alignment (DROGA) Server Strategy.

    Coordinates server-side gradient conflict inspection and alignment across
    participating federated edge clients.

    Args:
        mode: Alignment mode ('CAGrad', 'PCGrad', 'FedAvg').
        c_param: Conflict-aversion coefficient for CAGrad (default: 0.4).
        seed: Random seed for DR-PCGrad client permutations.
    """

    def __init__(
        self,
        mode: str = "CAGrad",
        c_param: float = 0.4,
        seed: Optional[int] = None,
    ):
        self.mode = mode
        self.c_param = c_param
        self.seed = seed
        self.history: List[Dict[str, Any]] = []

    def aggregate(
        self,
        client_gradients: List[Union[torch.Tensor, np.ndarray]],
        client_weights: Optional[List[float]] = None,
        round_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Aggregate client gradients into a single non-conflicting update.

        Args:
            client_gradients: List of M gradient vectors in R^P.
            client_weights: Optional list of M client weights.
            round_idx: Optional communication round index for logging.

        Returns:
            Tuple of (aligned_gradient, metrics_dict).
        """
        M = len(client_gradients)
        if M == 0:
            raise ValueError("No client gradients provided for aggregation")

        # 1. Compute pre-alignment conflict diagnostics
        pre_metrics = compute_gradient_conflict_metrics(client_gradients)

        # 2. Perform alignment based on chosen mode
        mode_upper = self.mode.upper()
        if mode_upper == "CAGRAD" or mode_upper == "DR-CAGRAD":
            g_aligned = dr_cagrad(
                client_gradients,
                c_param=self.c_param,
                weights=client_weights,
            )
        elif mode_upper == "PCGRAD" or mode_upper == "DR-PCGRAD":
            g_aligned = dr_pcgrad(
                client_gradients,
                weights=client_weights,
                seed=self.seed,
            )
        elif mode_upper == "FEDAVG" or mode_upper == "NAIVE":
            # Standard weighted FedAvg without gradient surgery
            if client_weights is None:
                w = [1.0 / float(M)] * M
            else:
                w_sum = sum(client_weights)
                w = [float(v) / w_sum for v in client_weights]

            first_g = client_gradients[0]
            if isinstance(first_g, torch.Tensor):
                g_aligned = torch.zeros_like(first_g.view(-1).float())
                for i, g in enumerate(client_gradients):
                    g_tensor = g.view(-1).float() if isinstance(g, torch.Tensor) else torch.from_numpy(g).view(-1).float()
                    g_aligned += w[i] * g_tensor
            else:
                g_aligned_np = np.zeros_like(np.asarray(first_g, dtype=np.float32).ravel())
                for i, g in enumerate(client_gradients):
                    g_np = g.detach().cpu().numpy().ravel() if isinstance(g, torch.Tensor) else np.asarray(g, dtype=np.float32).ravel()
                    g_aligned_np += w[i] * g_np
                g_aligned = torch.from_numpy(g_aligned_np)
        else:
            raise ValueError(f"Unknown alignment mode: {self.mode}. Choose 'CAGrad', 'PCGrad', or 'FedAvg'.")

        # 3. Compute post-alignment agreement metrics with client gradients
        post_inner_prods: List[float] = []
        post_cosines: List[float] = []
        aligned_norm = float(torch.norm(g_aligned)) + 1e-12

        for g in client_gradients:
            g_t = g.view(-1).float() if isinstance(g, torch.Tensor) else torch.from_numpy(np.asarray(g, dtype=np.float32)).view(-1)
            ip = float(torch.dot(g_aligned, g_t))
            c_norm = float(torch.norm(g_t)) + 1e-12
            cos = ip / (aligned_norm * c_norm)
            post_inner_prods.append(ip)
            post_cosines.append(cos)

        post_conflicts = sum(1 for ip in post_inner_prods if ip < 0.0)

        summary = {
            "round": round_idx,
            "mode": self.mode,
            "pre_gcr": pre_metrics["gcr"],
            "pre_gcr_percent": pre_metrics["gcr_percent"],
            "pre_conflicting_pairs": pre_metrics["conflicting_pairs"],
            "pre_total_pairs": pre_metrics["total_pairs"],
            "pre_mean_cosine": pre_metrics["mean_cosine"],
            "pre_min_cosine": pre_metrics["min_cosine"],
            "post_conflicts_with_clients": post_conflicts,
            "post_min_cosine_with_clients": min(post_cosines) if post_cosines else 1.0,
            "post_mean_cosine_with_clients": float(np.mean(post_cosines)) if post_cosines else 1.0,
            "aligned_norm": aligned_norm,
        }
        self.history.append(summary)

        return g_aligned, summary
