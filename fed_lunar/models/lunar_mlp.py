"""
LUNAR: Learnable Unified Neighborhood-based Anomaly Ranking MLP
and k-NN Distance-Ranking Feature Extractor.

Reference:
    Goodge et al., "LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks",
    AAAI 2022.
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUNAR_MLP(nn.Module):
    """
    Distance-Ranking Multi-Layer Perceptron for LUNAR.
    
    Maps ordered k-NN Euclidean distance vectors d(z) in R^k to scalar anomaly logits
    and probabilities in [0, 1].

    Args:
        k: Neighborhood size (number of nearest neighbor distances).
        hidden_dims: List of hidden layer dimensions (default: [64, 32, 16]).
        dropout: Dropout probability between hidden layers (default: 0.1).
        activation: Activation function name ('leaky_relu', 'relu', 'elu').
        negative_slope: Negative slope for LeakyReLU (default: 0.1).
    """

    def __init__(
        self,
        k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "leaky_relu",
        negative_slope: float = 0.1,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        self.k = k
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32, 16]
        self.dropout_rate = dropout
        self.negative_slope = negative_slope
        self.activation_name = activation.lower()

        # Build network layers
        layers: List[nn.Module] = []
        prev_dim = k
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if self.activation_name == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            elif self.activation_name == "relu":
                layers.append(nn.ReLU())
            elif self.activation_name == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim

        # Output layer produces a scalar logit
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for LeakyReLU/ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing scalar anomaly logits.

        Args:
            d: Tensor of sorted k-NN Euclidean distances of shape (batch_size, k).

        Returns:
            Logit tensor of shape (batch_size, 1).
        """
        if d.dim() == 1:
            d = d.unsqueeze(0)
        if d.size(-1) != self.k:
            raise ValueError(f"Expected input with {self.k} features, but got shape {tuple(d.shape)}")
        
        features = self.feature_extractor(d)
        logits = self.output_layer(features)
        return logits

    def predict_proba(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted anomaly probability \\hat{y} in [0, 1].

        Args:
            d: Tensor of sorted k-NN Euclidean distances of shape (batch_size, k).

        Returns:
            Probability tensor of shape (batch_size, 1).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(d)
            probs = torch.sigmoid(logits)
        if was_training:
            self.train()
        return probs

    def predict_score(self, d: torch.Tensor) -> torch.Tensor:
        """
        Alias for predict_proba: anomaly score in [0, 1]. Higher indicates higher anomaly likelihood.
        """
        return self.predict_proba(d)


class KNNDistanceExtractor:
    """
    Extracts ordered k-nearest-neighbor Euclidean distance vectors against a reference dictionary.

    Args:
        reference_data: Reference normal dataset D_c of shape (N_ref, D).
        device: Torch device ('cpu' or 'cuda').
        batch_size: Batch size for distance computation to manage memory (default: 1024).
    """

    def __init__(
        self,
        reference_data: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 1024,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(reference_data, np.ndarray):
            self.reference_data = torch.from_numpy(reference_data).float()
        elif isinstance(reference_data, torch.Tensor):
            self.reference_data = reference_data.float()
        else:
            raise TypeError("reference_data must be a numpy.ndarray or torch.Tensor")

        if self.reference_data.dim() != 2:
            raise ValueError(f"reference_data must be 2D, got shape {tuple(self.reference_data.shape)}")

        self.num_samples, self.feature_dim = self.reference_data.shape
        self.batch_size = batch_size

    def to(self, device: Union[str, torch.device]) -> "KNNDistanceExtractor":
        """Move reference tensor to specified device."""
        self.device = torch.device(device)
        return self

    def extract_distances(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        k: int = 10,
        is_reference_member: bool = False,
    ) -> torch.Tensor:
        """
        Compute sorted Euclidean distances to k nearest neighbors in the reference set.

        Args:
            queries: Query points of shape (N_query, D).
            k: Number of nearest neighbors to retrieve.
            is_reference_member: True if queries are drawn from reference_data (omits self-distance at index 0).

        Returns:
            Tensor of sorted distances of shape (N_query, k) where
            0 <= d_1 <= d_2 <= ... <= d_k.
        """
        if k > self.num_samples:
            raise ValueError(
                f"Requested k={k} exceeds reference size {self.num_samples}"
            )
        if is_reference_member and k >= self.num_samples:
            raise ValueError(
                f"For reference members, k={k} must be strictly less than reference size {self.num_samples}"
            )

        if isinstance(queries, np.ndarray):
            queries_tensor = torch.from_numpy(queries).float()
        else:
            queries_tensor = queries.float()

        if queries_tensor.dim() == 1:
            queries_tensor = queries_tensor.unsqueeze(0)

        num_queries = queries_tensor.size(0)
        k_retrieve = k + 1 if is_reference_member else k

        ref_tensor = self.reference_data.to(self.device)
        all_distances: List[torch.Tensor] = []

        # Process queries in batches to bound peak memory
        for start_idx in range(0, num_queries, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_queries)
            q_batch = queries_tensor[start_idx:end_idx].to(self.device)

            # Efficient pairwise Euclidean distance: ||q - r||_2
            # torch.cdist computes exact Euclidean distances
            dist_batch = torch.cdist(q_batch, ref_tensor, p=2.0)

            # Retrieve top k_retrieve smallest distances, sorted ascending
            # topk with largest=False returns smallest values
            topk_dists, _ = torch.topk(dist_batch, k=k_retrieve, dim=1, largest=False, sorted=True)

            if is_reference_member:
                # Omit index 0 (self-distance = 0) and take indices 1 to k
                result = topk_dists[:, 1 : k + 1]
            else:
                result = topk_dists[:, :k]

            all_distances.append(result.cpu())

        return torch.cat(all_distances, dim=0)

    def __call__(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        k: int = 10,
        is_reference_member: bool = False,
    ) -> torch.Tensor:
        return self.extract_distances(queries, k=k, is_reference_member=is_reference_member)


class LunarDistanceRankingLoss(nn.Module):
    """
    Binary Cross-Entropy Distance-Ranking Loss for LUNAR.
    
    Evaluates:
        L = - 1/N_norm * sum log(1 - sigma(logits_norm))
            - lambda_anom / N_anom * sum w_i * log(sigma(logits_anom))
    
    With optional sample-wise continuous debiased weights w_i for pseudo-negatives.
    """

    def __init__(self, lambda_anom: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.lambda_anom = lambda_anom
        self.eps = eps

    def forward(
        self,
        logits_norm: torch.Tensor,
        logits_anom: torch.Tensor,
        weights_anom: Optional[torch.Tensor] = None,
        lambda_anom: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute total distance-ranking loss.

        Args:
            logits_norm: Anomaly logits for normal points (target y = 0), shape (N_norm, 1) or (N_norm,).
            logits_anom: Anomaly logits for pseudo-negative points (target y = 1), shape (N_anom, 1) or (N_anom,).
            weights_anom: Optional debiasing weights w(x_tilde) in [0, 1] of shape (N_anom,) or (N_anom, 1).
            lambda_anom: Optional override for anomaly loss weight factor.

        Returns:
            Scalar loss tensor.
        """
        lam = self.lambda_anom if lambda_anom is None else lambda_anom

        # Ensure flat or (N, 1) consistency
        if logits_norm.dim() > 1:
            logits_norm = logits_norm.view(-1)
        if logits_anom.dim() > 1:
            logits_anom = logits_anom.view(-1)

        # Normal loss: BCE against zeros (y=0)
        # Using binary_cross_entropy_with_logits for maximum numerical stability
        targets_norm = torch.zeros_like(logits_norm)
        loss_norm = F.binary_cross_entropy_with_logits(logits_norm, targets_norm, reduction="mean")

        # Anomaly loss: BCE against ones (y=1)
        targets_anom = torch.ones_like(logits_anom)
        if weights_anom is not None:
            if weights_anom.dim() > 1:
                weights_anom = weights_anom.view(-1)
            weights_anom = weights_anom.to(logits_anom.device)
            # Weighted average
            weight_sum = weights_anom.sum() + self.eps
            bce_elementwise = F.binary_cross_entropy_with_logits(
                logits_anom, targets_anom, reduction="none"
            )
            loss_anom = (weights_anom * bce_elementwise).sum() / weight_sum
        else:
            loss_anom = F.binary_cross_entropy_with_logits(logits_anom, targets_anom, reduction="mean")

        return loss_norm + lam * loss_anom
