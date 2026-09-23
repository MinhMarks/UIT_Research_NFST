"""
SimpleAutoEncoder Architecture for Tabular IoT Anomaly Detection (Tier 2 Baseline).

Reference:
    Standard Deep AutoEncoder baseline for Federated One-Class NIDS,
    compatible with baseline_model/dasvdd_wrapper.py.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoEncoder(nn.Module):
    """
    Symmetric Deep AutoEncoder for Tabular IoT Telemetry.

    Maps input features x in R^D to a compressed latent code z in R^{code_size},
    then reconstructs x_hat in R^D. Anomaly score is measured via reconstruction error
    ||x - x_hat||_2^2.

    Args:
        input_dim: Ambient dimension D of input features.
        code_size: Dimension of latent bottleneck (default: 32).
        hidden1: Dimension of first hidden layer (default: max(64, input_dim // 2)).
        hidden2: Dimension of second hidden layer (default: max(32, input_dim // 4)).
        reconstruction_activation: Activation on reconstructed output ('sigmoid', 'none', default: 'none').
        negative_slope: Negative slope for LeakyReLU activations (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        code_size: int = 32,
        hidden1: Optional[int] = None,
        hidden2: Optional[int] = None,
        reconstruction_activation: Optional[str] = "none",
        negative_slope: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        
        self.input_dim = input_dim
        self.code_size = code_size
        self.negative_slope = negative_slope
        self.reconstruction_activation = (
            reconstruction_activation.lower() if reconstruction_activation else "none"
        )

        h1 = hidden1 if hidden1 is not None else max(64, input_dim // 2)
        h2 = hidden2 if hidden2 is not None else max(32, input_dim // 4)
        self.hidden1 = h1
        self.hidden2 = h2

        # Encoder
        self.encoder_layer1 = nn.Linear(input_dim, h1)
        self.encoder_layer2 = nn.Linear(h1, h2)
        self.encoder_code = nn.Linear(h2, code_size)

        # Decoder
        self.decoder_layer1 = nn.Linear(code_size, h2)
        self.decoder_layer2 = nn.Linear(h2, h1)
        self.decoder_out = nn.Linear(h1, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features to bottleneck code representation."""
        h = F.leaky_relu(self.encoder_layer1(x), negative_slope=self.negative_slope)
        h = F.leaky_relu(self.encoder_layer2(h), negative_slope=self.negative_slope)
        code = F.leaky_relu(self.encoder_code(h), negative_slope=self.negative_slope)
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from bottleneck code representation."""
        h = F.leaky_relu(self.decoder_layer1(code), negative_slope=self.negative_slope)
        h = F.leaky_relu(self.decoder_layer2(h), negative_slope=self.negative_slope)
        out = self.decoder_out(h)

        if self.reconstruction_activation == "sigmoid":
            return torch.sigmoid(out)
        return out

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            features: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (reconstructed, code).
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def reconstruction_error(
        self,
        features: Union[np.ndarray, torch.Tensor],
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Compute sample-wise Mean Squared Error reconstruction loss.
        Higher error signifies higher anomaly likelihood.

        Args:
            features: Input features of shape (N, input_dim).
            reduction: 'none' (returns (N,) errors) or 'mean' (scalar mean).

        Returns:
            Tensor of reconstruction errors.
        """
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float()
        else:
            x = features.float()

        device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            diff_sq = (x - reconstructed) ** 2
            errors = torch.mean(diff_sq, dim=-1)

        if reduction == "mean":
            return errors.mean()
        return errors

    def predict_score(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict anomaly scores (sample-wise MSE reconstruction errors)."""
        errors = self.reconstruction_error(features, reduction="none")
        return errors.cpu().numpy()
