"""
Tier 2 Baseline: Federated Deep AutoEncoder (Feature F6).

Federated Deep Autoencoder baseline using SimpleAutoEncoder.
Trained exclusively with MSE reconstruction loss on normal client data,
aggregated via standard FedAvg across communication rounds.

Reference:
    - Sakurada & Yairi, "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction", MLSDA 2014.
    - Milestone M2 Specification (Feature F6).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fed_lunar.models.autoencoder import SimpleAutoEncoder


class FedAutoEncoder:
    """
    Tier 2 Baseline: Federated Deep AutoEncoder (Fed-AE).

    Trains a symmetric deep autoencoder across decentralized client nodes using
    MSE reconstruction loss on strictly normal telemetry, aggregating weights via FedAvg.
    At test time, anomaly scores are computed as the sample-wise reconstruction error:
        score(x) = ||x - \\hat{x}||_2^2

    Args:
        code_size: Bottleneck latent dimension (default: 32).
        hidden1: Dimension of first encoder hidden layer (default: None, auto-calculated).
        hidden2: Dimension of second encoder hidden layer (default: None, auto-calculated).
        lr: Local learning rate (default: 0.001).
        batch_size: Local batch size (default: 128).
        local_epochs: Number of local training epochs per round (default: 3).
        weight_decay: L2 parameter regularization (default: 1e-4).
        device: Torch device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        code_size: int = 32,
        hidden1: Optional[int] = None,
        hidden2: Optional[int] = None,
        lr: float = 0.001,
        batch_size: int = 128,
        local_epochs: int = 3,
        weight_decay: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        self.code_size = code_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.global_model: Optional[SimpleAutoEncoder] = None
        self.input_dim: Optional[int] = None
        self.max_train_score: float = 1.0
        self.history: List[Dict[str, Any]] = []

    def _format_client_data(
        self,
        client_train_data: Union[List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray], np.ndarray],
    ) -> List[np.ndarray]:
        """Normalize client data input to a list of float32 numpy arrays."""
        if isinstance(client_train_data, np.ndarray):
            return [client_train_data.astype(np.float32)]
        elif isinstance(client_train_data, dict):
            return [np.asarray(data, dtype=np.float32) for data in client_train_data.values()]
        elif isinstance(client_train_data, (list, tuple)):
            return [np.asarray(data, dtype=np.float32) for data in client_train_data]
        else:
            raise TypeError(
                f"Unsupported client_train_data type: {type(client_train_data)}. Expected list, tuple, dict, or ndarray."
            )

    def fit(
        self,
        client_train_data: Union[List[np.ndarray], Dict[Any, np.ndarray], np.ndarray],
        rounds: int = 10,
        verbose: bool = False,
    ) -> "FedAutoEncoder":
        """
        Train Fed-AE across decentralized clients.

        Args:
            client_train_data: Decentralized normal training data across M clients.
            rounds: Number of federated communication rounds (default: 10).
            verbose: Whether to log round training progress.

        Returns:
            self (fitted instance).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        client_datasets = self._format_client_data(client_train_data)
        M = len(client_datasets)
        if M == 0:
            raise ValueError("client_train_data cannot be empty")

        self.input_dim = client_datasets[0].shape[1]
        client_sizes = [len(d) for d in client_datasets]
        total_samples = sum(client_sizes)
        client_weights = [n / total_samples for n in client_sizes]

        # Initialize global autoencoder
        self.global_model = SimpleAutoEncoder(
            input_dim=self.input_dim,
            code_size=self.code_size,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
        ).to(self.device)

        self.history = []

        for r in range(1, rounds + 1):
            client_state_dicts = []
            client_losses = []

            global_state = self.global_model.state_dict()

            for i in range(M):
                client_model = SimpleAutoEncoder(
                    input_dim=self.input_dim,
                    code_size=self.code_size,
                    hidden1=self.hidden1,
                    hidden2=self.hidden2,
                ).to(self.device)
                client_model.load_state_dict(global_state)
                client_model.train()

                optimizer = torch.optim.Adam(
                    client_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

                X_norm = client_datasets[i]
                n_samples = client_sizes[i]
                batch_size = min(self.batch_size, n_samples)

                epoch_losses = []
                for _ in range(self.local_epochs):
                    perm = np.random.permutation(n_samples)
                    for start_idx in range(0, n_samples, batch_size):
                        batch_idx = perm[start_idx : start_idx + batch_size]
                        x_batch = torch.from_numpy(X_norm[batch_idx]).to(self.device)

                        recon, _ = client_model(x_batch)
                        loss = F.mse_loss(recon, x_batch)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_losses.append(loss.item())

                client_state_dicts.append(client_model.state_dict())
                client_losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

            # FedAvg aggregation of encoder/decoder weights
            new_global_state = {}
            for key in global_state.keys():
                new_global_state[key] = sum(
                    client_weights[i] * client_state_dicts[i][key].float() for i in range(M)
                )

            self.global_model.load_state_dict(new_global_state)

            round_summary = {
                "round": r,
                "mean_mse_loss": float(np.mean(client_losses)),
                "client_losses": client_losses,
            }
            self.history.append(round_summary)

            if verbose:
                print(f"[FedAutoEncoder] Round {r}/{rounds} - Mean MSE Loss: {round_summary['mean_mse_loss']:.6f}")

        # Compute calibration score on training data for probability normalization
        train_scores: List[np.ndarray] = []
        for X_norm in client_datasets:
            sc = self.decision_function(X_norm)
            train_scores.append(sc)
        all_train_scores = np.concatenate(train_scores, axis=0) if train_scores else np.array([1.0])
        # Use 99th percentile to prevent extreme outlier distortion during min-max calibration
        self.max_train_score = float(np.percentile(all_train_scores, 99.0))
        if self.max_train_score <= 1e-12:
            self.max_train_score = float(np.max(all_train_scores)) + 1e-6

        return self

    def decision_function(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Compute anomaly score for test samples as sample-wise MSE reconstruction error:
            score(x) = ||x - \\hat{x}||_2^2

        Args:
            X: Input samples of shape (N, D).
            batch_size: Mini-batch size for model evaluation.

        Returns:
            1D numpy array of shape (N,) containing reconstruction error anomaly scores.
        """
        if self.global_model is None:
            raise RuntimeError("Model must be fitted before calling decision_function")

        self.global_model.eval()
        if isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))

        N = X_tensor.shape[0]
        scores: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_x = X_tensor[i : i + batch_size].to(self.device)
                recon, _ = self.global_model(batch_x)
                # Sample-wise squared Euclidean reconstruction error ||x - x_hat||_2^2
                err = torch.sum((batch_x - recon) ** 2, dim=-1)
                scores.append(err.cpu().numpy().ravel())

        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    score = decision_function

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Predict class probabilities [P(normal), P(anomaly)] using calibrated reconstruction error.

        Args:
            X: Input samples of shape (N, D).
            batch_size: Mini-batch size.

        Returns:
            2D numpy array of shape (N, 2).
        """
        scores = self.decision_function(X, batch_size=batch_size)
        # Normalize with respect to normal training envelope
        norm_scores = np.clip(scores / (self.max_train_score + 1e-12), 0.0, 1.0)
        proba = np.zeros((len(norm_scores), 2), dtype=np.float32)
        proba[:, 1] = norm_scores
        proba[:, 0] = 1.0 - norm_scores
        return proba

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Predict binary labels (0 = normal, 1 = anomaly).
        If threshold is None, uses calibrated training envelope max_train_score.
        """
        scores = self.decision_function(X, batch_size=batch_size)
        thresh = self.max_train_score if threshold is None else threshold
        return (scores >= thresh).astype(int)
