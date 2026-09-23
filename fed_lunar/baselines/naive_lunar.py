"""
Tier 1 Baseline: Naive Federated LUNAR (Feature F5).

Standard FedAvg on LUNAR MLP weights with uncoordinated local subspace perturbation
(NO Cross-Manifold Negative Purging (CMNP), NO DROGA gradient projection).

Reference:
    - Goodge et al., "LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks", AAAI 2022.
    - McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
    - Milestone M2 Specification (Feature F5).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator


class NaiveFedLunar:
    """
    Tier 1 Baseline: Naive Federated LUNAR.

    Executes standard FedAvg parameter averaging across local LUNAR client models.
    Each client independently synthesizes pseudo-negatives via local subspace perturbation
    without CMNP filtering or gradient surgery.

    Args:
        k: Neighborhood size for k-NN distance ranking (default: 10).
        hidden_dims: Architecture of LUNAR MLP hidden layers (default: [64, 32, 16]).
        dropout: Dropout rate in LUNAR MLP (default: 0.1).
        negative_ratio: Ratio of generated pseudo-negatives to normal points (default: 1.0).
        sigma_pert: Perturbation noise scale (default: 0.1).
        lr: Local learning rate (default: 0.001).
        batch_size: Local training batch size (default: 128).
        local_epochs: Number of local training epochs per communication round (default: 3).
        weight_decay: L2 parameter regularization (default: 1e-4).
        device: Torch device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        negative_ratio: float = 1.0,
        sigma_pert: float = 0.1,
        multi_scale: bool = True,
        scales: Optional[List[float]] = None,
        lr: float = 0.001,
        batch_size: int = 128,
        local_epochs: int = 3,
        weight_decay: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        self.k = k
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32, 16]
        self.dropout = dropout
        self.negative_ratio = negative_ratio
        self.sigma_pert = sigma_pert
        self.multi_scale = multi_scale
        self.scales = scales if scales is not None else [0.2, 0.5, 1.5, 3.0, 6.0]
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.global_model: Optional[LUNAR_MLP] = None
        self.reference_data: Optional[np.ndarray] = None
        self.global_knn_extractor: Optional[KNNDistanceExtractor] = None
        self.history: List[Dict[str, Any]] = []

    def _format_client_data(
        self,
        client_train_data: Union[List[np.ndarray], Dict[Any, np.ndarray], np.ndarray],
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
    ) -> "NaiveFedLunar":
        """
        Train Naive Fed-LUNAR across multiple communication rounds.

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

        client_sizes = [len(d) for d in client_datasets]
        total_samples = sum(client_sizes)
        client_weights = [n / total_samples for n in client_sizes]

        # Determine effective k (must not exceed local client dataset sizes)
        min_client_n = min(client_sizes)
        effective_k = min(self.k, max(1, min_client_n - 1))

        # Initialize global model
        self.global_model = LUNAR_MLP(
            k=effective_k,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Build local client components (uncoordinated: no CMNP filter)
        local_extractors = [
            KNNDistanceExtractor(data, device=self.device) for data in client_datasets
        ]
        local_generators = [
            SubspaceNegativeGenerator(
                negative_ratio=self.negative_ratio,
                sigma_pert=self.sigma_pert,
                cmnp_filter=None,  # Naive: NO CMNP filter
                mode="subspace",
                multi_scale=self.multi_scale,
                scales=self.scales,
                seed=self.seed + idx,
            )
            for idx in range(M)
        ]

        loss_fn = LunarDistanceRankingLoss()

        self.history = []

        for r in range(1, rounds + 1):
            client_state_dicts = []
            client_round_losses = []

            # Global weights broadcast
            global_state = self.global_model.state_dict()

            for i in range(M):
                # Instantiate client model with global weights
                client_model = LUNAR_MLP(
                    k=effective_k,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                ).to(self.device)
                client_model.load_state_dict(global_state)
                client_model.train()

                optimizer = torch.optim.Adam(
                    client_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

                X_norm = client_datasets[i]
                n_norm = client_sizes[i]
                batch_size = min(self.batch_size, n_norm)

                losses = []
                for _ in range(self.local_epochs):
                    perm = np.random.permutation(n_norm)
                    for start_idx in range(0, n_norm, batch_size):
                        batch_idx = perm[start_idx : start_idx + batch_size]
                        batch_norm = X_norm[batch_idx]

                        # Uncoordinated pseudo-negative generation
                        batch_anom, _ = local_generators[i].generate(batch_norm)

                        # Extract k-NN distance vectors
                        d_norm = local_extractors[i](
                            batch_norm, k=effective_k, is_reference_member=True
                        ).to(self.device)
                        d_anom = local_extractors[i](
                            batch_anom, k=effective_k, is_reference_member=False
                        ).to(self.device)

                        # Forward & loss
                        logits_norm = client_model(d_norm)
                        logits_anom = client_model(d_anom)
                        loss = loss_fn(logits_norm, logits_anom)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        losses.append(loss.item())

                client_state_dicts.append(client_model.state_dict())
                client_round_losses.append(float(np.mean(losses)) if losses else 0.0)

            # Standard FedAvg parameter aggregation
            new_global_state = {}
            for key in global_state.keys():
                new_global_state[key] = sum(
                    client_weights[i] * client_state_dicts[i][key].float() for i in range(M)
                )

            self.global_model.load_state_dict(new_global_state)

            round_summary = {
                "round": r,
                "mean_loss": float(np.mean(client_round_losses)),
                "client_losses": client_round_losses,
            }
            self.history.append(round_summary)

            if verbose:
                print(f"[NaiveFedLunar] Round {r}/{rounds} - Mean Loss: {round_summary['mean_loss']:.4f}")

        # Store pooled normal data for test inference
        self.reference_data = np.concatenate(client_datasets, axis=0)
        self.global_knn_extractor = KNNDistanceExtractor(self.reference_data, device=self.device)

        return self

    def decision_function(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Compute anomaly score for test samples.
        Higher score corresponds to higher likelihood of anomaly.

        Args:
            X: Input samples of shape (N, D).
            batch_size: Batch size for distance computation.

        Returns:
            1D numpy array of shape (N,) with anomaly scores in [0, 1].
        """
        if self.global_model is None or self.global_knn_extractor is None:
            raise RuntimeError("Model must be fitted before calling decision_function")

        self.global_model.eval()
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        N = X_np.shape[0]
        scores: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_x = X_np[i : i + batch_size]
                dists = self.global_knn_extractor(batch_x, k=self.global_model.k, is_reference_member=False).to(
                    self.device
                )
                probs = self.global_model.predict_proba(dists).cpu().numpy().ravel()
                scores.append(probs)

        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Predict class probabilities [P(normal), P(anomaly)] for test samples.

        Args:
            X: Input samples of shape (N, D).
            batch_size: Batch size for distance computation.

        Returns:
            2D numpy array of shape (N, 2) where column 0 is P(normal) and column 1 is P(anomaly).
        """
        anomaly_scores = self.decision_function(X, batch_size=batch_size)
        anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)
        proba = np.zeros((len(anomaly_scores), 2), dtype=np.float32)
        proba[:, 1] = anomaly_scores
        proba[:, 0] = 1.0 - anomaly_scores
        return proba

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Predict binary labels (0 = normal, 1 = anomaly) based on decision threshold.
        """
        scores = self.decision_function(X, batch_size=batch_size)
        return (scores >= threshold).astype(int)

    def score(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """Backward-compatible alias for decision_function."""
        return self.decision_function(X, batch_size=batch_size)
