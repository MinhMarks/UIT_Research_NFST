"""
Tier 2 Baselines: FedProx-LUNAR and PCGrad-LUNAR (Feature F7).

1. FedProxLunar: Federated LUNAR with proximal regularization (mu / 2) * ||theta - theta_t||^2
   added to local distance-ranking loss to limit client drift under non-IID conditions.
2. PCGradFedLunar: Federated LUNAR using standard PCGrad gradient surgery (Yu et al., NeurIPS 2020)
   without Federated Subspace Density Sketches (FSDS) or CMNP filtering.

Reference:
    - Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020.
    - Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), NeurIPS 2020.
    - Milestone M2 Specification (Feature F7).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator
from fed_lunar.federated.strategy import dr_pcgrad


class FedProxLunar:
    """
    Tier 2 Baseline: FedProx-adapted Federated LUNAR.

    Incorporates a proximal regularization penalty (mu / 2) * ||theta - theta_t||^2
    to client distance-ranking optimization to counter client drift across disjoint IoT sub-manifolds.

    Args:
        k: Neighborhood size for k-NN distance ranking (default: 10).
        mu: Proximal regularization coefficient (default: 0.01).
        hidden_dims: LUNAR MLP hidden dimensions (default: [64, 32, 16]).
        dropout: Dropout rate in LUNAR MLP (default: 0.1).
        negative_ratio: Ratio of generated pseudo-negatives to normal points (default: 1.0).
        sigma_pert: Perturbation noise scale (default: 0.1).
        lr: Local learning rate (default: 0.001).
        batch_size: Local batch size (default: 128).
        local_epochs: Number of local training epochs per communication round (default: 3).
        weight_decay: L2 parameter regularization (default: 1e-4).
        device: Torch device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 10,
        mu: float = 0.01,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        negative_ratio: float = 1.0,
        sigma_pert: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 128,
        local_epochs: int = 3,
        weight_decay: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        self.k = k
        self.mu = mu
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32, 16]
        self.dropout = dropout
        self.negative_ratio = negative_ratio
        self.sigma_pert = sigma_pert
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
    ) -> "FedProxLunar":
        """
        Train FedProx-LUNAR across multiple communication rounds.

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

        min_client_n = min(client_sizes)
        effective_k = min(self.k, max(1, min_client_n - 1))

        self.global_model = LUNAR_MLP(
            k=effective_k,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        local_extractors = [
            KNNDistanceExtractor(data, device=self.device) for data in client_datasets
        ]
        local_generators = [
            SubspaceNegativeGenerator(
                negative_ratio=self.negative_ratio,
                sigma_pert=self.sigma_pert,
                cmnp_filter=None,  # No CMNP filter
                mode="subspace",
                seed=self.seed + idx,
            )
            for idx in range(M)
        ]

        loss_fn = LunarDistanceRankingLoss()
        self.history = []

        for r in range(1, rounds + 1):
            client_state_dicts = []
            client_round_losses = []

            global_state = self.global_model.state_dict()

            for i in range(M):
                client_model = LUNAR_MLP(
                    k=effective_k,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                ).to(self.device)
                client_model.load_state_dict(global_state)
                client_model.train()

                # Snapshot initial parameters for FedProx proximal regularization: (mu/2) * ||theta - theta_t||^2
                init_params = [p.detach().clone() for p in client_model.parameters()]

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

                        batch_anom, _ = local_generators[i].generate(batch_norm)

                        d_norm = local_extractors[i](
                            batch_norm, k=effective_k, is_reference_member=True
                        ).to(self.device)
                        d_anom = local_extractors[i](
                            batch_anom, k=effective_k, is_reference_member=False
                        ).to(self.device)

                        logits_norm = client_model(d_norm)
                        logits_anom = client_model(d_anom)
                        ranking_loss = loss_fn(logits_norm, logits_anom)

                        # Compute proximal term: (mu / 2) * sum_l ||theta_l - theta_t,l||_2^2
                        prox_term = torch.tensor(0.0, device=self.device)
                        for p, p_init in zip(client_model.parameters(), init_params):
                            prox_term = prox_term + torch.sum((p - p_init) ** 2)

                        total_loss = ranking_loss + (0.5 * self.mu) * prox_term

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        losses.append(total_loss.item())

                client_state_dicts.append(client_model.state_dict())
                client_round_losses.append(float(np.mean(losses)) if losses else 0.0)

            # Standard FedAvg aggregation
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
                print(f"[FedProxLunar] Round {r}/{rounds} - Mean Loss: {round_summary['mean_loss']:.4f}")

        self.reference_data = np.concatenate(client_datasets, axis=0)
        self.global_knn_extractor = KNNDistanceExtractor(self.reference_data, device=self.device)

        return self

    def decision_function(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Compute anomaly scores for test points.
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

    score = decision_function

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """
        Predict class probabilities [P(normal), P(anomaly)].
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
        Predict binary labels (0 = normal, 1 = anomaly).
        """
        scores = self.decision_function(X, batch_size=batch_size)
        return (scores >= threshold).astype(int)


class PCGradFedLunar:
    """
    Tier 2 Baseline: Federated LUNAR with Standard PCGrad Aggregation.

    Applies standard PCGrad gradient projection (Yu et al., NeurIPS 2020) on client
    parameter updates at the server level, but WITHOUT Federated Subspace Density Sketches (FSDS)
    or Cross-Manifold Negative Purging (CMNP).

    Args:
        k: Neighborhood size for k-NN distance ranking (default: 10).
        hidden_dims: Architecture of LUNAR MLP hidden layers (default: [64, 32, 16]).
        dropout: Dropout rate in LUNAR MLP (default: 0.1).
        negative_ratio: Ratio of generated pseudo-negatives to normal points (default: 1.0).
        sigma_pert: Perturbation noise scale (default: 0.1).
        lr: Local learning rate (default: 0.001).
        batch_size: Local batch size (default: 128).
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
    ) -> "PCGradFedLunar":
        """
        Train PCGradFedLunar across decentralized clients.

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

        min_client_n = min(client_sizes)
        effective_k = min(self.k, max(1, min_client_n - 1))

        self.global_model = LUNAR_MLP(
            k=effective_k,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        local_extractors = [
            KNNDistanceExtractor(data, device=self.device) for data in client_datasets
        ]
        local_generators = [
            SubspaceNegativeGenerator(
                negative_ratio=self.negative_ratio,
                sigma_pert=self.sigma_pert,
                cmnp_filter=None,  # No CMNP filter
                mode="subspace",
                seed=self.seed + idx,
            )
            for idx in range(M)
        ]

        loss_fn = LunarDistanceRankingLoss()
        self.history = []

        for r in range(1, rounds + 1):
            client_gradients: List[torch.Tensor] = []
            client_round_losses: List[float] = []

            global_state = self.global_model.state_dict()
            global_flat = torch.cat([p.detach().view(-1) for p in self.global_model.parameters()])

            for i in range(M):
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

                        batch_anom, _ = local_generators[i].generate(batch_norm)

                        d_norm = local_extractors[i](
                            batch_norm, k=effective_k, is_reference_member=True
                        ).to(self.device)
                        d_anom = local_extractors[i](
                            batch_anom, k=effective_k, is_reference_member=False
                        ).to(self.device)

                        logits_norm = client_model(d_norm)
                        logits_anom = client_model(d_anom)
                        loss = loss_fn(logits_norm, logits_anom)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        losses.append(loss.item())

                # Client effective gradient = theta_initial - theta_final
                client_flat = torch.cat([p.detach().view(-1) for p in client_model.parameters()])
                g_c = (global_flat - client_flat).cpu()
                client_gradients.append(g_c)
                client_round_losses.append(float(np.mean(losses)) if losses else 0.0)

            # Apply standard PCGrad aggregation at server
            g_aligned = dr_pcgrad(
                client_gradients,
                weights=client_weights,
                seed=self.seed + r,
            ).to(self.device)

            # Apply server update: theta_{t+1} = theta_t - g_aligned
            new_global_flat = global_flat - g_aligned
            offset = 0
            with torch.no_grad():
                for p in self.global_model.parameters():
                    numel = p.numel()
                    p.copy_(new_global_flat[offset : offset + numel].view_as(p))
                    offset += numel

            round_summary = {
                "round": r,
                "mean_loss": float(np.mean(client_round_losses)),
                "aligned_gradient_norm": float(torch.norm(g_aligned).item()),
            }
            self.history.append(round_summary)

            if verbose:
                print(
                    f"[PCGradFedLunar] Round {r}/{rounds} - Mean Loss: {round_summary['mean_loss']:.4f} "
                    f"Aligned Norm: {round_summary['aligned_gradient_norm']:.4f}"
                )

        self.reference_data = np.concatenate(client_datasets, axis=0)
        self.global_knn_extractor = KNNDistanceExtractor(self.reference_data, device=self.device)

        return self

    def decision_function(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """Compute anomaly scores for test points."""
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

    score = decision_function

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
        """Predict class probabilities [P(normal), P(anomaly)]."""
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
        """Predict binary labels (0 = normal, 1 = anomaly)."""
        scores = self.decision_function(X, batch_size=batch_size)
        return (scores >= threshold).astype(int)
