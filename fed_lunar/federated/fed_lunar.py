"""
Proposed Novel Model: Federated LUNAR with Cross-Manifold Negative Purging (CMNP)
and Distance-Ranking Orthogonal Gradient Alignment (DROGA).

Overcomes the universal FL-IDS challenge: Adversarial Negative Gradient Cancellation
and Cross-Manifold Intrusion under Non-IID client distributions.

References:
    - Goodge et al., "LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks", AAAI 2022.
    - Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
    - Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning", NeurIPS 2021.
    - Milestone M1 & M3 Specifications.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator, CMNPFilter
from fed_lunar.federated.sketches import FSDSSketch, compute_fsds_sketch
from fed_lunar.federated.strategy import DROGAStrategy, compute_gradient_conflict_metrics


class FedLUNAR:
    """
    Proposed Fed-LUNAR: Cross-Manifold Negative Purging (CMNP) + DROGA.

    Integrates:
    1. Federated Subspace Density Sketches (FSDS): Privacy-preserving manifold descriptors (mu, Lambda, U, r_max).
    2. Cross-Manifold Negative Purging (CMNP): Filters synthesized negatives that intrude into peer clients' manifolds.
    3. Distance-Ranking Orthogonal Gradient Alignment (DROGA): Server-side gradient surgery (DR-CAGrad / DR-PCGrad)
       with unit-norm scaling to resolve scale disparity and eliminate conflicting gradient directions.

    Args:
        k: Neighborhood size for k-NN distance ranking (default: 10).
        rank: Subspace rank r for FSDS sketches (default: 10).
        tau_null: Null-space manifold boundary margin for CMNP (default: 1.0).
        alpha: In-subspace Chi-squared confidence significance level (default: 0.01).
        c_param: Conflict-aversion coefficient for DR-CAGrad (default: 0.4).
        mode: DROGA alignment strategy ('CAGrad', 'PCGrad', or 'FedAvg') (default: 'CAGrad').
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
        rank: int = 10,
        tau_null: float = 1.0,
        alpha: float = 0.01,
        c_param: float = 0.4,
        mode: str = "CAGrad",
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
        self.rank = rank
        self.tau_null = tau_null
        self.alpha = alpha
        self.c_param = c_param
        self.mode = mode
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
        self.droga_strategy = DROGAStrategy(mode=self.mode, c_param=self.c_param, seed=self.seed)
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
        client_train_data: Union[List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray], np.ndarray],
        rounds: int = 10,
        verbose: bool = False,
    ) -> "FedLUNAR":
        """
        Train Proposed Fed-LUNAR across multiple communication rounds.

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

        # Effective neighborhood size
        min_client_n = min(client_sizes)
        effective_k = min(self.k, max(1, min_client_n - 1))
        ambient_dim = client_datasets[0].shape[1]
        effective_rank = min(self.rank, ambient_dim, min_client_n - 1)

        # Initialize global LUNAR MLP
        self.global_model = LUNAR_MLP(
            k=effective_k,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # 1. Privacy-Preserving FSDS Sketch Computation and Broadcast
        client_sketches: Dict[int, FSDSSketch] = {}
        for i in range(M):
            client_sketches[i] = compute_fsds_sketch(
                client_datasets[i],
                client_id=i,
                rank=effective_rank,
            )

        # 2. Build local client components with Cross-Manifold Negative Purging (CMNP)
        local_extractors = [
            KNNDistanceExtractor(data, device=self.device) for data in client_datasets
        ]
        local_generators: List[SubspaceNegativeGenerator] = []
        for i in range(M):
            # Register peer sketches (excluding self)
            peer_sketches = {cid: sk for cid, sk in client_sketches.items() if cid != i}
            cmnp = CMNPFilter(peer_sketches=peer_sketches, tau_null=self.tau_null, alpha=self.alpha)
            gen = SubspaceNegativeGenerator(
                negative_ratio=self.negative_ratio,
                sigma_pert=self.sigma_pert,
                cmnp_filter=cmnp,
                subspace_U=client_sketches[i].U,
                mode="subspace",
                seed=self.seed + i,
            )
            local_generators.append(gen)

        loss_fn = LunarDistanceRankingLoss()
        self.history = []

        for r in range(1, rounds + 1):
            client_gradients: List[torch.Tensor] = []
            client_round_losses: List[float] = []
            client_rejection_rates: List[float] = []

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
                rejections = []
                for _ in range(self.local_epochs):
                    perm = np.random.permutation(n_norm)
                    for start_idx in range(0, n_norm, batch_size):
                        batch_idx = perm[start_idx : start_idx + batch_size]
                        batch_norm = X_norm[batch_idx]

                        # CMNP-filtered pseudo-negative synthesis
                        batch_anom, stats = local_generators[i].generate(batch_norm)
                        rejections.append(stats.get("rejection_rate", 0.0))

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
                client_rejection_rates.append(float(np.mean(rejections)) if rejections else 0.0)

            # 3. Server-Side DROGA Gradient Alignment
            g_aligned, droga_meta = self.droga_strategy.aggregate(
                client_gradients=client_gradients,
                client_weights=client_weights,
                round_idx=r,
            )
            g_aligned = g_aligned.to(self.device)

            # Update global model parameters: theta_{t+1} = theta_t - g_aligned
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
                "mean_cmnp_rejection_rate": float(np.mean(client_rejection_rates)),
                "pre_gcr": droga_meta.get("pre_gcr", 0.0),
                "pre_mean_cosine": droga_meta.get("pre_mean_cosine", 1.0),
                "aligned_gradient_norm": float(torch.norm(g_aligned).item()),
            }
            self.history.append(round_summary)

            if verbose:
                print(
                    f"[FedLUNAR] Round {r}/{rounds} - Loss: {round_summary['mean_loss']:.4f} | "
                    f"CMNP Rej: {round_summary['mean_cmnp_rejection_rate']:.3f} | "
                    f"Pre-GCR: {round_summary['pre_gcr']:.2f}"
                )

        # Reference data dictionary pooled from normal clients for test inference
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
