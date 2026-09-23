"""
Fed-LUNAR Client: Local Node Coordinator.

Orchestrates local training iterations, subspace pseudo-negative synthesis,
Cross-Manifold Negative Purging (CMNP) with peer FSDS sketches, and client gradient export.

Reference:
    Explorer 3 Report (Mathematical Foundations), Algorithm 1 & Algorithm 3.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator, CMNPFilter
from fed_lunar.federated.sketches import FSDSSketch, compute_fsds_sketch


class LunarClient:
    """
    Federated LUNAR Local Client.

    Args:
        client_id: Unique client identifier.
        data_normal: Strictly benign local training data (N, D).
        k: Neighborhood size for k-NN distance ranking (default: 10).
        rank: Subspace rank r for FSDS sketches (default: 10).
        negative_ratio: Pseudo-negative generation ratio (default: 1.0).
        sigma_pert: Tangential/orthogonal perturbation scale (default: 0.1).
        tau_null: CMNP null-space rejection threshold multiplier (default: 1.0).
        alpha: CMNP Chi-squared in-subspace significance level (default: 0.01).
        hidden_dims: LUNAR MLP hidden layer architecture (default: [64, 32, 16]).
        dropout: LUNAR MLP dropout rate (default: 0.1).
        device: PyTorch device ('cpu' or 'cuda').
        seed: Random seed.
    """

    def __init__(
        self,
        client_id: Union[int, str],
        data_normal: Union[np.ndarray, torch.Tensor],
        k: int = 10,
        rank: int = 10,
        negative_ratio: float = 1.0,
        sigma_pert: float = 0.1,
        tau_null: float = 1.0,
        alpha: float = 0.01,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        self.client_id = client_id
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store training data
        if isinstance(data_normal, torch.Tensor):
            self.data_normal = data_normal.detach().cpu().numpy().astype(np.float32)
        else:
            self.data_normal = np.asarray(data_normal, dtype=np.float32)

        self.n_samples, self.ambient_dim = self.data_normal.shape
        self.k = min(k, self.n_samples - 1)
        self.rank = min(rank, self.ambient_dim, self.n_samples - 1)
        self.negative_ratio = negative_ratio
        self.sigma_pert = sigma_pert
        self.seed = seed

        # 1. Initialize k-NN distance extractor on local reference set
        self.knn_extractor = KNNDistanceExtractor(self.data_normal, device=self.device)

        # 2. Compute local FSDS manifold sketch
        self.sketch: FSDSSketch = compute_fsds_sketch(
            self.data_normal,
            client_id=self.client_id,
            rank=self.rank,
        )

        # 3. Initialize CMNP filter and negative generator
        self.cmnp_filter = CMNPFilter(tau_null=tau_null, alpha=alpha)
        self.negative_gen = SubspaceNegativeGenerator(
            negative_ratio=self.negative_ratio,
            sigma_pert=self.sigma_pert,
            cmnp_filter=self.cmnp_filter,
            subspace_U=self.sketch.U,
            mode="subspace",
            seed=seed,
        )

        # 4. Initialize local LUNAR MLP
        self.model = LUNAR_MLP(
            k=self.k,
            hidden_dims=hidden_dims if hidden_dims is not None else [64, 32, 16],
            dropout=dropout,
        ).to(self.device)

        self.criterion = LunarDistanceRankingLoss()

    def export_sketch(self) -> FSDSSketch:
        """Export local FSDS sketch for privacy-preserving broadcast."""
        return self.sketch

    def receive_peer_sketches(
        self,
        peer_sketches: Union[Dict[Union[int, str], FSDSSketch], List[FSDSSketch]],
    ) -> None:
        """
        Receive peer manifold sketches from server and register them in CMNPFilter.
        Sketches corresponding to this client's own ID are automatically filtered out.
        """
        filtered_peers: Dict[Union[int, str], FSDSSketch] = {}
        if isinstance(peer_sketches, list):
            for s in peer_sketches:
                if str(s.client_id) != str(self.client_id):
                    filtered_peers[s.client_id] = s
        elif isinstance(peer_sketches, dict):
            for cid, s in peer_sketches.items():
                if str(cid) != str(self.client_id):
                    filtered_peers[cid] = s

        self.cmnp_filter.set_peer_sketches(filtered_peers)

    def get_parameters_flat(self) -> torch.Tensor:
        """Get flattened 1D model parameters."""
        return torch.cat([p.detach().cpu().view(-1) for p in self.model.parameters()])

    def set_parameters_flat(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from a flattened 1D tensor."""
        offset = 0
        flat_p = flat_params.to(self.device).float()
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p.copy_(flat_p[offset : offset + numel].view_as(p))
                offset += numel

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 128,
        lambda_anom: float = 1.0,
    ) -> Dict[str, float]:
        """
        Execute a single local training epoch.

        Returns:
            Dict containing average loss and rejection stats.
        """
        self.model.train()
        n_samples = self.n_samples
        indices = np.random.permutation(n_samples)

        epoch_losses: List[float] = []
        epoch_rejections: List[float] = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]
            batch_norm_np = self.data_normal[batch_idx]

            # 1. Generate pseudo-negatives with CMNP filtering
            batch_anom_np, stats = self.negative_gen.generate(batch_norm_np)
            epoch_rejections.append(stats.get("rejection_rate", 0.0))

            # 2. Extract k-NN distance vectors
            # For normal points in reference dictionary, exclude self (distance 0)
            d_norm = self.knn_extractor(batch_norm_np, k=self.k, is_reference_member=True).to(self.device)
            d_anom = self.knn_extractor(batch_anom_np, k=self.k, is_reference_member=False).to(self.device)

            # 3. Forward pass
            logits_norm = self.model(d_norm)
            logits_anom = self.model(d_anom)

            # 4. Compute distance ranking loss
            loss = self.criterion(logits_norm, logits_anom, lambda_anom=lambda_anom)

            # 5. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        return {
            "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "rejection_rate": float(np.mean(epoch_rejections)) if epoch_rejections else 0.0,
        }

    def train_round(
        self,
        global_weights: Optional[torch.Tensor] = None,
        epochs: int = 3,
        lr: float = 0.001,
        batch_size: int = 128,
        lambda_anom: float = 1.0,
        weight_decay: float = 1e-4,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute local federated round training.

        Args:
            global_weights: Current global model parameters (flattened 1D tensor).
            epochs: Number of local training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.
            lambda_anom: Weight of anomaly loss term.
            weight_decay: L2 regularization.

        Returns:
            Tuple of (effective_gradient, metrics_dict).
            effective_gradient = theta_initial - theta_final (or normalized by lr*epochs).
        """
        # Synchronize with global model if provided
        if global_weights is not None:
            self.set_parameters_flat(global_weights)

        theta_initial = self.get_parameters_flat()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        losses = []
        rejections = []
        for _ in range(epochs):
            metrics = self.train_epoch(optimizer, batch_size=batch_size, lambda_anom=lambda_anom)
            losses.append(metrics["loss"])
            rejections.append(metrics["rejection_rate"])

        theta_final = self.get_parameters_flat()

        # Effective gradient / pseudo-gradient: direction of parameter displacement
        # g_c = theta_initial - theta_final
        effective_gradient = theta_initial - theta_final

        train_metrics = {
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "epochs": epochs,
            "final_loss": float(losses[-1]) if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_rejection_rate": float(np.mean(rejections)) if rejections else 0.0,
            "cumulative_rejection_rate": self.cmnp_filter.get_cumulative_rejection_rate(),
            "grad_norm": float(torch.norm(effective_gradient)),
        }

        return effective_gradient, train_metrics

    def predict_score(
        self,
        X_test: Union[np.ndarray, torch.Tensor],
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Compute anomaly scores (probabilities in [0, 1]) for test points.

        Args:
            X_test: Test features of shape (N_test, D).
            batch_size: Batch size for distance computation.

        Returns:
            1D array of anomaly probabilities in [0, 1].
        """
        self.model.eval()
        if isinstance(X_test, torch.Tensor):
            X_test_np = X_test.detach().cpu().numpy()
        else:
            X_test_np = np.asarray(X_test, dtype=np.float32)

        n_test = X_test_np.shape[0]
        scores_list = []

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                batch_x = X_test_np[i : i + batch_size]
                dists = self.knn_extractor(batch_x, k=self.k, is_reference_member=False).to(self.device)
                probs = self.model.predict_proba(dists).cpu().numpy().ravel()
                scores_list.append(probs)

        return np.concatenate(scores_list, axis=0) if scores_list else np.empty(0, dtype=np.float32)

    def evaluate(
        self,
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate local model on test data.

        Args:
            X_test: Test samples (N, D).
            y_test: Ground truth labels (0 = normal, 1 = anomaly).
            threshold: Decision threshold for classification metrics.

        Returns:
            Dict containing predictions, scores, and (if y_test given) AUC-ROC, F1, FAR.
        """
        scores = self.predict_score(X_test)
        preds = (scores >= threshold).astype(int)

        results: Dict[str, Any] = {
            "scores": scores,
            "predictions": preds,
        }

        if y_test is not None:
            if isinstance(y_test, torch.Tensor):
                y_true = y_test.detach().cpu().numpy().astype(int).ravel()
            else:
                y_true = np.asarray(y_test, dtype=int).ravel()

            # AUC-ROC
            try:
                auc_roc = float(roc_auc_score(y_true, scores) * 100.0)
            except Exception:
                auc_roc = 50.0

            # F1-Score
            f1 = float(f1_score(y_true, preds, zero_division=0) * 100.0)

            # False Alarm Rate (FAR) = FP / (FP + TN)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
                far = float(fp / max(1, fp + tn) * 100.0)
            except Exception:
                far = 0.0

            results["auc_roc"] = auc_roc
            results["f1_score"] = f1
            results["far"] = far

        return results
