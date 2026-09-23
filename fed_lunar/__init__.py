"""
Fed-LUNAR: Federated Learnable Unified Neighborhood-based Anomaly Ranking
with Cross-Manifold Negative Purging (CMNP) and Orthogonal Gradient Alignment (DROGA).
"""

__version__ = "0.1.0"

from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator, CMNPFilter
from fed_lunar.models.autoencoder import SimpleAutoEncoder
from fed_lunar.federated.sketches import FSDSSketch, compute_fsds_sketch
from fed_lunar.federated.client import LunarClient
from fed_lunar.federated.strategy import DROGAStrategy, dr_pcgrad, dr_cagrad, compute_gradient_conflict_metrics
from fed_lunar.federated.fed_lunar import FedLUNAR

__all__ = [
    "LUNAR_MLP",
    "KNNDistanceExtractor",
    "LunarDistanceRankingLoss",
    "SubspaceNegativeGenerator",
    "CMNPFilter",
    "SimpleAutoEncoder",
    "FSDSSketch",
    "compute_fsds_sketch",
    "LunarClient",
    "DROGAStrategy",
    "dr_pcgrad",
    "dr_cagrad",
    "compute_gradient_conflict_metrics",
    "FedLUNAR",
]
