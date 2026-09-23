"""
Fed-LUNAR Federated: Client coordination, FSDS density sketches, and DROGA server aggregation.
"""

from fed_lunar.federated.sketches import FSDSSketch, compute_fsds_sketch
from fed_lunar.federated.strategy import (
    DROGAStrategy,
    dr_pcgrad,
    dr_cagrad,
    compute_gradient_conflict_metrics,
)
from fed_lunar.federated.client import LunarClient

__all__ = [
    "FSDSSketch",
    "compute_fsds_sketch",
    "DROGAStrategy",
    "dr_pcgrad",
    "dr_cagrad",
    "compute_gradient_conflict_metrics",
    "LunarClient",
]
