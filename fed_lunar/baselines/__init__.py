"""
3-Tier Baseline Hierarchy for Federated One-Class IoT Intrusion Detection.

- Tier 1: Naive Federated LUNAR (NaiveFedLunar)
- Tier 2: Federated Deep AutoEncoder (FedAutoEncoder)
          FedProx-adapted LUNAR (FedProxLunar)
          Standard PCGrad-adapted LUNAR (PCGradFedLunar)
- Tier 3: LOC-NFST Closed-Form Null-Space Theoretical Bound (LOC_NFST_Bound)
"""

from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.baselines.fed_ae import FedAutoEncoder
from fed_lunar.baselines.fedprox_lunar import FedProxLunar, PCGradFedLunar
from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound

__all__ = [
    "NaiveFedLunar",
    "FedAutoEncoder",
    "FedProxLunar",
    "PCGradFedLunar",
    "LOC_NFST_Bound",
]
