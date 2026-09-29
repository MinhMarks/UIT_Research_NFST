import sys
sys.path.insert(0, ".")
import time
import numpy as np
import torch
from fed_lunar.federated.strategy import (
    dr_cagrad,
    dr_pcgrad,
    compute_gradient_conflict_metrics,
    DROGAStrategy,
)
from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss
from fed_lunar.models.negative_gen import SubspaceNegativeGenerator, CMNPFilter
from fed_lunar.federated.sketches import compute_fsds_sketch, FSDSSketch

print("=== Starting Adversarial & Boundary Stress Tests ===")

# 1. Scalability of DROGA with M=20 clients and P=10,000 parameters
torch.manual_seed(42)
M, P = 20, 10000
grads = [torch.randn(P) for _ in range(M)]

t0 = time.time()
metrics = compute_gradient_conflict_metrics(grads)
t_metrics = time.time() - t0

t0 = time.time()
g_pc = dr_pcgrad(grads, seed=42)
t_pc = time.time() - t0

t0 = time.time()
g_ca = dr_cagrad(grads, c_param=0.4)
t_ca = time.time() - t0

print(f"DROGA M={M}, P={P}:")
print(f"  Conflict Metrics time: {t_metrics*1000:.2f}ms, GCR: {metrics['gcr_percent']:.1f}%")
print(f"  DR-PCGrad time: {t_pc*1000:.2f}ms, norm: {torch.norm(g_pc).item():.4f}")
print(f"  DR-CAGrad time: {t_ca*1000:.2f}ms, norm: {torch.norm(g_ca).item():.4f}")

# 2. Check inner products for DR-CAGrad
for i, g in enumerate(grads):
    ip = torch.dot(g_ca, g).item()
    assert ip >= -1e-4, f"CAGrad failed monotonicity on client {i}: ip={ip}"
print("DR-CAGrad inner product monotonicity holds across all M=20 clients!")

# 3. Test negative generator boundary cases
cmnp = CMNPFilter()
gen = SubspaceNegativeGenerator(cmnp_filter=cmnp)

# Normal data with identical points (0 variance)
X_const = np.ones((50, 10))
neg, stats = gen.generate(X_const)
print(f"Constant input: neg shape={neg.shape}, fallback={stats['fallback_used']}")

# Normal data with small sample size
X_small = np.random.randn(3, 5)
neg_small, stats_small = gen.generate(X_small)
print(f"Small N=3 input: neg shape={neg_small.shape}")

# 4. Test LUNAR_MLP with extreme input distances (inf, nan, large values)
mlp = LUNAR_MLP(k=5)
extreme_dists = torch.tensor([[1e6, 2e6, 3e6, 4e6, 5e6]])
score = mlp.predict_proba(extreme_dists)
print(f"Extreme large distances score: {score.item():.4f}")
assert 0.0 <= score.item() <= 1.0

# 5. Check kwargs interface compatibility for NegativeGenerator
# PROJECT.md specifies: NegativeGenerator(negative_ratio: float = 1.0, epsilon: float = 0.1, cmnp: Optional[CMNPFilter] = None)
try:
    from fed_lunar.models.negative_gen import NegativeGenerator
    # Try calling with PROJECT.md keyword arguments
    ng = NegativeGenerator(negative_ratio=1.0, epsilon=0.1, cmnp=None)
    print("NegativeGenerator accepted epsilon and cmnp kwargs directly.")
except TypeError as e:
    print(f"Interface observation: NegativeGenerator kwargs mismatch: {e}")

print("=== All Stress Tests Completed ===")
