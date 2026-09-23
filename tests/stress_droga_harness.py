"""
Independent Empirical Stress Harness for DROGA (DR-PCGrad and DR-CAGrad).

Challenges Theorem 3 & 4 (DROGA non-conflicting descent guarantee):
    <g_aligned, g_i> >= -1e-5 for all i in [M].

Evaluates:
1. 1,000 randomized trials for M in {3, 5, 8, 10} with antagonistic angles in [-1, -0.1].
2. Extreme edge cases:
   - Collinear opposing gradients (g1 = -g2)
   - Zero gradients (gi = 0)
   - Scale disparities (||g1|| = 10^4 ||g2||)
   - Combined scale disparity + opposing
   - All zero gradients
3. Logs numerical results, minimum inner products, violation rates, and solver convergence.
"""

import sys
import os
import pathlib
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fed_lunar.federated.strategy import (
    dr_pcgrad,
    dr_cagrad,
    compute_gradient_conflict_metrics,
)


def generate_antagonistic_gradients(
    M: int,
    dim: int = 50,
    rng: np.random.Generator = None,
    scenario: str = "cluster_opposing",
) -> List[torch.Tensor]:
    """
    Generate M gradient vectors with antagonistic angles (cos(g_i, g_j) in [-1.0, -0.1]).
    """
    if rng is None:
        rng = np.random.default_rng()

    if scenario == "simplex":
        # Simplex in M-1 dimensions embedded in dim:
        # All pairwise cosines are exactly -1 / (M - 1).
        # Perturbed slightly so they are not perfectly symmetric.
        A = rng.normal(size=(dim, M))
        Q, _ = np.linalg.qr(A)
        # Center the first M columns of Q to form regular simplex
        V = Q[:, :M]
        # Mean subtract
        V_centered = V - np.mean(V, axis=1, keepdims=True)
        # Add random scale and small perturbation
        scales = rng.uniform(0.5, 2.0, size=M)
        grads = []
        for i in range(M):
            v = V_centered[:, i]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-12:
                v = v / norm_v
            # small jitter
            v = v + rng.normal(scale=0.02, size=dim)
            v = v * scales[i]
            grads.append(torch.from_numpy(v.astype(np.float32)))
        return grads

    elif scenario == "cluster_opposing":
        # Split into two opposing groups
        # v is principal axis
        v = rng.normal(size=dim)
        v = v / (np.linalg.norm(v) + 1e-12)
        n1 = M // 2
        n2 = M - n1

        grads = []
        # Group 1 along +v with angular spread
        for _ in range(n1):
            noise = rng.normal(scale=0.3, size=dim)
            g = v + noise
            scale = rng.uniform(0.5, 2.0)
            grads.append(torch.from_numpy((g * scale).astype(np.float32)))

        # Group 2 along -v with angular spread
        # cosine with group 1 will be in [-1.0, -0.2]
        for _ in range(n2):
            noise = rng.normal(scale=0.3, size=dim)
            g = -v + noise
            scale = rng.uniform(0.5, 2.0)
            grads.append(torch.from_numpy((g * scale).astype(np.float32)))

        return grads

    elif scenario == "multi_axis_antagonistic":
        # M axes with negative mutual projections
        # Generate random base vectors, then reflect some dimensions to create antagonistic angles
        grads = []
        base = rng.normal(size=(M, dim))
        # Gram-Schmidt then skew
        for i in range(M):
            g = base[i]
            if i % 2 == 1:
                # Oppose the previous client with some orthogonal noise
                prev = grads[i - 1].numpy()
                g = -prev + rng.normal(scale=0.4, size=dim)
            scale = rng.uniform(0.2, 5.0)
            grads.append(torch.from_numpy((g * scale).astype(np.float32)))
        return grads

    else:
        # Fallback: random vectors with negative cosine target
        grads = [torch.randn(dim) for _ in range(M)]
        return grads


def evaluate_trial(
    gradients: List[torch.Tensor],
    c_param: float = 0.4,
    tol: float = -1e-5,
) -> Dict[str, Any]:
    """
    Run DR-PCGrad and DR-CAGrad on a set of gradients and check inner products.
    """
    M = len(gradients)
    metrics = compute_gradient_conflict_metrics(gradients)
    min_cos = metrics["min_cosine"]
    gcr = metrics["gcr"]

    # 1. DR-PCGrad
    g_pc = dr_pcgrad(gradients, seed=42)
    pc_ips = [float(torch.dot(g_pc, g).item()) for g in gradients]
    pc_min_ip = min(pc_ips)
    pc_violation = pc_min_ip < tol

    # 2. DR-CAGrad (default dual_simplex_qp with c_param=0.4)
    g_ca = dr_cagrad(gradients, c_param=c_param, mode="dual_simplex_qp")
    ca_ips = [float(torch.dot(g_ca, g).item()) for g in gradients]
    ca_min_ip = min(ca_ips)
    ca_violation = ca_min_ip < tol

    # 3. DR-CAGrad (minimax_simplex with c_param=0.4)
    g_ca_mm = dr_cagrad(gradients, c_param=c_param, mode="minimax_simplex")
    ca_mm_ips = [float(torch.dot(g_ca_mm, g).item()) for g in gradients]
    ca_mm_min_ip = min(ca_mm_ips)
    ca_mm_violation = ca_mm_min_ip < tol

    # 4. Check critical radius c_crit from Theorem 3
    # c_crit = max_i |sin angle(g0, g_i)|
    w0 = [1.0 / M] * M
    g0 = torch.zeros_like(gradients[0])
    for w, g in zip(w0, gradients):
        g0 += w * g
    norm_g0 = float(torch.norm(g0).item())

    c_crit = 0.0
    if norm_g0 > 1e-12:
        for g in gradients:
            norm_g = float(torch.norm(g).item())
            if norm_g > 1e-12:
                cos_val = float(torch.dot(g0, g).item()) / (norm_g0 * norm_g)
                cos_val = max(-1.0, min(1.0, cos_val))
                sin_val = np.sqrt(max(0.0, 1.0 - cos_val ** 2))
                if sin_val > c_crit:
                    c_crit = sin_val

    # 5. DR-CAGrad with adaptive c >= c_crit
    c_adapt = min(0.99, max(c_param, c_crit + 0.01))
    g_ca_adapt = dr_cagrad(gradients, c_param=c_adapt, mode="dual_simplex_qp")
    ca_adapt_ips = [float(torch.dot(g_ca_adapt, g).item()) for g in gradients]
    ca_adapt_min_ip = min(ca_adapt_ips)
    ca_adapt_violation = ca_adapt_min_ip < tol

    return {
        "min_cosine": min_cos,
        "gcr": gcr,
        "c_crit": c_crit,
        "pc_min_ip": pc_min_ip,
        "pc_violation": pc_violation,
        "ca_min_ip": ca_min_ip,
        "ca_violation": ca_violation,
        "ca_mm_min_ip": ca_mm_min_ip,
        "ca_mm_violation": ca_mm_violation,
        "ca_adapt_min_ip": ca_adapt_min_ip,
        "ca_adapt_violation": ca_adapt_violation,
        "g_pc_norm": float(torch.norm(g_pc).item()),
        "g_ca_norm": float(torch.norm(g_ca).item()),
    }


def run_randomized_stress_suite(
    client_counts: List[int] = [3, 5, 8, 10],
    num_trials: int = 1000,
    seed: int = 2026,
) -> Dict[int, Dict[str, Any]]:
    """
    Run 1,000 randomized trials for each M in client_counts.
    """
    rng = np.random.default_rng(seed)
    results = {}

    scenarios = ["simplex", "cluster_opposing", "multi_axis_antagonistic"]

    for M in client_counts:
        print(f"=== Running {num_trials} randomized stress trials for M = {M} clients ===")
        start_time = time.time()

        pc_violations = 0
        ca_violations = 0
        ca_mm_violations = 0
        ca_adapt_violations = 0

        pc_min_all = float("inf")
        ca_min_all = float("inf")
        ca_mm_min_all = float("inf")
        ca_adapt_min_all = float("inf")

        worst_pc_trial = None
        worst_ca_trial = None

        min_cosines = []
        gcrs = []

        for trial in range(num_trials):
            # Select scenario
            scen = scenarios[trial % len(scenarios)]
            grads = generate_antagonistic_gradients(M=M, dim=50, rng=rng, scenario=scen)

            eval_res = evaluate_trial(grads, c_param=0.4)

            min_cosines.append(eval_res["min_cosine"])
            gcrs.append(eval_res["gcr"])

            # Check PCGrad
            if eval_res["pc_min_ip"] < pc_min_all:
                pc_min_all = eval_res["pc_min_ip"]
                worst_pc_trial = (trial, scen, eval_res["pc_min_ip"])
            if eval_res["pc_violation"]:
                pc_violations += 1

            # Check CAGrad (default)
            if eval_res["ca_min_ip"] < ca_min_all:
                ca_min_all = eval_res["ca_min_ip"]
                worst_ca_trial = (trial, scen, eval_res["ca_min_ip"])
            if eval_res["ca_violation"]:
                ca_violations += 1

            # Check CAGrad (minimax)
            if eval_res["ca_mm_min_ip"] < ca_mm_min_all:
                ca_mm_min_all = eval_res["ca_mm_min_ip"]
            if eval_res["ca_mm_violation"]:
                ca_mm_violations += 1

            # Check CAGrad (adaptive c)
            if eval_res["ca_adapt_min_ip"] < ca_adapt_min_all:
                ca_adapt_min_all = eval_res["ca_adapt_min_ip"]
            if eval_res["ca_adapt_violation"]:
                ca_adapt_violations += 1

        elapsed = time.time() - start_time
        res_M = {
            "M": M,
            "trials": num_trials,
            "elapsed_s": elapsed,
            "mean_min_cos": float(np.mean(min_cosines)),
            "mean_gcr": float(np.mean(gcrs)),
            "pc_violations": pc_violations,
            "pc_violation_rate": pc_violations / num_trials,
            "pc_min_all": pc_min_all,
            "worst_pc_trial": worst_pc_trial,
            "ca_violations": ca_violations,
            "ca_violation_rate": ca_violations / num_trials,
            "ca_min_all": ca_min_all,
            "worst_ca_trial": worst_ca_trial,
            "ca_mm_violations": ca_mm_violations,
            "ca_mm_violation_rate": ca_mm_violations / num_trials,
            "ca_mm_min_all": ca_mm_min_all,
            "ca_adapt_violations": ca_adapt_violations,
            "ca_adapt_violation_rate": ca_adapt_violations / num_trials,
            "ca_adapt_min_all": ca_adapt_min_all,
        }
        results[M] = res_M

        print(f"Results for M={M} ({elapsed:.2f}s):")
        print(f"  Avg Min Cosine: {res_M['mean_min_cos']:.4f} | Avg GCR: {res_M['mean_gcr']:.4f}")
        print(f"  DR-PCGrad: {pc_violations}/{num_trials} violations ({res_M['pc_violation_rate']*100:.2f}%), Min IP: {pc_min_all:.6e}")
        print(f"  DR-CAGrad (c=0.4): {ca_violations}/{num_trials} violations ({res_M['ca_violation_rate']*100:.2f}%), Min IP: {ca_min_all:.6e}")
        print(f"  DR-CAGrad (adapt c): {ca_adapt_violations}/{num_trials} violations ({res_M['ca_adapt_violation_rate']*100:.2f}%), Min IP: {ca_adapt_min_all:.6e}")
        print()

    return results


def run_extreme_edge_cases() -> Dict[str, Any]:
    """
    Test extreme edge cases:
    1. Collinear opposing gradients (g1 = -g2)
    2. Zero gradients (gi = 0)
    3. Scale disparities (||g1|| = 10^4 ||g2||)
    4. Combined scale disparity + opposing
    5. All zero gradients
    """
    print("=== Running Extreme Edge Cases ===")
    results = {}
    dim = 50

    # Edge Case 1: Collinear opposing gradients (g1 = -g2)
    # Subcase 1a: Exactly 2 clients g1 = [1, 0, ...], g2 = [-1, 0, ...]
    v1 = torch.zeros(dim)
    v1[0] = 1.0
    v2 = -v1.clone()
    grads_opp2 = [v1, v2]

    # DR-PCGrad on 1a
    g_pc_1a = dr_pcgrad(grads_opp2, seed=42)
    ip_pc_1a = [float(torch.dot(g_pc_1a, g).item()) for g in grads_opp2]

    # DR-CAGrad on 1a
    g_ca_1a = dr_cagrad(grads_opp2, c_param=0.4)
    ip_ca_1a = [float(torch.dot(g_ca_1a, g).item()) for g in grads_opp2]

    results["opposing_2clients"] = {
        "pc_ips": ip_pc_1a,
        "ca_ips": ip_ca_1a,
        "g_pc_norm": float(torch.norm(g_pc_1a).item()),
        "g_ca_norm": float(torch.norm(g_ca_1a).item()),
    }
    print(f"Edge Case 1a (g1 = -g2):")
    print(f"  DR-PCGrad norm: {results['opposing_2clients']['g_pc_norm']}, IPs: {ip_pc_1a}")
    print(f"  DR-CAGrad norm: {results['opposing_2clients']['g_ca_norm']}, IPs: {ip_ca_1a}")

    # Subcase 1b: 3 clients with g1 = -g2 and a third non-conflicting gradient g3
    v3 = torch.zeros(dim)
    v3[1] = 1.0
    grads_opp3 = [v1, v2, v3]

    g_pc_1b = dr_pcgrad(grads_opp3, seed=42)
    ip_pc_1b = [float(torch.dot(g_pc_1b, g).item()) for g in grads_opp3]

    g_ca_1b = dr_cagrad(grads_opp3, c_param=0.4)
    ip_ca_1b = [float(torch.dot(g_ca_1b, g).item()) for g in grads_opp3]

    results["opposing_3clients_with_orthogonal"] = {
        "pc_ips": ip_pc_1b,
        "ca_ips": ip_ca_1b,
        "g_pc_norm": float(torch.norm(g_pc_1b).item()),
        "g_ca_norm": float(torch.norm(g_ca_1b).item()),
    }
    print(f"Edge Case 1b (g1 = -g2, g3 orthogonal):")
    print(f"  DR-PCGrad norm: {results['opposing_3clients_with_orthogonal']['g_pc_norm']}, IPs: {ip_pc_1b}")
    print(f"  DR-CAGrad norm: {results['opposing_3clients_with_orthogonal']['g_ca_norm']}, IPs: {ip_ca_1b}")

    # Edge Case 2: Zero gradients (gi = 0)
    # Client 1 is zero, Client 2 and 3 are standard
    v_zero = torch.zeros(dim)
    grads_zero = [v_zero, v1, v3]

    try:
        g_pc_zero = dr_pcgrad(grads_zero, seed=42)
        ip_pc_zero = [float(torch.dot(g_pc_zero, g).item()) for g in grads_zero]
        pc_zero_ok = not torch.isnan(g_pc_zero).any().item()
    except Exception as e:
        g_pc_zero = torch.tensor([float("nan")])
        ip_pc_zero = []
        pc_zero_ok = False

    try:
        g_ca_zero = dr_cagrad(grads_zero, c_param=0.4)
        ip_ca_zero = [float(torch.dot(g_ca_zero, g).item()) for g in grads_zero]
        ca_zero_ok = not torch.isnan(g_ca_zero).any().item()
    except Exception as e:
        g_ca_zero = torch.tensor([float("nan")])
        ip_ca_zero = []
        ca_zero_ok = False

    results["zero_gradient_one_client"] = {
        "pc_ok": pc_zero_ok,
        "pc_ips": ip_pc_zero,
        "ca_ok": ca_zero_ok,
        "ca_ips": ip_ca_zero,
    }
    print(f"Edge Case 2 (One zero gradient client):")
    print(f"  DR-PCGrad ok: {pc_zero_ok}, IPs: {ip_pc_zero}")
    print(f"  DR-CAGrad ok: {ca_zero_ok}, IPs: {ip_ca_zero}")

    # Edge Case 2b: All clients zero gradients
    grads_all_zero = [torch.zeros(dim), torch.zeros(dim), torch.zeros(dim)]
    try:
        g_pc_all_zero = dr_pcgrad(grads_all_zero)
        pc_all_zero_ok = not torch.isnan(g_pc_all_zero).any().item()
    except Exception as e:
        pc_all_zero_ok = False

    try:
        g_ca_all_zero = dr_cagrad(grads_all_zero)
        ca_all_zero_ok = not torch.isnan(g_ca_all_zero).any().item()
    except Exception as e:
        ca_all_zero_ok = False

    results["all_zero_gradients"] = {
        "pc_ok": pc_all_zero_ok,
        "ca_ok": ca_all_zero_ok,
    }
    print(f"Edge Case 2b (All zero gradients): PCGrad ok: {pc_all_zero_ok}, CAGrad ok: {ca_all_zero_ok}")

    # Edge Case 3: Scale disparities (||g1|| = 10^4 ||g2||)
    g_large = v1 * 1e4
    g_small = (v1 * (-0.5) + v3 * 0.866) * 1.0  # conflicting angle, small norm 1.0
    grads_scale = [g_large, g_small]

    g_pc_scale = dr_pcgrad(grads_scale, seed=42)
    ip_pc_scale = [float(torch.dot(g_pc_scale, g).item()) for g in grads_scale]

    g_ca_scale = dr_cagrad(grads_scale, c_param=0.4)
    ip_ca_scale = [float(torch.dot(g_ca_scale, g).item()) for g in grads_scale]

    results["scale_disparity_1e4"] = {
        "pc_ips": ip_pc_scale,
        "ca_ips": ip_ca_scale,
        "pc_min_ip": min(ip_pc_scale),
        "ca_min_ip": min(ip_ca_scale),
    }
    print(f"Edge Case 3 (Scale disparity 10^4):")
    print(f"  DR-PCGrad IPs: {ip_pc_scale}")
    print(f"  DR-CAGrad IPs: {ip_ca_scale}")

    # Edge Case 4: Scale disparity with collinear opposing (g1 = 10^4 * v, g2 = -10^-4 * v)
    g_huge = v1 * 1e4
    g_tiny = -v1 * 1e-4
    grads_scale_opp = [g_huge, g_tiny]

    g_pc_scale_opp = dr_pcgrad(grads_scale_opp, seed=42)
    ip_pc_scale_opp = [float(torch.dot(g_pc_scale_opp, g).item()) for g in grads_scale_opp]

    g_ca_scale_opp = dr_cagrad(grads_scale_opp, c_param=0.4)
    ip_ca_scale_opp = [float(torch.dot(g_ca_scale_opp, g).item()) for g in grads_scale_opp]

    results["scale_disparity_opposing"] = {
        "pc_ips": ip_pc_scale_opp,
        "ca_ips": ip_ca_scale_opp,
    }
    print(f"Edge Case 4 (Scale disparity + opposing):")
    print(f"  DR-PCGrad IPs: {ip_pc_scale_opp}")
    print(f"  DR-CAGrad IPs: {ip_ca_scale_opp}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EMPIRICAL CHALLENGER STRESS HARNESS: THEOREMS 3 & 4")
    print("=" * 60)

    # Run randomized suite across M in {3, 5, 8, 10}
    rand_results = run_randomized_stress_suite(
        client_counts=[3, 5, 8, 10],
        num_trials=1000,
        seed=2026,
    )

    # Run edge cases
    edge_results = run_extreme_edge_cases()

    print("=" * 60)
    print("STRESS HARNESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
