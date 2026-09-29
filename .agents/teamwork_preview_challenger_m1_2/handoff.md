# Milestone M1 Empirical Challenge Report: Verification of DROGA Gradient Alignment (Theorems 3 & 4)

**Agent ID:** Challenger 2 (`teamwork_preview_challenger_m1_2`)  
**Milestone:** M1 (Empirical Verification of DROGA Gradient Alignment)  
**Parent Agent ID:** `37c8034b-fcb6-4906-bcf8-1f986e523ea0`  
**Timestamp:** 2026-09-23T05:56:00Z  
**Verdict:** **`DISPROVEN`** (Unconditional Descent Guarantee & Monotonic Decentralized Convergence)

---

## 1. Observation

### 1.1. Theoretical Claims Under Evaluation
From `.agents/teamwork_preview_explorer_survey_3/report.md` (lines 370–398):
- **Theorem 3 (Strict Non-Conflict Guarantee of DROGA):**
  - Claim (i): *"Under DR-PCGrad, the projection guarantees that at the conclusion of each pairwise adjustment: $\langle g_i^{\text{proj}}, g_j \rangle \ge 0$."*
  - Claim (ii): *"Under DR-CAGrad, if the conflict aversion radius satisfies $c \ge c_{\text{crit}} \triangleq \max_{i \in [M]} |\sin \angle(g_0, g_i)|$, then the aggregated update direction $g^*$ is a simultaneous descent direction for all client losses: $\langle g^*, g_i \rangle \ge 0 \quad \forall i \in \{1, \dots, M\}$."*
- **Theorem 4 (Monotonic Decentralized Convergence):**
  - Claim: *"Under the DROGA update $\theta_{t+1} = \theta_t - \eta g^*$ with learning rate $\eta \le \frac{2 \min_i \langle g^*, g_i \rangle}{L \|g^*\|_2^2}$, every individual client achieves strict monotonic loss reduction: $\mathcal{L}_i(\theta_{t+1}) < \mathcal{L}_i(\theta_t) \quad \forall i \in \{1, \dots, M\}$."*
- **Interface Contract #2 (`PROJECT.md`, lines 70–72):**
  - Requirement: *"Output: Aligned global update $g_{\text{aligned}}$ satisfying $\langle g_{\text{aligned}}, g_i \rangle \ge 0$ for all $i \in [M]$."*

### 1.2. Independent Stress Harness Implementation & Execution
An independent stress harness was authored at `tests/stress_droga_harness.py` and executed against `fed_lunar/federated/strategy.py`.
- **Harness command:** `python tests/stress_droga_harness.py`
- **Scope:** 1,000 randomized trials per client federation size $M \in \{3, 5, 8, 10\}$ (4,000 total trials) evaluating:
  1. `dr_pcgrad` (sequential orthogonal projection)
  2. `dr_cagrad` with default `mode="dual_simplex_qp"`, $c=0.4$
  3. `dr_cagrad` with adaptive $c \ge c_{\text{crit}}$ per Theorem 3
  4. `dr_cagrad` with alternative `mode="minimax_simplex"`
  5. Five extreme geometric edge cases (opposing, zero gradients, scale disparities)
- **Empirical Threshold:** $\min_{i \in [M]} \langle g_{\text{aligned}}, g_i \rangle \ge -10^{-5}$.

### 1.3. Numerical Results: 1,000 Randomized Trials per Federation Size

| $M$ Clients | Method / Mode | Violations ($< -10^{-5}$) | Violation Rate | Worst-Case Inner Product ($\min_{i} \langle g_{\text{aligned}}, g_i \rangle$) | Avg Min Cosine | Avg GCR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$M = 3$** | DR-PCGrad | **381 / 1000** | **38.10%** | **-13.0866** | -0.5740 | 0.7700 |
| | DR-CAGrad ($c=0.4$, QP) | **636 / 1000** | **63.60%** | **-1593.1816** | -0.5740 | 0.7700 |
| | DR-CAGrad ($c \ge c_{\text{crit}}$) | **574 / 1000** | **57.40%** | **-1813.4380** | -0.5740 | 0.7700 |
| | DR-CAGrad (Minimax) | **599 / 1000** | **59.90%** | **-1804.4960** | -0.5740 | 0.7700 |
| **$M = 5$** | DR-PCGrad | **576 / 1000** | **57.60%** | **-158.7576** | -0.5385 | 0.7260 |
| | DR-CAGrad ($c=0.4$, QP) | **683 / 1000** | **68.30%** | **-1535.9360** | -0.5385 | 0.7260 |
| | DR-CAGrad ($c \ge c_{\text{crit}}$) | **662 / 1000** | **66.20%** | **-1298.6800** | -0.5385 | 0.7260 |
| | DR-CAGrad (Minimax) | **659 / 1000** | **65.90%** | **-1454.8790** | -0.5385 | 0.7260 |
| **$M = 8$** | DR-PCGrad | **663 / 1000** | **66.30%** | **-259.1573** | -0.5312 | 0.7097 |
| | DR-CAGrad ($c=0.4$, QP) | **691 / 1000** | **69.10%** | **-863.8671** | -0.5312 | 0.7097 |
| | DR-CAGrad ($c \ge c_{\text{crit}}$) | **670 / 1000** | **67.00%** | **-926.9380** | -0.5312 | 0.7097 |
| | DR-CAGrad (Minimax) | **667 / 1000** | **66.70%** | **-1149.5500** | -0.5312 | 0.7097 |
| **$M = 10$** | DR-PCGrad | **670 / 1000** | **67.00%** | **-508.1361** | -0.5319 | 0.6998 |
| | DR-CAGrad ($c=0.4$, QP) | **715 / 1000** | **71.50%** | **-608.3111** | -0.5319 | 0.6998 |
| | DR-CAGrad ($c \ge c_{\text{crit}}$) | **672 / 1000** | **67.20%** | **-644.0867** | -0.5319 | 0.6998 |
| | DR-CAGrad (Minimax) | **667 / 1000** | **66.70%** | **-817.6791** | -0.5319 | 0.6998 |

### 1.4. Scenario Breakdown (for $M = 3$)
- **Regular Simplex Scenario ($\cos \angle(g_i, g_j) = -0.5$):**
  - DR-PCGrad: **100% violations** (Worst IP: $-0.3878$)
  - DR-CAGrad (QP): **96% violations** (Worst IP: $-0.2336$)
  - DR-CAGrad (Minimax): **94% violations** (Worst IP: $-0.1548$)
- **Cluster Opposing (1D principal axis opposition):**
  - DR-PCGrad: **0% violations** (Min IP: $+0.3313$)
  - DR-CAGrad (QP): **1% violations** (Min IP: $-0.2033$)
  - DR-CAGrad (Minimax): **0% violations** (Min IP: $+0.4111$)
- **Multi-Axis Antagonistic (independent antagonistic pairs & scale jitter):**
  - DR-PCGrad: **9% violations** (Worst IP: $-7.1308$)
  - DR-CAGrad (QP): **90% violations** (Worst IP: $-1764.5155$)
  - DR-CAGrad (Minimax): **87% violations** (Worst IP: $-2228.4553$)

### 1.5. Numerical Results: Extreme Edge Cases

| Edge Case | Description | DR-PCGrad Result | DR-CAGrad Result | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Case 1a** | Collinear opposing: $g_1 = -g_2$ | $g_{\text{aligned}} = 0$, IPs: $[0.0, 0.0]$ | $g_{\text{aligned}} = 0$, IPs: $[0.0, 0.0]$ | **PASS** (Zero collapse) |
| **Case 1b** | Opposing pair + orthogonal: $g_1 = -g_2, g_3 \perp g_1$ | $\|g_{\text{aligned}}\| = 0.333$, IPs: $[0.0, 0.0, 0.333]$ | $\|g_{\text{aligned}}\| = 0.333$, IPs: $[0.0, 0.0, 0.333]$ | **PASS** (Descent on $g_3$) |
| **Case 2a** | Zero gradient client: $g_1 = 0, g_2, g_3 \ne 0$ | No NaN/crash; IPs: $[0.0, 0.333, 0.333]$ | No NaN/crash; IPs: $[0.0, 0.333, 0.333]$ | **PASS** |
| **Case 2b** | All zero gradients: $g_i = 0 \;\forall i$ | No NaN/crash; outputs zero tensor | No NaN/crash; outputs zero tensor | **PASS** |
| **Case 3** | Scale disparity: $\|g_1\| = 10^4 \|g_2\|$ (antagonistic) | IPs: $[3.75 \times 10^7, +0.375]$ | IPs: $[6.00 \times 10^7, \mathbf{-2999.38}]$ | **CRITICAL FAIL (CAGrad)** |
| **Case 4** | Scale disparity + collinear opposing ($g_1 = 10^4 v, g_2 = -10^{-4} v$) | IPs: $[4995.12, \mathbf{-4.995 \times 10^{-5}}]$ | IPs: $[6.00 \times 10^7, \mathbf{-0.60}]$ | **CRITICAL FAIL (Both)** |

### 1.6. Solver Convergence
Across 500 optimization calls per configuration, SLSQP convergence in `scipy.optimize.minimize` succeeded in **100.0%** of trials:
- $M=3$: 100.0% success (avg 2.6 iterations for QP, 5.9 for Minimax)
- $M=5$: 100.0% success (avg 3.5 iterations for QP, 10.0 for Minimax)
- $M=8$: 100.0% success (avg 4.7 iterations for QP, 15.5 for Minimax)
- $M=10$: 100.0% success (avg 5.5 iterations for QP, 18.5 for Minimax)
The solver converged to machine precision; the violations are **inherent to the mathematical formulation**, not optimization failure.

---

## 2. Logic Chain

1. **Step 1 (Fundamental Geometrical Barrier / Farkas' Lemma):**  
   From Section 1.4, when client gradients form a regular simplex (e.g. 3 vectors at $120^\circ$ angles in $\mathbb{R}^2$), the origin is in the interior of their convex hull: $0 \in \text{int}(\text{Conv}(g_1, \dots, g_M))$.  
   By Farkas' Lemma / Gordan's Theorem, the system $\langle d, g_i \rangle > 0 \;\forall i \in [M]$ has a solution $d \in \mathbb{R}^P$ **if and only if** $0 \notin \text{Conv}(g_1, \dots, g_M)$.  
   Therefore, when $0 \in \text{Conv}(g_1, \dots, g_M)$, no non-zero common descent direction exists in the universe. Theorems 3 & 4 failed to include this necessary prerequisite.

2. **Step 2 (Sequential Projection Destabilization in DR-PCGrad):**  
   In `dr_pcgrad` (`fed_lunar/federated/strategy.py`, lines 174–179), client $i$'s gradient $g_i^{\text{proj}}$ is sequentially projected against peer gradients $j \in \text{peers}(i)$.  
   When $M \ge 3$, projecting $g_i^{\text{proj}}$ onto $g_{j_2}$ rotates $g_i^{\text{proj}}$ and systematically destroys the non-negativity achieved during the earlier projection onto $g_{j_1}$.  
   Furthermore, the aggregated update is the average $g_{\text{aligned}} = \frac{1}{M} \sum_i g_i^{\text{proj}}$. Even if individual projected vectors had non-negative inner products with their specific targets, their linear combination $\langle \sum_i g_i^{\text{proj}}, g_k \rangle$ has no guarantee of non-negativity. This directly accounts for the 38.1% to 67.0% failure rate observed in Section 1.3.

3. **Step 3 (Formulation Flaw in DR-CAGrad Dual Simplex QP):**  
   In `dr_cagrad` (`fed_lunar/federated/strategy.py`, lines 255–291), the implementation solves:  
   $$\min_\alpha \frac{1}{2} \|g_0 + \sum_{i=1}^M \alpha_i g_i\|_2^2 \quad \text{s.t.} \quad \alpha \ge 0, \;\sum \alpha_i = \phi$$  
   The gradient of this objective with respect to $\alpha_i$ is $\langle g_{\text{aligned}}, g_i \rangle$. Minimizing this objective drives $\alpha_i$ to push $g_{\text{aligned}}$ in directions that minimize norm, effectively opposing the dominant gradient components.  
   When scale disparities exist ($\|g_1\| = 10^4 \|g_2\|$, Case 3), $g_0$ is dominated by $g_1$. Setting $\alpha \ge 0$ cannot pull $g_{\text{aligned}}$ away from $g_1$, causing $g_{\text{aligned}}$ to remain closely aligned with $g_1$ and yielding a massive negative inner product with $g_2$ ($\langle g_{\text{aligned}}, g_2 \rangle = -2999.38$).

4. **Step 4 (Invalidation of Theorem 4 Monotonic Decentralized Convergence):**  
   Theorem 4 proves monotonic loss descent $\mathcal{L}_i(\theta_{t+1}) < \mathcal{L}_i(\theta_t)$ relying strictly on the premise $\langle g_i, g^* \rangle \ge \gamma_{\min} > 0$ (line 397).  
   Because $\langle g^*, g_i \rangle < 0$ in over 57%–71% of randomized trials (and down to $-1813.44$), the update $\theta_{t+1} = \theta_t - \eta g^*$ directly causes loss *increase* ($\mathcal{L}_i(\theta_{t+1}) > \mathcal{L}_i(\theta_t)$) for the conflicting clients. Strict monotonic convergence does not hold.

---

## 3. Caveats

1. **2-Client / 1D Opposing Manifolds:**  
   In the specific case where client conflict is restricted to a single opposing dimension with balanced norms (Section 1.4, `cluster_opposing`), DR-PCGrad and DR-CAGrad (Minimax) achieved 0% violations. The failure occurs predominantly when conflict spans multiple non-collinear dimensions ($M \ge 3$) or exhibits scale disparities.
2. **Gradient Normalization:**  
   In this benchmark, gradients were fed into DROGA with raw Euclidean norms (as output by typical neural net backward passes). Normalizing client gradients to unit spheres ($\tilde{g}_i = g_i / \|g_i\|$) prior to projection reduces scale disparity violations, but does not resolve the geometric barrier of Farkas' Lemma for overcomplete simplexes.
3. **Zero Collapse Fallback:**  
   When gradients are perfectly collinear and opposing ($g_1 = -g_2$), both algorithms output $g_{\text{aligned}} = 0$. While technically satisfying $\langle 0, g_i \rangle = 0 \ge -10^{-5}$, this represents complete optimization stagnation (zero descent) rather than active progress.

---

## 4. Conclusion & Verdict

### Final Verdict: **`DISPROVEN`**

**Summary Assessment:**  
The universal claim that DROGA (DR-PCGrad and DR-CAGrad) unconditionally guarantees non-conflicting descent ($\langle g_{\text{aligned}}, g_i \rangle \ge 0$) and decentralized monotonic loss descent across arbitrary non-IID IoT federations is mathematically and empirically **disproven**.
- DR-PCGrad violates the descent condition in **38.1% to 67.0%** of trials for $M \in \{3, 5, 8, 10\}$ (worst IP: $-508.14$).
- DR-CAGrad violates the descent condition in **63.6% to 71.5%** of trials (worst IP: $-1813.44$; up to $-2999.38$ under scale disparity).

### Required Scientific Recalibration for Milestones M2–M5:
1. **Recalibrate Theorem 3 Statement:**  
   Restrict Theorem 3 with the explicit geometric condition:  
   *"Assuming the client gradient convex hull does not contain the origin ($0 \notin \text{Conv}(g_1, \dots, g_M)$) and client gradients are normalized to equal scale..."*
2. **Mitigate Scale Disparities in `strategy.py`:**  
   Incorporate unit-norm scaling $\tilde{g}_i = g_i / (\|g_i\| + \epsilon)$ prior to Gram matrix computation in `dr_cagrad` to eliminate the $-2999.38$ scale disparity failure mode.
3. **Recalibrate Theorem 4 Convergence:**  
   Acknowledge that when $0 \in \text{Conv}(g_1, \dots, g_M)$, Pareto stationary points require tradeoff descent (e.g. weighted average) rather than strict simultaneous monotonic loss reduction for every individual client.

---

## 5. Verification Method

To independently reproduce and verify all observations and numerical results:

1. **Run the Independent Empirical Stress Harness:**
   ```powershell
   cd d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST
   python tests/stress_droga_harness.py
   ```
2. **Inspect Detailed Scenario Breakdown Script:**
   ```powershell
   python -c "import sys; sys.path.insert(0, '.'); from tests.stress_droga_harness import generate_antagonistic_gradients, evaluate_trial; import numpy as np; rng = np.random.default_rng(2026); [print(s, [evaluate_trial(generate_antagonistic_gradients(3, 50, rng, s))['ca_violation'] for _ in range(20)].count(True)) for s in ['simplex', 'cluster_opposing', 'multi_axis_antagonistic']]"
   ```
3. **Invalidation Condition:**  
   This challenge would be invalidated if a mathematical formulation or parameter setting of DROGA is demonstrated that achieves 0 violations ($\min_i \langle g_{\text{aligned}}, g_i \rangle \ge -10^{-5}$) across 1,000 randomized trials on the regular simplex configuration ($M \ge 3$) without collapsing to the zero vector ($g_{\text{aligned}} \ne 0$). By Farkas' Lemma, such a demonstration is mathematically impossible.
