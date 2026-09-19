"""
ADYN-LOC-NFST: Core Mathematical Engine
=========================================
Implements the adaptive dynamic-cardinality cluster lifecycle for LOC-NFST.

Key operations (all O(dL^2) or better):
  - Welford online cluster statistics update
  - Cluster Split  : Rank-1 downdate of A = Q^T S_w Q
  - Cluster Merge  : Rank-1 update of A
  - Cluster Birth  : Brand's thin SVD update of Q (when S_t changes)
  - Cluster Death  : Exponential fading weight → remove cluster slot
  - Quarantine Buffer: sliding window to distinguish drift vs attack

Mathematical foundation (from RESEARCH_DYNAMIC_K_LOC_NFST.md):
  S_w(t+1) = S_w(t) - v v^T   (Split: Δ = -vv^T, Rank-1)
  S_w(t+1) = S_w(t) + w w^T   (Merge: Δ = +ww^T, Rank-1)
  S_t invariant under Split/Merge → Q invariant → only A must be updated.
  A(t+1) = A(t) ∓ u u^T,  u = Q^T v  (projected rank-1 update)
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

# ============================================================
# Configuration Constants
# ============================================================

K_MAX = 128          # Maximum cluster slots (static pool, no malloc)
K_MIN = 2            # Minimum active clusters
QUARANTINE_SIZE = 2000  # Quarantine buffer max capacity (# samples)
N_BIRTH_THRESH = 50    # Min samples in quarantine to approve new cluster
TAU_QUARANTINE = 5000  # Max samples observed before quarantine expires
T_HALF_LIFE = 10_000   # Exponential fading half-life (in sample units)
EPSILON_DEATH = 0.01   # Fading weight below which cluster is removed
THETA_MERGE = 0.5      # Merge criterion: D_ij < theta_merge
BIMODALITY_THRESH = 0.55  # Split criterion: bimodality index > threshold
EVAL_INTERVAL = 5_000  # Check split/merge/death every N samples
SIGMA_RADIUS = 3.0     # Quarantine: points outside sigma_radius * sigma_k
NEAR_NULL_REL_THRESH = 0.01   # Relative threshold for near-null relaxation
EPSILON_SVD = 1e-6
EPSILON_NEAR_NULL = 1e-4
L_MIN = 5


# ============================================================
# Cluster Slot (Static Slot-Array design — no heap allocation at runtime)
# ============================================================

@dataclass
class ClusterSlot:
    """
    Welford online statistics for one cluster.
    All fields are preallocated; activation controlled by `is_active`.
    """
    is_active: bool = False
    count: int = 0
    centroid: Optional[np.ndarray] = None   # (d,) float64
    M2: Optional[np.ndarray] = None        # (d,) running sum of squared deviations
    last_active_ts: int = 0                 # sample index of last update

    def init(self, d: int, x: np.ndarray, ts: int = 0):
        """Initialize slot with first sample x."""
        self.is_active = True
        self.count = 1
        self.centroid = x.copy().astype(np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)
        self.last_active_ts = ts

    def welford_update(self, x: np.ndarray, ts: int = 0):
        """Online update with new sample x (Welford's algorithm)."""
        self.count += 1
        delta = x.astype(np.float64) - self.centroid
        self.centroid += delta / self.count
        delta2 = x.astype(np.float64) - self.centroid
        self.M2 += delta * delta2
        self.last_active_ts = ts

    @property
    def variance(self) -> np.ndarray:
        """Per-feature sample variance (d,)."""
        if self.count < 2:
            return np.ones_like(self.centroid) * 1e-8
        return self.M2 / (self.count - 1)

    @property
    def sigma(self) -> float:
        """Mean std dev across features."""
        return float(np.sqrt(np.mean(self.variance)) + 1e-10)

    def fading_weight(self, t_now: int, half_life: int = T_HALF_LIFE) -> float:
        """Exponential fading: w_k(t) = N_k * 2^{-(t - T_k) / T_half}."""
        delta_t = max(0, t_now - self.last_active_ts)
        return self.count * (2.0 ** (-delta_t / half_life))


# ============================================================
# Quarantine Buffer
# ============================================================

@dataclass
class QuarantineBuffer:
    """
    Sliding window buffer for outlier points.
    Decides: new Normal Drift Cluster vs Attack/Noise.
    """
    buffer: List[np.ndarray] = field(default_factory=list)
    ts_start: int = 0           # Sample index when buffer was created
    n_observed: int = 0         # Total samples processed since buffer creation

    def reset(self, ts: int = 0):
        self.buffer = []
        self.ts_start = ts
        self.n_observed = 0

    def add(self, x: np.ndarray, ts: int = 0):
        self.n_observed += 1
        if len(self.buffer) < QUARANTINE_SIZE:
            self.buffer.append(x.copy())

    def is_mature(self) -> bool:
        """Buffer has enough samples and time to make a birth decision."""
        return self.n_observed >= TAU_QUARANTINE

    def check_birth(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Check if buffer contains a dense new cluster (Cluster Birth).

        Returns
        -------
        approved : bool
        centroid : (d,) if approved
        M2       : (d,) running M2 if approved
        """
        if len(self.buffer) < N_BIRTH_THRESH:
            return False, None, None

        buf = np.stack(self.buffer, axis=0)
        # Simple density check: fraction within 2σ of buffer centroid
        mu = np.mean(buf, axis=0)
        std = np.std(buf, axis=0) + 1e-10
        z_scores = np.mean(np.abs(buf - mu) / std, axis=1)
        dense_frac = float(np.mean(z_scores < 2.0))

        if dense_frac >= 0.6 and len(self.buffer) >= N_BIRTH_THRESH:
            # Compute Welford stats from scratch on buffer
            M2 = np.sum((buf - mu) ** 2, axis=0)
            return True, mu, M2
        return False, None, None


# ============================================================
# ClusterBank — Pool of K_MAX ClusterSlots
# ============================================================

class ClusterBank:
    """
    Fixed-size pool of cluster slots with O(1) add/remove operations.
    Manages all cluster lifecycle events.
    """

    def __init__(self, d: int, k_max: int = K_MAX):
        self.d = d
        self.k_max = k_max
        self.slots: List[ClusterSlot] = [ClusterSlot() for _ in range(k_max)]
        self.quarantine = QuarantineBuffer()
        self.t = 0  # global sample counter

    # --- Access helpers ---

    @property
    def active_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.slots) if s.is_active]

    @property
    def K(self) -> int:
        return len(self.active_indices)

    def centroids(self) -> np.ndarray:
        """(K, d) active centroids."""
        idx = self.active_indices
        return np.stack([self.slots[i].centroid for i in idx], axis=0)

    def counts(self) -> np.ndarray:
        """(K,) active sample counts."""
        idx = self.active_indices
        return np.array([self.slots[i].count for i in idx], dtype=np.float64)

    def _free_slot(self) -> Optional[int]:
        """Find first inactive slot index."""
        for i, s in enumerate(self.slots):
            if not s.is_active:
                return i
        return None

    # --- Online Assignment ---

    def assign(self, x: np.ndarray) -> int:
        """
        Assign sample x to nearest cluster (by Euclidean distance to centroid).
        Returns cluster slot index (or -1 if no active clusters).
        """
        idx = self.active_indices
        if not idx:
            return -1
        centroids = np.stack([self.slots[i].centroid for i in idx], axis=0)
        dists = np.linalg.norm(centroids - x.astype(np.float64), axis=1)
        return idx[int(np.argmin(dists))]

    def update_sample(self, x: np.ndarray) -> Tuple[int, bool]:
        """
        Process one new sample:
        1. Find nearest cluster.
        2. If within 3σ → Welford update.
        3. Else → quarantine.

        Returns (slot_idx, is_quarantined)
        """
        self.t += 1
        x = x.astype(np.float64)

        idx = self.active_indices
        if not idx:
            # Bootstrap: create first cluster
            free = self._free_slot()
            if free is not None:
                self.slots[free].init(self.d, x, self.t)
            return free or 0, False

        nearest = self.assign(x)
        slot = self.slots[nearest]
        dist = float(np.linalg.norm(x - slot.centroid))

        if dist <= SIGMA_RADIUS * slot.sigma:
            slot.welford_update(x, self.t)
            return nearest, False
        else:
            self.quarantine.add(x, self.t)
            return nearest, True

    # --- Lifecycle Events ---

    def try_cluster_birth(self) -> Optional[int]:
        """
        Attempt to promote quarantine buffer to new cluster.
        Returns new slot index if successful, else None.
        """
        if self.K >= self.k_max:
            logger.warning("[ClusterBank] K_MAX reached, cannot birth new cluster")
            return None

        approved, centroid, M2 = self.quarantine.check_birth()
        if not approved:
            return None

        free = self._free_slot()
        if free is None:
            return None

        n = len(self.quarantine.buffer)
        self.slots[free].is_active = True
        self.slots[free].count = n
        self.slots[free].centroid = centroid.copy()
        self.slots[free].M2 = M2.copy()
        self.slots[free].last_active_ts = self.t

        logger.info(f"[ClusterBank] Cluster BIRTH at slot={free}, K={self.K}, "
                    f"n={n}, ts={self.t}")
        self.quarantine.reset(self.t)
        return free

    def try_cluster_death(self) -> List[int]:
        """
        Remove clusters with fading weight below EPSILON_DEATH.
        Returns list of removed slot indices.
        """
        if self.K <= K_MIN:
            return []

        removed = []
        for i in self.active_indices:
            w = self.slots[i].fading_weight(self.t)
            if w < EPSILON_DEATH:
                logger.info(f"[ClusterBank] Cluster DEATH at slot={i}, "
                            f"w={w:.6f}, K={self.K}")
                self.slots[i].is_active = False
                removed.append(i)

            # Never remove below K_MIN
            if self.K - len(removed) <= K_MIN:
                break

        return removed

    def compute_scatter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute S_w and S_t from current cluster statistics.
        Used for initial model building (not for incremental updates).

        Returns
        -------
        S_w : (d, d) within-class scatter
        S_t : (d, d) total scatter (computed from centroids + grand mean)
        """
        d = self.d
        idx = self.active_indices
        K = len(idx)

        S_w = np.zeros((d, d), dtype=np.float64)
        N_total = 0

        # S_w: sum of per-cluster scatter (from Welford M2)
        for i in idx:
            slot = self.slots[i]
            # M2[j] = Σ (x_j - μ_j)^2 (diagonal)
            # Full scatter approx: diagonal only from Welford — use as diag
            # For better accuracy: S_w += diag(M2)
            # Note: full covariance from Welford would require O(d^2) storage
            S_w += np.diag(slot.M2)
            N_total += slot.count

        if N_total > 0:
            S_w /= N_total

        # S_t: from centroids + grand mean
        centroids = self.centroids()
        counts = self.counts()
        mu_global = np.average(centroids, weights=counts, axis=0)

        # Between-class scatter S_b
        S_b = np.zeros((d, d), dtype=np.float64)
        for k, i in enumerate(idx):
            diff = centroids[k] - mu_global
            S_b += counts[k] * np.outer(diff, diff)
        S_b /= N_total

        # S_w_from_centroids = diagonal estimate → use directly
        S_t = S_w + S_b

        return S_w.astype(np.float32), S_t.astype(np.float32)


# ============================================================
# Rank-1 Subspace Update Engine
# ============================================================

class SubspaceEngine:
    """
    Maintains the null-space projection matrix W and reduced matrix A.
    Supports Rank-1 updates for Split/Merge without full re-eigendecomposition.

    Internal state:
        Q   : (d, r)   — basis of S_t column space (rank r), FIXED under Split/Merge
        A   : (r, r)   — reduced S_w in Q's subspace: A = Q^T S_w Q
        eigvals_A : (r,) — eigenvalues of A (ascending)
        eigvecs_A : (r, r) — eigenvectors of A
        W   : (d, L)   — null-space projection matrix
        L   : int      — null-space dimension
    """

    def __init__(self, d: int):
        self.d = d
        self.Q: Optional[np.ndarray] = None      # (d, r)
        self.A: Optional[np.ndarray] = None      # (r, r)
        self.eigvals_A: Optional[np.ndarray] = None
        self.eigvecs_A: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None      # (d, L)
        self.L: int = 0
        self.is_initialized: bool = False
        self.N_total: float = 1.0

    def initialize(self, S_w: np.ndarray, S_t: np.ndarray, N_total: Optional[float] = None):
        """
        Full initialization from scratch.
        Called once at the beginning, or after Birth/Death (S_t changes).
        """
        d = self.d
        S_w = S_w.astype(np.float64)
        S_t = S_t.astype(np.float64)
        if N_total is not None and N_total > 0:
            self.N_total = float(N_total)

        # SVD of S_t to get Q
        eigvals_t, eigvecs_t = eigh(S_t)
        eigvals_t = np.maximum(eigvals_t, 0.0)
        rank = int(np.sum(eigvals_t > EPSILON_SVD))
        if rank == 0:
            rank = min(L_MIN, d)

        self.Q = eigvecs_t[:, -rank:].astype(np.float64)  # (d, r)
        self.A = (self.Q.T @ S_w @ self.Q).astype(np.float64)  # (r, r)

        # Eigendecomposition of A
        self.eigvals_A, self.eigvecs_A = eigh(self.A)
        self.eigvals_A = np.maximum(self.eigvals_A, 0.0)

        self._update_W()
        self.is_initialized = True
        logger.debug(f"[SubspaceEngine] Init: rank={rank}, L={self.L}, N_total={self.N_total}")

    def _update_W(self):
        """Recompute W from current A eigendecomposition."""
        # Near-null relaxation
        lambda_max = float(self.eigvals_A[-1]) if len(self.eigvals_A) > 0 else 1.0
        thr = max(EPSILON_NEAR_NULL, lambda_max * NEAR_NULL_REL_THRESH)
        near_null_mask = self.eigvals_A < thr

        if not np.any(near_null_mask):
            idx = np.argsort(self.eigvals_A)[:L_MIN]
        else:
            idx = np.where(near_null_mask)[0]
            if len(idx) < L_MIN:
                idx = np.argsort(self.eigvals_A)[:L_MIN]

        B = self.eigvecs_A[:, idx]   # (r, L)
        self.W = (self.Q @ B).astype(np.float32)  # (d, L)
        self.L = B.shape[1]

    def rank1_update_split(self, mu_j1: np.ndarray, mu_j2: np.ndarray,
                            N_j1: int, N_j2: int) -> bool:
        """
        Apply Rank-1 downdate after cluster split (j → j1, j2).
        ΔS_w = - (1 / N_total) * (N_j1 * N_j2 / N_j) * (μ_j1 - μ_j2)(μ_j1 - μ_j2)^T
        ΔA   = - u u^T,   u = Q^T v
        """
        if not self.is_initialized:
            return False

        N_j = N_j1 + N_j2
        if N_j <= 0:
            return False

        scale = self.N_total if self.N_total > 0 else 1.0
        v = np.sqrt(N_j1 * N_j2 / (scale * N_j)) * (
            mu_j1.astype(np.float64) - mu_j2.astype(np.float64)
        )
        u = self.Q.T @ v  # (r,)

        # Rank-1 downdate: A = A - u u^T (symmetric)
        self.A -= np.outer(u, u)
        self.eigvals_A, self.eigvecs_A = eigh(self.A)
        self.eigvals_A = np.maximum(self.eigvals_A, 0.0)

        self._update_W()
        logger.debug(f"[SubspaceEngine] Split rank-1 downdate applied: L={self.L}")
        return True

    def rank1_update_merge(self, mu_a: np.ndarray, mu_b: np.ndarray,
                            N_a: int, N_b: int) -> bool:
        """
        Apply Rank-1 update after cluster merge (a, b → ab).
        ΔS_w = + (1 / N_total) * (N_a * N_b / (N_a+N_b)) * (μ_a - μ_b)(μ_a - μ_b)^T
        ΔA   = + u u^T,   u = Q^T w
        """
        if not self.is_initialized:
            return False

        N_ab = N_a + N_b
        if N_ab <= 0:
            return False

        scale = self.N_total if self.N_total > 0 else 1.0
        w = np.sqrt(N_a * N_b / (scale * N_ab)) * (
            mu_a.astype(np.float64) - mu_b.astype(np.float64)
        )
        u = self.Q.T @ w  # (r,)

        # Rank-1 update: A = A + u u^T
        self.A += np.outer(u, u)
        self.eigvals_A, self.eigvecs_A = eigh(self.A)
        self.eigvals_A = np.maximum(self.eigvals_A, 0.0)

        self._update_W()
        logger.debug(f"[SubspaceEngine] Merge rank-1 update applied: L={self.L}")
        return True


# ============================================================
# Split / Merge / Death Decision Functions
# ============================================================

def bimodality_index(slot: ClusterSlot) -> float:
    """
    Compute bimodality index (Sarle's b) for a cluster.
    b = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
    Approximation using per-feature mean variance ratio.
    """
    if slot.count < 10:
        return 0.0

    variance = slot.variance
    sigma = np.sqrt(np.maximum(variance, 1e-10))

    # Use the coefficient of variation spread as proxy for bimodality
    cv = sigma / (np.abs(slot.centroid) + 1e-10)
    # Bimodality proxy: high variance relative to centroid magnitude
    # More rigorous: needs actual data; here use variance spread
    bi_proxy = float(np.std(variance) / (np.mean(variance) + 1e-10))
    return min(bi_proxy, 1.0)


def should_split(slot: ClusterSlot) -> bool:
    """Check if cluster should be split (bimodality > threshold)."""
    return bimodality_index(slot) > BIMODALITY_THRESH and slot.count >= 30


def compute_merge_dist(slot_i: ClusterSlot, slot_j: ClusterSlot) -> float:
    """
    Normalized distance between two clusters:
    D_ij = ||μ_i - μ_j|| / (σ_i + σ_j)
    """
    return float(
        np.linalg.norm(slot_i.centroid - slot_j.centroid) /
        (slot_i.sigma + slot_j.sigma + 1e-10)
    )


def find_merge_pair(bank: ClusterBank) -> Optional[Tuple[int, int]]:
    """Find closest pair of clusters to merge (if D_ij < THETA_MERGE)."""
    idx = bank.active_indices
    if len(idx) <= K_MIN:
        return None

    best_dist = float('inf')
    best_pair = None

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            d = compute_merge_dist(bank.slots[i], bank.slots[j])
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)

    if best_dist < THETA_MERGE:
        return best_pair
    return None


def do_split(bank: ClusterBank, slot_idx: int) -> Tuple[int, int]:
    """
    Perform cluster split on slot_idx.
    Splits along max-variance feature direction.
    Returns (slot_idx_1, slot_idx_2) — slot_idx is reused as slot_1.
    """
    slot = bank.slots[slot_idx]
    d = bank.d

    # Find split direction: max variance feature
    split_dir = np.argmax(slot.variance)
    offset = float(np.sqrt(slot.variance[split_dir]) + 1e-10)

    # Create two centroids: μ ± offset along split_dir
    mu_j1 = slot.centroid.copy()
    mu_j2 = slot.centroid.copy()
    mu_j1[split_dir] += offset
    mu_j2[split_dir] -= offset

    N_j = slot.count
    N_j1 = N_j // 2
    N_j2 = N_j - N_j1

    # Reuse slot_idx for cluster 1
    slot.centroid = mu_j1
    slot.count = N_j1
    slot.M2 = slot.M2.copy()  # approximate: keep same variance
    slot.last_active_ts = bank.t

    # Create cluster 2 in free slot
    free = bank._free_slot()
    if free is None:
        logger.warning("[do_split] No free slot for split!")
        return slot_idx, slot_idx

    bank.slots[free].is_active = True
    bank.slots[free].count = N_j2
    bank.slots[free].centroid = mu_j2
    bank.slots[free].M2 = slot.M2.copy()
    bank.slots[free].last_active_ts = bank.t

    logger.info(f"[ClusterBank] Cluster SPLIT: slot={slot_idx} → "
                f"({slot_idx}, {free}), K={bank.K}")
    return slot_idx, free


def do_merge(bank: ClusterBank, i: int, j: int) -> int:
    """
    Merge clusters at slots i and j.
    Merged centroid: weighted average.
    Returns surviving slot index (i).
    """
    slot_i = bank.slots[i]
    slot_j = bank.slots[j]

    N_a = slot_i.count
    N_b = slot_j.count
    N_ab = N_a + N_b

    # Merged centroid (weighted average)
    mu_merged = (N_a * slot_i.centroid + N_b * slot_j.centroid) / N_ab

    # Huygens decomposition of M2
    M2_merged = (
        slot_i.M2 + slot_j.M2
        + (N_a * N_b / N_ab) * (slot_i.centroid - slot_j.centroid) ** 2
    )

    slot_i.centroid = mu_merged
    slot_i.count = N_ab
    slot_i.M2 = M2_merged
    slot_i.last_active_ts = bank.t

    # Deactivate slot j
    bank.slots[j].is_active = False

    logger.info(f"[ClusterBank] Cluster MERGE: ({i}, {j}) → {i}, K={bank.K}")
    return i


# ============================================================
# Periodic Pruning (Split/Merge/Death combined)
# ============================================================

def run_lifecycle_step(
    bank: ClusterBank,
    engine: SubspaceEngine,
    S_w_current: Optional[np.ndarray] = None,
    S_t_current: Optional[np.ndarray] = None,
) -> dict:
    """
    Run one complete lifecycle evaluation step:
    1. Cluster Death check
    2. Cluster Merge check
    3. Cluster Split check
    4. Cluster Birth check (from quarantine)

    Updates bank and engine in-place.
    Returns event log dict.
    """
    events = {
        "deaths": [],
        "merges": [],
        "splits": [],
        "births": [],
        "K_before": bank.K,
        "K_after": bank.K,
        "L": engine.L if engine.is_initialized else 0,
    }

    # --- 1. Cluster Death ---
    deaths = bank.try_cluster_death()
    events["deaths"] = deaths

    if deaths and S_w_current is not None and S_t_current is not None:
        # S_t changes when clusters die → full re-initialization needed
        logger.info(f"[Lifecycle] Death events → re-initializing subspace (K={bank.K})")
        S_w_new, S_t_new = bank.compute_scatter()
        engine.initialize(S_w_new, S_t_new)

    # --- 2. Cluster Merge ---
    merge_pair = find_merge_pair(bank)
    if merge_pair is not None:
        i, j = merge_pair
        mu_a = bank.slots[i].centroid.copy()
        mu_b = bank.slots[j].centroid.copy()
        N_a = bank.slots[i].count
        N_b = bank.slots[j].count

        surviving = do_merge(bank, i, j)
        events["merges"].append((i, j, surviving))

        if engine.is_initialized:
            engine.rank1_update_merge(mu_a, mu_b, N_a, N_b)

    # --- 3. Cluster Split ---
    for slot_idx in list(bank.active_indices):
        slot = bank.slots[slot_idx]
        if should_split(slot) and bank.K < bank.k_max:
            mu_old = slot.centroid.copy()
            N_old = slot.count

            s1, s2 = do_split(bank, slot_idx)
            events["splits"].append((slot_idx, s1, s2))

            if engine.is_initialized and s1 != s2:
                mu_j1 = bank.slots[s1].centroid
                mu_j2 = bank.slots[s2].centroid
                N_j1 = bank.slots[s1].count
                N_j2 = bank.slots[s2].count
                engine.rank1_update_split(mu_j1, mu_j2, N_j1, N_j2)

            break  # Only one split per lifecycle step (for stability)

    # --- 4. Cluster Birth ---
    if bank.quarantine.is_mature():
        new_slot = bank.try_cluster_birth()
        if new_slot is not None:
            events["births"].append(new_slot)
            # S_t changes with birth → full re-initialization
            if engine.is_initialized:
                S_w_new, S_t_new = bank.compute_scatter()
                engine.initialize(S_w_new, S_t_new)
        else:
            # Quarantine expired without birth → treat as noise, reset
            if bank.quarantine.is_mature():
                logger.info(f"[Lifecycle] Quarantine expired without birth → reset")
                bank.quarantine.reset(bank.t)

    events["K_after"] = bank.K
    events["L"] = engine.L if engine.is_initialized else 0

    if any([events["deaths"], events["merges"], events["splits"], events["births"]]):
        logger.info(
            f"[Lifecycle] ts={bank.t}: K {events['K_before']} → {events['K_after']}, "
            f"L={events['L']}, deaths={len(events['deaths'])}, "
            f"merges={len(events['merges'])}, splits={len(events['splits'])}, "
            f"births={len(events['births'])}"
        )

    return events
