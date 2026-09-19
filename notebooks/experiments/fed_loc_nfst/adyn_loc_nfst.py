"""
ADYN-LOC-NFST: Adaptive Dynamic-Cardinality LOC-NFST
======================================================
Approach A — Offline Temporal Simulation with Concept Drift.

Concept drift is simulated by:
  1. Splitting training data into T temporal chunks (preserving order).
  2. Running LOC-NFST on each chunk sequentially, allowing K(t) to adapt.
  3. At each checkpoint, the model adapts to new normal patterns:
     - New dense regions in quarantine → Cluster Birth
     - Collapsed clusters → Cluster Merge
     - Bimodal clusters → Cluster Split
     - Stale clusters → Cluster Death (exponential fading)

The key comparison is:
  Static-K LOC-NFST (K fixed at start) vs ADYN-LOC-NFST (K adaptive)
  Measured: AUC-ROC, FAR (False Alarm Rate), K(t), L(t) over time.

Usage:
    from fed_loc_nfst.adyn_loc_nfst import AdynLOCNFST

    model = AdynLOCNFST(K_init=20, n_chunks=5)
    history = model.fit_temporal(X_train, chunk_ids=None)
    y_proba = model.predict(X_test)
"""
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh, null_space
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from .adyn_core import (
    ClusterBank, SubspaceEngine, run_lifecycle_step,
    EVAL_INTERVAL, L_MIN, EPSILON_SVD, EPSILON_NEAR_NULL, NEAR_NULL_REL_THRESH
)
from .evaluate import project_to_null, min_dist_to_centers, compute_fl_scores, evaluate_scores

logger = logging.getLogger(__name__)


# ============================================================
# Static-K LOC-NFST (Baseline for comparison)
# ============================================================

def fit_static_loc_nfst(
    X_train: np.ndarray,
    K: int,
    seed: int = 42,
    epsilon_svd: float = EPSILON_SVD,
    epsilon_near_null: float = EPSILON_NEAR_NULL,
    L_min: int = L_MIN,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fit static K LOC-NFST on the full training set.

    Returns
    -------
    W            : (d, L) null-space projection matrix
    null_centers : (K, L) projected cluster centers
    max_train    : float
    L            : int (null-space dimension)
    """
    N, d = X_train.shape
    K = min(K, N)

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    y = km.fit_predict(X_train)
    centroids = km.cluster_centers_.astype(np.float64)
    classes = np.unique(y)

    # S_w
    S_w = np.zeros((d, d), dtype=np.float64)
    for cls in classes:
        X_cls = X_train[y == cls].astype(np.float64)
        mu_cls = centroids[cls]
        diff = X_cls - mu_cls
        S_w += diff.T @ diff
    S_w /= N

    # S_t
    mu_total = np.mean(X_train, axis=0).astype(np.float64)
    diff_t = X_train.astype(np.float64) - mu_total
    S_t = (diff_t.T @ diff_t) / N

    # SVD of S_t
    eigvals_t, eigvecs_t = eigh(S_t)
    eigvals_t = np.maximum(eigvals_t, 0.0)
    rank = int(np.sum(eigvals_t > epsilon_svd))
    if rank == 0:
        rank = min(L_min, d)

    Q = eigvecs_t[:, -rank:]
    A = Q.T @ S_w @ Q

    # Null-space
    B = null_space(A, rcond=epsilon_svd)
    L = B.shape[1]

    if L < L_min:
        eigvals_A, eigvecs_A = eigh(A)
        eigvals_A = np.maximum(eigvals_A, 0.0)
        lambda_max = float(eigvals_A[-1]) if len(eigvals_A) > 0 else 1.0
        thr = max(epsilon_near_null, lambda_max * NEAR_NULL_REL_THRESH)
        near_null_mask = eigvals_A < thr
        if not np.any(near_null_mask):
            idx = np.argsort(eigvals_A)[:L_min]
        else:
            idx = np.where(near_null_mask)[0]
            if len(idx) < L_min:
                idx = np.argsort(eigvals_A)[:L_min]
        B = eigvecs_A[:, idx]
        L = B.shape[1]

    W = (Q @ B).astype(np.float32)

    # Null centers
    null_train = X_train.astype(np.float32) @ W
    null_centers = np.array([
        np.mean(null_train[y == cls], axis=0) for cls in classes
    ], dtype=np.float32)

    train_dists = min_dist_to_centers(null_train, null_centers)
    max_train = float(np.max(train_dists)) if len(train_dists) > 0 else 1.0
    if max_train < 1e-10:
        max_train = 1.0

    return W, null_centers, max_train, L


def evaluate_static(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    K: int,
    seed: int = 42,
) -> dict:
    """Full pipeline: fit static K model, evaluate on test set."""
    t0 = time.time()
    W, null_centers, max_train, L = fit_static_loc_nfst(X_train, K, seed=seed)
    train_time = time.time() - t0

    t1 = time.time()
    y_proba = compute_fl_scores(X_test, W, null_centers, max_train)
    infer_time = time.time() - t1

    metrics = evaluate_scores(y_test, y_proba)
    return {
        "method": f"Static-K(K={K})",
        "K_init": K,
        "K_final": K,
        "L": L,
        "train_time_s": round(train_time, 3),
        "infer_time_s": round(infer_time, 3),
        **metrics,
    }


# ============================================================
# ADYN-LOC-NFST: Adaptive model
# ============================================================

class AdynLOCNFST:
    """
    Adaptive Dynamic-Cardinality LOC-NFST.

    Parameters
    ----------
    K_init      : Initial number of clusters (fitted by K-Means at start)
    n_chunks    : Number of temporal chunks to simulate concept drift
    eval_every  : Run lifecycle evaluation every N samples within a chunk
    seed        : Random seed
    """

    def __init__(
        self,
        K_init: int = 20,
        n_chunks: int = 5,
        eval_every: int = EVAL_INTERVAL,
        seed: int = 42,
    ):
        self.K_init = K_init
        self.n_chunks = n_chunks
        self.eval_every = eval_every
        self.seed = seed

        self.bank: Optional[ClusterBank] = None
        self.engine: Optional[SubspaceEngine] = None
        self.history: List[dict] = []

        # Final model state
        self.W: Optional[np.ndarray] = None
        self.null_centers: Optional[np.ndarray] = None
        self.max_train: Optional[float] = None
        self.L: int = 0
        self.K_final: int = 0

    def _init_from_chunk(self, X_chunk: np.ndarray) -> None:
        """
        Initialize model from raw data using K-Means + full d×d scatter solve.
        EXACTLY mirrors compute_centralized_NPD from run_centralized.py.
        """
        d = X_chunk.shape[1]
        N = len(X_chunk)
        K = min(self.K_init, N)

        np.random.seed(self.seed)
        km = KMeans(n_clusters=K, random_state=self.seed, n_init=10)
        y = km.fit_predict(X_chunk)
        centroids = km.cluster_centers_.astype(np.float64)
        classes = np.unique(y)

        # ── Full d×d scatter computation (centralized path) ──
        S_w = np.zeros((d, d), dtype=np.float64)
        for cls in classes:
            X_cls = X_chunk[y == cls].astype(np.float64)
            mu_cls = centroids[cls]
            diff = X_cls - mu_cls
            S_w += diff.T @ diff
        S_w /= N

        mu_total = np.mean(X_chunk, axis=0).astype(np.float64)
        diff_t = X_chunk.astype(np.float64) - mu_total
        S_t = (diff_t.T @ diff_t) / N

        # ── Initialize subspace engine with full scatter ──
        self.engine = SubspaceEngine(d=d)
        self.engine.initialize(S_w.astype(np.float32), S_t.astype(np.float32), N_total=N)

        # ── Compute null centers from raw projected training points (centralized logic) ──
        W = self.engine.W   # (d, L)
        null_train = X_chunk.astype(np.float32) @ W   # (N, L)
        null_centers_init = np.array([
            np.mean(null_train[y == cls], axis=0) for cls in classes
        ], dtype=np.float32)   # (K, L)

        train_dists = min_dist_to_centers(null_train, null_centers_init)
        max_train_init = float(np.max(train_dists)) if len(train_dists) > 0 else 1.0
        if max_train_init < 1e-10:
            max_train_init = 1.0

        # Store initial model state (used when no lifecycle event has fired)
        self.W = W
        self.L = self.engine.L
        self.null_centers = null_centers_init
        self.max_train = max_train_init

        # ── Initialize ClusterBank with K-Means centroids (for online tracking) ──
        self.bank = ClusterBank(d=d)
        for cls in classes:
            free = self.bank._free_slot()
            if free is None:
                break
            slot = self.bank.slots[free]
            X_cls = X_chunk[y == cls].astype(np.float64)
            n_cls = len(X_cls)
            mu_cls = centroids[cls]

            slot.is_active = True
            slot.count = n_cls
            slot.centroid = mu_cls.copy()
            slot.M2 = np.sum((X_cls - mu_cls) ** 2, axis=0)
            slot.last_active_ts = 0

        self.K_final = self.bank.K

        logger.info(f"[ADYN] Initialized: K={self.bank.K}, L={self.engine.L}, "
                    f"N={N}, d={d}, max_train={max_train_init:.4f}")

    def _process_chunk_online(self, X_chunk: np.ndarray, chunk_id: int) -> List[dict]:
        """
        Process one chunk sample-by-sample with periodic lifecycle evaluation.
        Returns list of lifecycle event dicts.
        """
        events = []
        n = len(X_chunk)

        for i, x in enumerate(X_chunk):
            _, quarantined = self.bank.update_sample(x)

            # Periodic lifecycle check
            if (i + 1) % self.eval_every == 0:
                S_w, S_t = self.bank.compute_scatter()
                ev = run_lifecycle_step(
                    self.bank, self.engine,
                    S_w_current=S_w,
                    S_t_current=S_t,
                )
                ev["chunk_id"] = chunk_id
                ev["sample_in_chunk"] = i
                events.append(ev)

        return events

    def _finalize_model(self) -> None:
        """
        After all chunks, finalize model state:
        - Preserve W and max_train from Phase 1 full-scatter init.
        - Update null_centers and K_final from current bank centroids.

        max_train MUST come from Phase 1 (computed from all raw training points).
        Recomputing from centroid-centroid distances after lifecycle gives wrong scale.
        """
        if self.engine is None or not self.engine.is_initialized:
            logger.error("[ADYN] Engine not initialized!")
            return
        if self.W is None:
            logger.error("[ADYN] W not set (init failed?)!")
            return

        # Preserve W from engine (updated by Rank-1 ops during lifecycle)
        self.W = self.engine.W
        self.L = self.engine.L

        # Update null centers from current adapted bank centroids
        centroids = self.bank.centroids()  # (K_current, d)
        null_centers_new = centroids.astype(np.float32) @ self.W  # (K_current, L)
        self.null_centers = null_centers_new

        # Keep max_train from Phase 1 (correct scale from raw training data)
        # self.max_train already set in _init_from_chunk — do NOT recompute
        self.K_final = self.bank.K

        logger.info(f"[ADYN] Model finalized: K_final={self.K_final}, "
                    f"L={self.L}, max_train={self.max_train:.6f} (preserved from init)")

    def fit_temporal(
        self,
        X_train: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Fit ADYN-LOC-NFST with temporal chunk concept drift simulation.

        Strategy (academically correct):
        1. Initialize model on ALL training data (K-Means + full scatter solve).
           This gives ADYN the SAME starting point as Static-K.
        2. Simulate concept drift by re-processing each temporal chunk online
           with lifecycle adaptation (Split/Merge/Birth/Death).
        3. At each chunk boundary, record K(t), L(t), and checkpoint AUC.

        Parameters
        ----------
        X_train : (N, d) training data (order preserved for temporal split)
        y_test  : (N_test,) test labels (for per-checkpoint AUC logging)
        X_test  : (N_test, d) test data

        Returns
        -------
        history : list of dicts with per-chunk stats
        """
        N, d = X_train.shape
        chunk_size = N // self.n_chunks
        self.history = []

        logger.info(f"[ADYN] Starting temporal fit: N={N}, d={d}, "
                    f"n_chunks={self.n_chunks}, chunk_size={chunk_size}, "
                    f"K_init={self.K_init}")

        # ── Phase 1: Full-data initialization (same as Static-K baseline) ──
        logger.info(f"[ADYN] Phase 1: Full-data K-Means init (K={self.K_init})")
        self._init_from_chunk(X_train)   # use ALL training data for init
        L_now = self.engine.L if self.engine.is_initialized else 0
        logger.info(f"[ADYN] Full init done: K={self.bank.K}, L={L_now}")

        # ── Phase 2: Online lifecycle pass over temporal chunks ──
        # (simulates how the model adapts as new traffic patterns emerge)
        for chunk_id in range(self.n_chunks):
            t0 = time.time()
            start = chunk_id * chunk_size
            end = start + chunk_size if chunk_id < self.n_chunks - 1 else N
            X_chunk = X_train[start:end]

            logger.info(f"[ADYN] === Chunk {chunk_id+1}/{self.n_chunks}: "
                        f"samples {start}-{end} (n={len(X_chunk)}) ===")

            # Online lifecycle pass
            chunk_events = self._process_chunk_online(X_chunk, chunk_id)

            chunk_time = time.time() - t0
            K_now = self.bank.K
            L_now = self.engine.L if self.engine.is_initialized else 0

            # Optional: compute checkpoint AUC
            checkpoint_auc = None
            checkpoint_far = None
            if X_test is not None and y_test is not None and self.engine.is_initialized:
                self._finalize_model()
                y_proba = self.predict(X_test)
                metrics = evaluate_scores(y_test, y_proba)
                checkpoint_auc = metrics["AUCROC"]
                # FAR = FP / (FP + TN)
                y_pred = (y_proba[:, 1] >= 0.5).astype(int)
                y_true_flipped = (1 - y_test).astype(int)
                tn = np.sum((y_pred == 0) & (y_true_flipped == 1))
                fp = np.sum((y_pred == 1) & (y_true_flipped == 1))
                checkpoint_far = fp / (fp + tn + 1e-10)

            chunk_stat = {
                "chunk_id": chunk_id,
                "n_samples": len(X_chunk),
                "K": K_now,
                "L": L_now,
                "chunk_time_s": round(chunk_time, 3),
                "AUC_checkpoint": checkpoint_auc,
                "FAR_checkpoint": checkpoint_far,
            }
            self.history.append(chunk_stat)
            logger.info(
                f"[ADYN] Chunk {chunk_id+1} done: K={K_now}, L={L_now}, "
                f"time={chunk_time:.2f}s"
                + (f", AUC={checkpoint_auc:.2f}%" if checkpoint_auc else "")
            )

        # Finalize model
        self._finalize_model()
        return self.history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Score test samples using current W, null_centers, max_train.
        Returns y_proba (n_test, 2) — same format as evaluate_scores expects.
        """
        if self.W is None or self.null_centers is None:
            raise RuntimeError("Model not fitted. Call fit_temporal() first.")
        return compute_fl_scores(X_test, self.W, self.null_centers, self.max_train)


# ============================================================
# Comparison Runner
# ============================================================

def run_adyn_vs_static(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    K_static_list: List[int] = [20, 50, 100],
    K_init_adyn: int = 20,
    n_chunks: int = 5,
    seed: int = 42,
    dataset_name: str = "unknown",
    scaler_name: str = "unknown",
) -> List[dict]:
    """
    Run full comparison: Static-K (multiple K values) vs ADYN-LOC-NFST.

    Returns list of result dicts (one per method).
    """
    results = []

    # --- Static-K baselines ---
    for K in K_static_list:
        logger.info(f"[Compare] Static-K={K} on {dataset_name}/{scaler_name}")
        try:
            r = evaluate_static(X_train, X_test, y_test, K=K, seed=seed)
            r.update({"dataset": dataset_name, "scaler": scaler_name})
            results.append(r)
            logger.info(f"[Compare] Static-K={K}: AUC={r['AUCROC']:.2f}%")
        except Exception as e:
            logger.error(f"[Compare] Static-K={K} failed: {e}")

    # --- ADYN-LOC-NFST ---
    logger.info(f"[Compare] ADYN-LOC-NFST (K_init={K_init_adyn}) on "
                f"{dataset_name}/{scaler_name}")
    try:
        model = AdynLOCNFST(K_init=K_init_adyn, n_chunks=n_chunks, seed=seed)
        t0 = time.time()
        history = model.fit_temporal(X_train, y_test=y_test, X_test=X_test)
        train_time = time.time() - t0

        y_proba = model.predict(X_test)
        metrics = evaluate_scores(y_test, y_proba)

        # FAR computation
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
        y_true_flipped = (1 - y_test).astype(int)
        tn = np.sum((y_pred == 0) & (y_true_flipped == 1))
        fp = np.sum((y_pred == 1) & (y_true_flipped == 1))
        far = float(fp / (fp + tn + 1e-10))

        r = {
            "method": f"ADYN-LOC-NFST(K_init={K_init_adyn})",
            "K_init": K_init_adyn,
            "K_final": model.K_final,
            "L": model.L,
            "n_chunks": n_chunks,
            "FAR": round(far, 6),
            "train_time_s": round(train_time, 3),
            "infer_time_s": 0.0,
            "dataset": dataset_name,
            "scaler": scaler_name,
            "drift_events": sum(
                len(h.get("splits", [])) + len(h.get("births", []))
                for h in history if isinstance(h, dict)
            ),
            **metrics,
        }
        results.append(r)
        logger.info(
            f"[Compare] ADYN-LOC-NFST: AUC={metrics['AUCROC']:.2f}%, "
            f"K_init={K_init_adyn}→K_final={model.K_final}, L={model.L}"
        )
    except Exception as e:
        logger.error(f"[Compare] ADYN failed: {e}", exc_info=True)

    return results
