import numpy as np
import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state


class PMKFN:
    """
    p-norm Multiple Kernel One-Class Fisher Null-Space (p MK-FN) baseline.
    Implements Algorithm 1 (alternating alpha/beta) from Rahimzadeh Arashloo (2020). :contentReference[oaicite:2]{index=2}

    One-class training: only normal samples.
    Score: |1 - f(x)| where f(x) = sum_i alpha_i * k(x_i, x).

    Parameters
    ----------
    p : float
        p-norm for kernel weights beta (p >= 1). p=1 gives sparse (select one kernel). :contentReference[oaicite:3]{index=3}
    gamma : Union[str, float, List[Union[str, float]]]
        RBF gamma. If X is a list of views, you can pass list of gammas of same length.
        "scale": 1 / (d * X.var()) computed per view; "auto": 1/d.
    delta : Union[str, float]
        Regularization delta in (delta I + sum_j beta_j K_j)^{-1} 1. :contentReference[oaicite:4]{index=4}
        - float: use provided value
        - "paper": try to compute delta using the paper's sensitivity-based rule (may be costly for large n). :contentReference[oaicite:5]{index=5}
        - "auto": use a cheap, stable heuristic (recommended for larger n).
    max_iter : int
        Max alternating iterations (ignored if only 1 kernel).
    tol : float
        Convergence tolerance on beta.
    max_train_size : Optional[int]
        Cap number of training samples for feasibility (kernel is O(n^2)/O(n^3)).
    random_state : int
        Random seed for subsampling.
    dtype : np.dtype
        float32 is faster/less memory.
    """

    def __init__(
        self,
        p: float = 2.0,
        gamma: Union[str, float, List[Union[str, float]]] = "scale",
        delta: Union[str, float] = "auto",
        max_iter: int = 50,
        tol: float = 1e-6,
        max_train_size: Optional[int] = 3000,
        random_state: int = 42,
        dtype: np.dtype = np.float32,
    ):
        if p < 1:
            raise ValueError("p must be >= 1")
        self.p = float(p)
        self.gamma = gamma
        self.delta = delta
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_train_size = max_train_size
        self.random_state = int(random_state)
        self.dtype = dtype

        # learned
        self.X_views_: Optional[List[np.ndarray]] = None
        self.gamma_: Optional[List[float]] = None
        self.beta_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.delta_: Optional[float] = None

    def _resolve_gamma(self, X: np.ndarray, g: Union[str, float]) -> float:
        d = X.shape[1]
        if isinstance(g, (int, float)):
            return float(g)
        if g == "auto":
            return 1.0 / max(d, 1)
        if g == "scale":
            v = float(np.var(X))
            v = v if v > 0 else 1.0
            return 1.0 / (max(d, 1) * v)
        raise ValueError(f"Unknown gamma: {g}")

    def _compute_kernels(self, X_views: List[np.ndarray]) -> List[np.ndarray]:
        gammas = self.gamma
        if not isinstance(gammas, list):
            gammas = [gammas] * len(X_views)
        if len(gammas) != len(X_views):
            raise ValueError("If gamma is a list, it must match number of views.")

        gamma_vals: List[float] = []
        Ks: List[np.ndarray] = []
        for X, g in zip(X_views, gammas):
            gv = self._resolve_gamma(X, g)
            K = rbf_kernel(X, X, gamma=gv).astype(self.dtype)
            Ks.append(K)
            gamma_vals.append(gv)

        self.gamma_ = gamma_vals
        return Ks

    def _choose_delta(self, Kcombo: np.ndarray) -> float:
        if isinstance(self.delta, (int, float)):
            return float(self.delta)

        if self.delta == "auto":
            # cheap & stable: scaled ridge based on trace
            tr = float(np.trace(Kcombo))
            n = Kcombo.shape[0]
            return max(1e-6, 1e-3 * (tr / max(n, 1)))

        if self.delta == "paper":
            # sensitivity-based delta (can be expensive due to eigendecomp). :contentReference[oaicite:6]{index=6}
            # Add tiny jitter to avoid singularity
            n = Kcombo.shape[0]
            Kj = Kcombo + (1e-8 * np.eye(n, dtype=self.dtype))
            eigs = np.linalg.eigvalsh(Kj.astype(np.float64))
            lam_min = float(max(eigs[0], 1e-12))
            lam_max = float(max(eigs[-1], lam_min))
            kappa = lam_max / lam_min
            a = (kappa + 1.0) / (2.0 * np.sqrt(kappa))
            denom = (a - 1.0)
            if abs(denom) < 1e-12:
                return max(1e-6, 1e-3 * (np.trace(Kcombo) / n))
            delta = lam_min * (kappa - a) / denom
            if not np.isfinite(delta) or delta <= 0:
                return max(1e-6, 1e-3 * (np.trace(Kcombo) / n))
            return float(delta)

        raise ValueError("delta must be float, 'auto', or 'paper'")

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y=None):
        rng = check_random_state(self.random_state)

        # Accept single view or multiple views
        if isinstance(X, list):
            X_views = [np.asarray(v, dtype=self.dtype) for v in X]
        else:
            X_views = [np.asarray(X, dtype=self.dtype)]

        n = X_views[0].shape[0]
        if any(v.shape[0] != n for v in X_views):
            raise ValueError("All views must have same number of samples.")

        # Optional subsampling to keep kernel feasible
        if self.max_train_size is not None and n > self.max_train_size:
            idx = rng.choice(n, size=self.max_train_size, replace=False)
            X_views = [v[idx] for v in X_views]
            n = self.max_train_size

        self.X_views_ = X_views

        Ks = self._compute_kernels(X_views)
        J = len(Ks)

        # init beta: uniform with unit p-norm (Algorithm 1). :contentReference[oaicite:7]{index=7}
        if J == 1:
            beta = np.array([1.0], dtype=self.dtype)
        else:
            beta = np.ones(J, dtype=self.dtype) * (J ** (-1.0 / self.p))

        ones = np.ones((n,), dtype=self.dtype)

        # alternating optimization (Algorithm 1). :contentReference[oaicite:8]{index=8}
        for it in range(max(1, self.max_iter if J > 1 else 1)):
            Kcombo = np.zeros((n, n), dtype=self.dtype)
            for j in range(J):
                Kcombo += beta[j] * Ks[j]

            delta_val = self._choose_delta(Kcombo)
            A = (Kcombo + (delta_val * np.eye(n, dtype=self.dtype))).astype(np.float64)
            alpha = np.linalg.solve(A, ones.astype(np.float64)).astype(self.dtype)

            if J == 1:
                # no MKL update needed
                self.beta_ = beta
                self.alpha_ = alpha
                self.delta_ = float(delta_val)
                return self

            # u_j = alpha^T K_j alpha (Algorithm 1). :contentReference[oaicite:9]{index=9}
            u = np.array([float(alpha @ (Ks[j] @ alpha)) for j in range(J)], dtype=np.float64)

            # beta update
            if abs(self.p - 1.0) < 1e-12:
                # p=1 -> pick max u (sparse)
                beta_new = np.zeros(J, dtype=np.float64)
                beta_new[int(np.argmax(u))] = 1.0
            else:
                pow_ = 1.0 / (self.p - 1.0)
                v = np.power(np.maximum(u, 0.0), pow_)
                norm_p = np.power(np.sum(np.power(v, self.p)), 1.0 / self.p)
                norm_p = norm_p if norm_p > 0 else 1.0
                beta_new = v / norm_p

            # convergence check
            if np.linalg.norm(beta_new - beta.astype(np.float64)) <= self.tol:
                beta = beta_new.astype(self.dtype)
                self.beta_ = beta
                self.alpha_ = alpha
                self.delta_ = float(delta_val)
                return self

            beta = beta_new.astype(self.dtype)

        # store after max_iter
        self.beta_ = beta
        self.alpha_ = alpha
        self.delta_ = float(delta_val)
        return self

    def transform(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        f(x) = sum_i alpha_i * k(x_i, x) using the learned composite kernel.
        """
        if self.X_views_ is None or self.alpha_ is None or self.beta_ is None or self.gamma_ is None:
            raise RuntimeError("Call fit() first.")

        if isinstance(X, list):
            X_views = [np.asarray(v, dtype=self.dtype) for v in X]
        else:
            X_views = [np.asarray(X, dtype=self.dtype)]

        if len(X_views) != len(self.X_views_):
            raise ValueError("Number of test views must match training views.")

        n_train = self.X_views_[0].shape[0]
        n_test = X_views[0].shape[0]
        J = len(self.X_views_)

        Ktx = np.zeros((n_train, n_test), dtype=self.dtype)
        for j in range(J):
            K_j = rbf_kernel(self.X_views_[j], X_views[j], gamma=self.gamma_[j]).astype(self.dtype)
            Ktx += self.beta_[j] * K_j

        f = (self.alpha_.reshape(1, -1) @ Ktx).reshape(-1)  # (n_test,)
        return f

    def decision_function(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Outlier score: |1 - f(x)|
        """
        f = self.transform(X)
        return np.abs(1.0 - f).astype(self.dtype)

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        # for your pipeline: return anomaly scores
        return self.decision_function(X)
