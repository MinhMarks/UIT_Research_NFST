#!/usr/bin/env python3
from __future__ import annotations

"""
NODE FINAL

Final deployment-oriented NODE implementation.

WHERE:
    D = learned-diagonal KNN distance
    R = D / local nominal scale
    WHERE = Gaussian-copula joint upper-tail score of (D, R)

WHAT:
    A = absolute grouped masked-conditional inconsistency
    Q = local-relative grouped masked-conditional inconsistency
    WHAT = positive correlated quadratic score of (A, Q)

FINAL:
    Gaussian-copula joint upper-tail score of (WHERE, WHAT)

Fitting uses nominal data only.
Python 3.10 compatible.
"""

from dataclasses import dataclass
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from pyod.models.lunar import generate_negative_samples

from model import node_v2 as _base
from model.node_v2_diag_where_lowrank_what import _DiagWhereLowRankAuxGeometry

EPS = 1e-12
P_MIN = 1e-15
RHO_CLIP = 0.995


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_fast_matmul(device: str) -> None:
    """Safe-ish CUDA throughput knobs; no algorithmic change."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


class _EmpiricalCDF:
    def fit(self, x):
        x = np.asarray(x, np.float64).reshape(-1)
        if len(x) < 10 or not np.isfinite(x).all():
            raise ValueError("Invalid nominal calibration scores.")
        self.ref_ = np.sort(x)
        self.n_ = int(len(x))
        self.lo_ = 0.5 / self.n_
        return self

    def transform(self, x):
        x = np.asarray(x, np.float64).reshape(-1)
        lo = np.searchsorted(self.ref_, x, side="left")
        hi = np.searchsorted(self.ref_, x, side="right")
        q = (lo.astype(np.float64) + hi.astype(np.float64)) / (2.0 * self.n_)
        return np.clip(q, self.lo_, 1.0 - self.lo_)


def _gaussianize(q, n):
    lo = 0.5 / float(n)
    return norm.ppf(np.clip(np.asarray(q, np.float64), lo, 1.0 - lo))


def _rho(z1, z2):
    z1 = np.asarray(z1, np.float64).reshape(-1)
    z2 = np.asarray(z2, np.float64).reshape(-1)
    if len(z1) < 3 or np.std(z1) <= 0 or np.std(z2) <= 0:
        return 0.0
    r = float(np.corrcoef(z1, z2)[0, 1])
    if not np.isfinite(r):
        r = 0.0
    return float(np.clip(r, -RHO_CLIP, RHO_CLIP))


def _bv_cdf(z1, z2, rho):
    pts = np.column_stack([
        np.asarray(z1, np.float64).reshape(-1),
        np.asarray(z2, np.float64).reshape(-1),
    ])
    cov = np.array([[1.0, rho], [rho, 1.0]], np.float64)
    out = multivariate_normal.cdf(
        pts, mean=np.zeros(2, np.float64), cov=cov, allow_singular=False
    )
    return np.asarray(out, np.float64).reshape(-1)


def _joint_and(z1, z2, rho):
    p = 1.0 - norm.cdf(z1) - norm.cdf(z2) + _bv_cdf(z1, z2, rho)
    return -np.log(np.clip(p, P_MIN, 1.0))


def _pos_quad(z1, z2, rho):
    z = np.column_stack([
        np.maximum(np.asarray(z1, np.float64), 0.0),
        np.maximum(np.asarray(z2, np.float64), 0.0),
    ])
    cov = np.array([[1.0, rho], [rho, 1.0]], np.float64)
    inv = np.linalg.inv(cov)
    return np.einsum("bi,ij,bj->b", z, inv, z)


class _PairCalibrator:
    def __init__(self, mode: str):
        self.mode = str(mode)

    def fit(self, a, b):
        self.a_cdf_ = _EmpiricalCDF().fit(a)
        self.b_cdf_ = _EmpiricalCDF().fit(b)
        qa = self.a_cdf_.transform(a)
        qb = self.b_cdf_.transform(b)
        za = _gaussianize(qa, self.a_cdf_.n_)
        zb = _gaussianize(qb, self.b_cdf_.n_)
        self.rho_ = _rho(za, zb)
        return self

    def score(self, a, b):
        qa = self.a_cdf_.transform(a)
        qb = self.b_cdf_.transform(b)
        za = _gaussianize(qa, self.a_cdf_.n_)
        zb = _gaussianize(qb, self.b_cdf_.n_)
        if self.mode == "and":
            return _joint_and(za, zb, self.rho_)
        if self.mode == "pos_quad":
            return _pos_quad(za, zb, self.rho_)
        raise ValueError("Unknown pair mode: " + self.mode)


class _RelativeDensityView:
    def __init__(self, geometry):
        self.geometry = geometry
        self.k = int(geometry.k)
        ref_dist = geometry.nn_.kneighbors(
            X=None, n_neighbors=self.k, return_distance=True
        )[0]
        self.ref_scale_ = np.maximum(
            np.asarray(ref_dist, np.float64).mean(axis=1), EPS
        )

    def transform(self, X):
        X = _base._as_float32(X)
        D = np.asarray(self.geometry.decision_function(X), np.float64).reshape(-1)
        Xs = self.geometry.scaler_.transform(X).astype(np.float32)
        Xt = self.geometry._transform(Xs, self.geometry.weights_)
        _, idx = self.geometry.nn_.kneighbors(
            Xt, n_neighbors=self.k, return_distance=True
        )
        L = np.maximum(self.ref_scale_[np.asarray(idx, np.int64)].mean(axis=1), EPS)
        return D, L, D / L


def _partial_dep(X, seed, fraction):
    X = _base._as_float32(X)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    Y = X.copy()
    fraction = float(np.clip(fraction, 0.0, 1.0))
    for j in range(d):
        m = min(n, max(2, int(round(fraction * n))))
        ids = rng.choice(n, size=m, replace=False)
        src = ids[rng.permutation(m)]
        Y[ids, j] = X[src, j]
    diff = float(np.max(np.abs(np.sort(X, axis=0) - np.sort(Y, axis=0))))
    if diff != 0.0:
        raise RuntimeError("Dependency generator failed marginal preservation.")
    return Y.astype(np.float32)


def _subsample_rows(X, fraction: float, seed: int, min_rows: int = 64):
    X = _base._as_float32(X)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction >= 1.0 or len(X) <= min_rows:
        return X
    n = min(len(X), max(min_rows, int(round(fraction * len(X)))))
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(X), size=n, replace=False)
    return X[np.sort(ids)]


class _GroupedConditionalRepresentation:
    """
    Group-masked conditional representation.

    For a group M, neighbors are searched using all features except M. The same
    KNN search predicts every j in M. With P complementary partitions, every
    feature is scored P times and the per-feature E/Q values are averaged.

    The group masking is exact for the DIAG + LOWRANK metric being used; only
    the choice of masking multiple dimensions together (instead of one at a
    time) changes the representation.
    """

    def __init__(
        self,
        geometry,
        k,
        device,
        rep_batch,
        local_floor,
        ratio_clip,
        reference_cap,
        mask_ratio,
        mask_partitions,
        seed,
    ):
        self.geometry = geometry
        self.k = int(k)
        self.device = str(device)
        self.rep_batch = int(rep_batch)
        self.local_floor = float(local_floor)
        self.ratio_clip = float(ratio_clip)
        self.reference_cap = int(reference_cap) if reference_cap else 0
        self.mask_ratio = float(mask_ratio)
        self.mask_partitions = max(1, int(mask_partitions))
        self.seed = int(seed)

        Rall = np.asarray(self.geometry.Xref_, np.float32)
        nref, d = Rall.shape
        if self.reference_cap > 0 and nref > self.reference_cap:
            rng = np.random.default_rng(self.seed + 9001)
            ref_ids = np.sort(rng.choice(nref, self.reference_cap, replace=False))
        else:
            ref_ids = np.arange(nref, dtype=np.int64)

        self.ref_ids_ = np.asarray(ref_ids, np.int64)
        Rn = Rall[self.ref_ids_]
        w = np.asarray(self.geometry.weights_, np.float32)
        U_np = np.asarray(self.geometry.U_, np.float32)

        self.n_features_ = int(d)
        self.mask_size_ = min(
            d,
            max(1, int(math.ceil(self.mask_ratio * d))),
        )
        self.groups_ = self._build_groups(d)
        self.n_masks_ = int(len(self.groups_))

        # Cache invariant reference tensors once on the target device.
        self.R = torch.as_tensor(Rn, dtype=torch.float32, device=self.device)
        self.W = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        self.U = torch.as_tensor(U_np, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            self.rn_diag = torch.sum(self.R.square() * self.W[None], dim=1)
            self.LR = self.R @ self.U

            # Reference-side masked quantities are invariant across every query
            # batch/call, so cache them once. This also avoids materializing a
            # B x Nref x |M| delta tensor during scoring.
            self.group_cache_ = []
            for g_np in self.groups_:
                g = torch.as_tensor(g_np, dtype=torch.long, device=self.device)
                Rg = self.R[:, g]
                Wg = self.W[g]
                Ug = self.U[g]
                r_removed_diag = torch.sum(Rg.square() * Wg[None], dim=1)
                LRm = self.LR - (Rg @ Ug)
                lrm_n = torch.sum(LRm.square(), dim=1)
                self.group_cache_.append((g, Rg, Wg, Ug, r_removed_diag, LRm, lrm_n))

    def _build_groups(self, d: int):
        groups = []
        # Each partition is a complete permutation/chunking, so every feature is
        # covered exactly once per partition.
        for p in range(self.mask_partitions):
            rng = np.random.default_rng(self.seed + 9101 + 104729 * p)
            perm = rng.permutation(d)
            for st in range(0, d, self.mask_size_):
                g = np.sort(perm[st:st + self.mask_size_]).astype(np.int64)
                groups.append(g)
        return groups

    def transform(self, X):
        X = _base._as_float32(X)
        Qn = self.geometry.scaler_.transform(X).astype(np.float32)
        n, d = Qn.shape
        if d != self.n_features_:
            raise ValueError("Conditional representation feature count mismatch.")

        Eout = np.empty((n, d), np.float32)
        Qout = np.empty((n, d), np.float32)

        R = self.R
        W = self.W
        U = self.U
        rn_diag = self.rn_diag

        with torch.inference_mode():
            for st in range(0, n, self.rep_batch):
                ed = min(n, st + self.rep_batch)
                Xq = torch.as_tensor(Qn[st:ed], dtype=torch.float32, device=self.device)
                b = Xq.shape[0]

                qn_diag = torch.sum(Xq.square() * W[None], dim=1)
                Ddiag = torch.clamp(
                    qn_diag[:, None] + rn_diag[None]
                    - 2.0 * ((Xq * W[None]) @ R.T),
                    min=0.0,
                )

                LQ = Xq @ U

                Eacc = torch.zeros((b, d), dtype=torch.float32, device=self.device)
                Qacc = torch.zeros((b, d), dtype=torch.float32, device=self.device)
                Cacc = torch.zeros((d,), dtype=torch.float32, device=self.device)

                for g, Rg, Wg, Ug, r_removed_diag, LRm, lrm_n in self.group_cache_:
                    # Remove the whole group from the diagonal metric using
                    # ||q-r||_W^2 = ||q||_W^2 + ||r||_W^2 - 2<qW,r>.
                    # This is algebraically identical to summing per-feature
                    # squared deltas but avoids a B x Nref x |M| tensor.
                    Xg = Xq[:, g]  # B x G
                    q_removed_diag = torch.sum(Xg.square() * Wg[None], dim=1)
                    removed_diag = (
                        q_removed_diag[:, None]
                        + r_removed_diag[None]
                        - 2.0 * ((Xg * Wg[None]) @ Rg.T)
                    )
                    masked_diag = torch.clamp(Ddiag - removed_diag, min=0.0)

                    # Remove the group from the low-rank transform. Reference
                    # side (LRm/lrm_n) was cached once in __init__.
                    LQm = LQ - (Xg @ Ug)  # B x r
                    lqm_n = torch.sum(LQm.square(), dim=1)
                    masked_low = torch.clamp(
                        lqm_n[:, None] + lrm_n[None] - 2.0 * (LQm @ LRm.T),
                        min=0.0,
                    )

                    Dm = torch.clamp(masked_diag + masked_low, min=0.0)
                    vals, idx = torch.topk(
                        Dm, k=self.k, dim=1, largest=False, sorted=False
                    )
                    dist = torch.sqrt(vals + EPS)
                    zero = dist <= 1e-8
                    has_zero = zero.any(1, keepdim=True)
                    iw = torch.where(
                        has_zero,
                        zero.float(),
                        1.0 / torch.clamp(dist, min=1e-8),
                    )
                    iw = iw / torch.clamp(iw.sum(1, keepdim=True), min=EPS)

                    # One neighbor search predicts every masked feature.
                    # Rg[idx]: B x k x G
                    targets = Rg[idx]
                    pred = torch.sum(targets * iw[:, :, None], dim=1)  # B x G
                    err = torch.abs(Xg - pred)
                    spread = torch.sum(
                        iw[:, :, None] * torch.abs(targets - pred[:, None, :]),
                        dim=1,
                    )
                    ratio = torch.clamp(
                        err / (spread + self.local_floor),
                        min=0.0,
                        max=self.ratio_clip,
                    )

                    Eacc[:, g] += err
                    Qacc[:, g] += torch.log1p(ratio)
                    Cacc[g] += 1.0

                Cacc = torch.clamp(Cacc, min=1.0)
                Eout[st:ed] = (Eacc / Cacc[None]).cpu().numpy()
                Qout[st:ed] = (Qacc / Cacc[None]).cpu().numpy()

        return Eout, Qout


class _DependencyExpert:
    """Same old absolute expert objective, but validate only every eval_every epochs."""

    def __init__(self, seed, device, batch_size, epochs, lr, patience, eval_every):
        self.seed = int(seed)
        self.device = device
        self.batch_size = int(batch_size)
        self.epochs_limit = int(epochs)
        self.lr = float(lr)
        self.patience = int(patience)
        self.eval_every = max(1, int(eval_every))

    @staticmethod
    def _auc_parts(kind, scores):
        normal = scores[kind == 0]
        out = {}
        for code, name in [(1, "MIXED"), (2, "DEP")]:
            anom = scores[kind == code]
            y = np.r_[np.zeros(len(normal), np.int8), np.ones(len(anom), np.int8)]
            out[name] = float(roc_auc_score(y, np.r_[normal, anom]))
        return out

    def _eval(self, model, B, kv):
        model.eval()
        out = []
        with torch.inference_mode():
            for st in range(0, len(B), 4096):
                s, _ = model(torch.from_numpy(B[st:st + 4096]).to(self.device))
                out.append(s.cpu().numpy())
        return self._auc_parts(kv, np.concatenate(out))

    def fit(self, Rn, Rmix, Rdep, Rnv, Rmixv, Rdepv):
        Rtr = np.vstack([Rn, Rmix, Rdep])
        ytr = np.r_[
            np.zeros(len(Rn), np.float32),
            np.ones(len(Rmix) + len(Rdep), np.float32),
        ]
        Rv = np.vstack([Rnv, Rmixv, Rdepv])
        kv = np.r_[
            np.zeros(len(Rnv), np.int8),
            np.ones(len(Rmixv), np.int8),
            np.full(len(Rdepv), 2, np.int8),
        ]

        self.scaler_ = StandardScaler().fit(Rn)
        A = self.scaler_.transform(Rtr).astype(np.float32)
        B = self.scaler_.transform(Rv).astype(np.float32)

        _seed_all(self.seed + 11)
        model = _base._DependencyNetwork(A.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed + 401)

        n0 = float(len(Rn))
        n1 = float(len(Rmix) + len(Rdep))
        pos_weight = torch.tensor(
            n0 / max(n1, 1.0), dtype=torch.float32, device=self.device
        )

        best = -np.inf
        best_state = None
        bad = 0

        for epoch in range(self.epochs_limit):
            model.train()
            perm = rng.permutation(len(A))
            for st in range(0, len(A), self.batch_size):
                ids = perm[st:st + self.batch_size]
                x = torch.from_numpy(A[ids]).to(self.device)
                y = torch.from_numpy(ytr[ids]).to(self.device)
                logit, _ = model(x)
                loss = F.binary_cross_entropy_with_logits(
                    logit, y, pos_weight=pos_weight
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            do_eval = ((epoch + 1) % self.eval_every == 0) or (epoch + 1 == self.epochs_limit)
            if not do_eval:
                continue
            parts = self._eval(model, B, kv)
            metric = 0.5 * (parts["MIXED"] + parts["DEP"])
            if metric > best + 1e-5:
                best = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is None:
            # Should only happen if epochs < eval_every; force one evaluation/state.
            parts = self._eval(model, B, kv)
            best = 0.5 * (parts["MIXED"] + parts["DEP"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model_ = model
        return self

    def score(self, R):
        Z = self.scaler_.transform(R).astype(np.float32)
        scores = []
        self.model_.eval()
        with torch.inference_mode():
            for st in range(0, len(Z), 4096):
                score, _ = self.model_(torch.from_numpy(Z[st:st + 4096]).to(self.device))
                scores.append(score.cpu().numpy())
        return np.concatenate(scores)


class _RankDependencyExpert:
    def __init__(self, seed, device, batch, epochs, lr, patience, temperature, eval_every):
        self.seed = int(seed)
        self.device = device
        self.batch = int(batch)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.patience = int(patience)
        self.temperature = float(temperature)
        self.eval_every = max(1, int(eval_every))

    @staticmethod
    def _auc(normal, anomaly):
        y = np.concatenate([
            np.zeros(len(normal), np.int8),
            np.ones(len(anomaly), np.int8),
        ])
        return float(roc_auc_score(y, np.concatenate([normal, anomaly])))

    def _score_scaled(self, model, Z):
        out = []
        model.eval()
        with torch.inference_mode():
            for st in range(0, len(Z), 4096):
                s, _ = model(torch.from_numpy(Z[st:st + 4096]).to(self.device))
                out.append(s.cpu().numpy())
        return np.concatenate(out)

    def _evaluate(self, model, Nv, Av):
        snv = self._score_scaled(model, Nv)
        return {
            name: self._auc(snv, self._score_scaled(model, Z))
            for name, Z in Av.items()
        }

    def fit(self, normal, train_regimes: dict[str, np.ndarray], normal_val,
            val_regimes: dict[str, np.ndarray]):
        self.scaler_ = StandardScaler().fit(normal)
        N = self.scaler_.transform(normal).astype(np.float32)
        Nv = self.scaler_.transform(normal_val).astype(np.float32)
        A = np.vstack([
            self.scaler_.transform(x).astype(np.float32)
            for x in train_regimes.values()
        ])
        Av = {
            k: self.scaler_.transform(v).astype(np.float32)
            for k, v in val_regimes.items()
        }

        _seed_all(self.seed + 51011)
        model = _base._DependencyNetwork(N.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed + 51021)

        best = -np.inf
        best_state = None
        bad = 0

        for epoch in range(self.epochs):
            model.train()
            order = rng.permutation(len(A))
            normal_ids = rng.integers(0, len(N), size=len(A))

            for st in range(0, len(order), self.batch):
                ids = order[st:st + self.batch]
                nids = normal_ids[ids]
                xa = torch.from_numpy(A[ids]).to(self.device)
                xn = torch.from_numpy(N[nids]).to(self.device)
                sa, _ = model(xa)
                sn, _ = model(xn)
                loss = F.softplus(
                    -(sa - sn) / max(self.temperature, 1e-6)
                ).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            do_eval = ((epoch + 1) % self.eval_every == 0) or (epoch + 1 == self.epochs)
            if not do_eval:
                continue
            parts = self._evaluate(model, Nv, Av)
            metric = float(np.mean(list(parts.values())))

            if metric > best + 1e-5:
                best = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is None:
            parts = self._evaluate(model, Nv, Av)
            best = float(np.mean(list(parts.values())))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model_ = model
        return self

    def score(self, X):
        Z = self.scaler_.transform(X).astype(np.float32)
        return self._score_scaled(self.model_, Z)


@dataclass
class NODEFinalConfig:
    # Core NODE geometry -- intentionally unchanged from the final model.
    k: int = 5
    random_state: int = 42
    val_size: float = 0.10
    negative_sampling: str = "MIXED"
    negative_proportion: float = 1.0
    negative_epsilon: float = 0.1
    outer_rounds: int = 5
    epochs_per_round: int = 12
    metric_batch: int = 1024
    metric_lr: float = 0.03
    weight_kl: float = 1e-3
    grad_clip: float = 5.0

    # Conditional representation.
    rep_batch: int = 1024
    conditional_reference_cap: int = 4096
    mask_ratio: float = 0.20
    mask_partitions: int = 2

    # Extra Q-only regimes: keep FULL at 100%; SPARSE/BLOCK use this fraction.
    q_aux_fraction: float = 0.25

    # Dependency experts.
    expert_batch: int = 512
    expert_epochs: int = 120
    expert_lr: float = 1e-3
    expert_patience: int = 8       # number of validation checks, not raw epochs
    expert_eval_every: int = 3

    local_floor: float = 1e-3
    ratio_clip: float = 100.0
    rank_temperature: float = 1.0
    component_cal_fraction: float = 0.40
    device: Optional[str] = None


class NODEFinal:
    """
    NODE:
        WHERE = D/R Gaussian-copula AND
        WHAT  = grouped-mask A/Q positive correlated quadratic
        FINAL = WHERE/WHAT Gaussian-copula AND
    """

    def __init__(self, config=None):
        self.config = config or NODEFinalConfig()
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _enable_fast_matmul(self.device)
        self._is_fitted = False

    def _mixed(self, X, seed):
        X = _base._as_float32(X)
        Xs = self.geometry_.scaler_.transform(X).astype(np.float32)
        neg, _ = generate_negative_samples(
            Xs,
            self.config.negative_sampling,
            self.config.negative_proportion,
            self.config.negative_epsilon,
            random_state=check_random_state(seed),
        )
        return self.geometry_.scaler_.inverse_transform(
            np.asarray(neg, np.float32)
        ).astype(np.float32)

    def _raw_where(self, X):
        D, L, R = self.where_view_.transform(X)
        W = self.where_pair_.score(D, R)
        return W, D, L, R

    def _raw_what(self, X):
        E, Q = self.cond_rep_.transform(X)
        A = self.abs_expert_.score(E)
        B = self.q_expert_.score(Q)
        return self.what_pair_.score(A, B)

    def fit(self, X_train, X_validation, X_calibration):
        c = self.config
        seed = int(c.random_state)
        X_train = _base._as_float32(X_train)
        X_validation = _base._as_float32(X_validation)
        X_calibration = _base._as_float32(X_calibration)

        if not (
            X_train.shape[1] == X_validation.shape[1] == X_calibration.shape[1]
        ):
            raise ValueError("Train/validation/calibration feature counts differ.")
        if min(len(X_train), len(X_validation), len(X_calibration)) < 20:
            raise ValueError("Each nominal split must contain at least 20 samples.")

        _seed_all(seed)
        self.n_features_in_ = int(X_train.shape[1])

        # 1) SAME learned geometry as FINAL NODE.
        self.geometry_ = _DiagWhereLowRankAuxGeometry(
            k=c.k,
            seed=seed,
            device=self.device,
            val_size=c.val_size,
            negative_sampling=c.negative_sampling,
            negative_proportion=c.negative_proportion,
            negative_epsilon=c.negative_epsilon,
            outer_rounds=c.outer_rounds,
            epochs_per_round=c.epochs_per_round,
            metric_batch=c.metric_batch,
            metric_lr=c.metric_lr,
            weight_kl=c.weight_kl,
            grad_clip=c.grad_clip,
        ).fit(X_train)

        self.where_view_ = _RelativeDensityView(self.geometry_)
        self.cond_rep_ = _GroupedConditionalRepresentation(
            geometry=self.geometry_,
            k=c.k,
            device=self.device,
            rep_batch=c.rep_batch,
            local_floor=c.local_floor,
            ratio_clip=c.ratio_clip,
            reference_cap=c.conditional_reference_cap,
            mask_ratio=c.mask_ratio,
            mask_partitions=c.mask_partitions,
            seed=seed,
        )

        # 2) WHAT development. Keep NORMAL/MIXED/FULL complete. Only the extra
        # SPARSE/BLOCK Q regimes are subsampled.
        Xetr, Xeval = train_test_split(
            X_validation, test_size=0.35, random_state=seed
        )
        mixed_tr = self._mixed(Xetr, seed + 101)
        mixed_va = self._mixed(Xeval, seed + 102)
        full_tr = _base.exact_dependency_negative(Xetr, seed + 111)[0]
        full_va = _base.exact_dependency_negative(Xeval, seed + 112)[0]

        sparse_frac = 1.0 / max(self.n_features_in_, 1)
        Xs_tr = _subsample_rows(Xetr, c.q_aux_fraction, seed + 601)
        Xs_va = _subsample_rows(Xeval, c.q_aux_fraction, seed + 602)
        Xb_tr = _subsample_rows(Xetr, c.q_aux_fraction, seed + 603)
        Xb_va = _subsample_rows(Xeval, c.q_aux_fraction, seed + 604)

        sparse_tr = _partial_dep(Xs_tr, seed + 611, sparse_frac)
        sparse_va = _partial_dep(Xs_va, seed + 612, sparse_frac)
        block_tr = _partial_dep(Xb_tr, seed + 621, 0.25)
        block_va = _partial_dep(Xb_va, seed + 622, 0.25)

        Etr, Qtr = self.cond_rep_.transform(Xetr)
        Eva, Qva = self.cond_rep_.transform(Xeval)
        Emix_tr, _ = self.cond_rep_.transform(mixed_tr)
        Emix_va, _ = self.cond_rep_.transform(mixed_va)
        Efull_tr, Qfull_tr = self.cond_rep_.transform(full_tr)
        Efull_va, Qfull_va = self.cond_rep_.transform(full_va)
        _, Qsparse_tr = self.cond_rep_.transform(sparse_tr)
        _, Qsparse_va = self.cond_rep_.transform(sparse_va)
        _, Qblock_tr = self.cond_rep_.transform(block_tr)
        _, Qblock_va = self.cond_rep_.transform(block_va)

        # 3) Absolute conditional WHAT: same BCE objective, fewer validation passes.
        self.abs_expert_ = _DependencyExpert(
            seed=seed,
            device=self.device,
            batch_size=c.expert_batch,
            epochs=c.expert_epochs,
            lr=c.expert_lr,
            patience=c.expert_patience,
            eval_every=c.expert_eval_every,
        ).fit(
            Etr, Emix_tr, Efull_tr,
            Eva, Emix_va, Efull_va,
        )

        # 4) Relative conditional WHAT: same rank loss; FULL stays full size.
        self.q_expert_ = _RankDependencyExpert(
            seed=seed,
            device=self.device,
            batch=c.expert_batch,
            epochs=c.expert_epochs,
            lr=c.expert_lr,
            patience=c.expert_patience,
            temperature=c.rank_temperature,
            eval_every=c.expert_eval_every,
        ).fit(
            Qtr,
            {"SPARSE": Qsparse_tr, "BLOCK": Qblock_tr, "FULL": Qfull_tr},
            Qva,
            {"SPARSE": Qsparse_va, "BLOCK": Qblock_va, "FULL": Qfull_va},
        )

        # 5) SAME independent component/final calibration logic.
        Xcomp, Xfinal = train_test_split(
            X_calibration,
            test_size=1.0 - c.component_cal_fraction,
            random_state=seed + 7,
            shuffle=True,
        )

        Dcal, _, Rcal = self.where_view_.transform(Xcomp)
        self.where_pair_ = _PairCalibrator("and").fit(Dcal, Rcal)

        Ecal, Qcal = self.cond_rep_.transform(Xcomp)
        Acal = self.abs_expert_.score(Ecal)
        Bcal = self.q_expert_.score(Qcal)
        self.what_pair_ = _PairCalibrator("pos_quad").fit(Acal, Bcal)

        Wfinal = self._raw_where(Xfinal)[0]
        Tfinal = self._raw_what(Xfinal)
        self.final_pair_ = _PairCalibrator("and").fit(Wfinal, Tfinal)


        self._is_fitted = True
        return self

    def _check(self, X):
        if not self._is_fitted:
            raise RuntimeError("NODEFinal is not fitted.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count mismatch.")

    def decision_function(self, X):
        X = _base._as_float32(X)
        self._check(X)
        where = self._raw_where(X)[0]
        what = self._raw_what(X)
        return np.asarray(self.final_pair_.score(where, what), np.float64)


__all__ = ["NODEFinal", "NODEFinalConfig"]
