# nbmf_mm.py
# -*- coding: utf-8 -*-
"""
NBMF-MM: Mean-parameterized Bernoulli NMF solved by Majorization-Minimization.

Implements the NBMF-MM algorithm of Magron & Févotte (2022) with scikit-learn-style API.
Supports two symmetric orientations of the Bernoulli mean factorization X̂ = W H:

  • Aspect Bernoulli (default):   orientation='dir-beta'
      - H columns lie on the simplex (Dirichlet-like constraint), columns sum to 1
      - W ∈ [0,1], with Beta(α,β) prior on W entries

  • Binary ICA:                    orientation='beta-dir'
      - W rows lie on the simplex (Dirichlet-like constraint), rows sum to 1
      - H ∈ [0,1], with Beta(α,β) prior on H entries
      - This matches Algorithm 1 in Magron & Févotte (2022) up to masking.

An optional binary mask allows training for matrix-completion protocols by ignoring
left-out entries in the likelihood and updates.

Scikit-learn-style estimator: BernoulliNMF_MM(BaseEstimator, TransformerMixin)

Notes on the objective:
  The MM monotonicity guarantee in Magron & Févotte (2022) is for the MAP
  objective (negative log-likelihood + negative log-prior on the Beta-
  constrained factor), not the likelihood alone. We therefore track and use
  the full MAP objective in `objective_history_` and for convergence.

Simplex projection:
  Default uses the Duchi et al. (ICML 2008) Euclidean projection ("duchi").
  Users may select the legacy "normalize" method (threshold + L1 renorm),
  which follows the original MM derivation for simplex steps and ensures
  monotone decrease of the MAP objective.

References:
  - P. Magron & C. Févotte (2022), A majorization-minimization algorithm for
    nonnegative binary matrix factorization. IEEE SPL. arXiv:2204.09741.
  - J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra (2008),
    Efficient Projections onto the ℓ₁-Ball for Learning in High Dimensions.
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np

# Optional acceleration for elementwise expressions
try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except Exception:
    _HAS_NUMEXPR = False

# Optional acceleration for simplex projection
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# Optional sparse acceptance
try:
    import scipy.sparse as sp
    _HAS_SPARSE = True
except Exception:
    _HAS_SPARSE = False

# scikit-learn integration (as in UMAP/HDBSCAN)
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils import check_random_state
    _HAS_SKLEARN = True
except Exception:
    class BaseEstimator:  # type: ignore
        pass
    class TransformerMixin:  # type: ignore
        pass
    def check_random_state(seed):  # type: ignore
        return np.random.default_rng(seed)
    _HAS_SKLEARN = False


def _is_sparse(X) -> bool:
    return _HAS_SPARSE and sp.issparse(X)


def _to_dense(X):
    if _is_sparse(X):
        return X.toarray()
    return np.asarray(X)


def _clip01(A, eps: float) -> np.ndarray:
    return np.clip(A, eps, 1.0 - eps, out=A)


def _check_inputs(
    X, mask, alpha, beta, n_components: int, orientation: str, eps: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X = _to_dense(X).astype(np.float64, copy=False)
    if np.any((X < -eps) | (X > 1 + eps)):
        raise ValueError("X must be in [0,1] (binary or probabilities).")

    if mask is not None:
        mask = _to_dense(mask).astype(np.float64, copy=False)
        if mask.shape != X.shape:
            raise ValueError("mask shape must match X.")
        if np.any((mask < -eps) | (mask > 1 + eps)):
            raise ValueError("mask must be binary or in [0,1].")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if np.any(np.asarray(alpha) <= 0) or np.any(np.asarray(beta) <= 0):
        raise ValueError("alpha and beta must be > 0.")
    if orientation not in ("dir-beta", "beta-dir"):
        raise ValueError('orientation must be "dir-beta" or "beta-dir".')
    return X, mask


def _bern_nll_masked(X, P, mask=None, eps=1e-9) -> float:
    P = np.clip(P, eps, 1.0 - eps)
    if mask is None:
        return float(-(X * np.log(P) + (1.0 - X) * np.log(1.0 - P)).sum())
    M = mask
    return float(-((M * X) * np.log(P) + (M * (1.0 - X)) * np.log(1.0 - P)).sum())


def _beta_neglogprior(Z, alpha: float, beta: float, eps: float = 1e-9) -> float:
    """Negative log-prior for independent Beta(alpha, beta) entries of Z."""
    Z = np.clip(Z, eps, 1.0 - eps)
    a = float(alpha)
    b = float(beta)
    # Drop constants: we only need a monotone proxy for convergence & history.
    return float(-((a - 1.0) * np.log(Z) + (b - 1.0) * np.log(1.0 - Z)).sum())


def _safe_div(num, den, eps=1e-12):
    return num / (den + eps)


# -------- Simplex projections -------------------------------------------------
# Duchi-style projection: O(d log d) via sorting, per row/column.

def _project_rows_simplex_numpy_impl(W: np.ndarray) -> np.ndarray:
    M, K = W.shape
    out = np.empty_like(W)
    for m in range(M):
        v = W[m, :]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        j = np.arange(1, K + 1)
        t = (cssv - 1.0) / j
        rho_idx = np.nonzero(u - t > 0)[0]
        rho = rho_idx[-1] if rho_idx.size else 0
        theta = t[rho]
        out[m, :] = np.maximum(v - theta, 0.0)
    return out

def _project_cols_simplex_numpy_impl(H: np.ndarray) -> np.ndarray:
    return _project_rows_simplex_numpy_impl(H.T).T

if _HAS_NUMBA:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _project_simplex_row_numba(v):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = -1
        theta = 0.0
        for j in range(len(u)):
            t = (cssv[j] - 1.0) / (j + 1)
            if u[j] - t > 0:
                rho = j
                theta = t
        w = np.maximum(v - theta, 0.0)
        return w

    @njit(cache=True, fastmath=True)
    def _project_rows_simplex_numba_impl(W):
        M, K = W.shape
        out = np.empty_like(W)
        for m in range(M):
            out[m, :] = _project_simplex_row_numba(W[m, :])
        return out

    def _project_cols_simplex_numba_impl(H):
        return _project_rows_simplex_numba_impl(H.T).T
else:
    _project_rows_simplex_numba_impl = _project_rows_simplex_numpy_impl
    _project_cols_simplex_numba_impl = _project_cols_simplex_numpy_impl


# Legacy “normalize” projection (nonnegative + renormalize)
def _normalize_rows_simplex(W: np.ndarray) -> np.ndarray:
    W = np.maximum(W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return W / s

def _normalize_cols_simplex(H: np.ndarray) -> np.ndarray:
    H = np.maximum(H, 0.0)
    s = H.sum(axis=0, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return H / s


# -------- Initialization ------------------------------------------------------

def _rand_beta(shape, alpha, beta, rng: np.random.Generator, eps: float) -> np.ndarray:
    A = rng.gamma(alpha, 1.0, size=shape)
    B = rng.gamma(beta, 1.0, size=shape)
    Z = A / (A + B + eps)
    return np.clip(Z, eps, 1.0 - eps)


def _init_factors(
    X: np.ndarray,
    K: int,
    orientation: str,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    M, N = X.shape
    if orientation == "dir-beta":
        H = rng.random((K, N))
        H = _project_cols_simplex_numpy_impl(H)
        W = _rand_beta((M, K), alpha, beta, rng, eps)
    else:  # "beta-dir"
        W = rng.random((M, K))
        W = _project_rows_simplex_numpy_impl(W)
        H = _rand_beta((K, N), alpha, beta, rng, eps)
    return W, H


# -------- Estimator -----------------------------------------------------------

class BernoulliNMF_MM(BaseEstimator, TransformerMixin):
    """
    Mean-parameterized Bernoulli (Binary) Matrix Factorization via MM.

    Parameters
    ----------
    n_components : int, default=10
    orientation : {"dir-beta","beta-dir"}, default="dir-beta"
    alpha, beta : float, default=1.2
    max_iter : int, default=2000
    tol : float, default=1e-6
    n_init : int, default=1
    random_state : int|None, default=None
    use_numexpr : bool, default=True
    use_numba : bool, default=True  # kept for backward compat
    projection_method : {"duchi","normalize"}, default="duchi"
        "duchi": Euclidean projection to simplex (Duchi et al., 2008).
        "normalize": legacy behavior (threshold + L1 renorm, MM-faithful).
    projection_backend : {"auto","numba","numpy"}, default="auto"
        Backend for the "duchi" method. "auto": prefer numba if available.
    dtype : {np.float64, np.float32}, default=np.float64
    verbose : int, default=0

    Attributes
    ----------
    components_ : (K, N) ndarray
    W_ : (M, K) ndarray
    n_iter_ : int
    objective_history_ : list[float]
        Full MAP objective values (NLL + negative Beta log-prior) per outer iter.
    """

    def __init__(
        self,
        n_components: int = 10,
        *,
        orientation: str = "dir-beta",
        alpha: float = 1.2,
        beta: float = 1.2,
        max_iter: int = 2000,
        tol: float = 1e-6,
        n_init: int = 1,
        random_state: Optional[int] = None,
        use_numexpr: bool = True,
        use_numba: bool = True,
        projection_method: str = "duchi",
        projection_backend: str = "auto",
        dtype=np.float64,
        verbose: int = 0,
    ):
        self.n_components = n_components
        self.orientation = orientation
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.use_numexpr = use_numexpr
        self.use_numba = use_numba
        self.projection_method = projection_method
        self.projection_backend = projection_backend
        self.dtype = dtype
        self.verbose = verbose

    # ---- projection selector -------------------------------------------------

    def _select_projection_ops(self):
        method = self.projection_method
        backend = self.projection_backend

        if method not in ("duchi", "normalize"):
            raise ValueError('projection_method must be "duchi" or "normalize".')

        if method == "normalize":
            self._proj_rows = _normalize_rows_simplex
            self._proj_cols = _normalize_cols_simplex
            return

        # "duchi"
        if backend == "numba":
            if not _HAS_NUMBA:
                warnings.warn("projection_backend='numba' requested but numba not installed; "
                              "falling back to NumPy.")
                self._proj_rows = _project_rows_simplex_numpy_impl
                self._proj_cols = _project_cols_simplex_numpy_impl
            else:
                self._proj_rows = _project_rows_simplex_numba_impl
                self._proj_cols = _project_cols_simplex_numba_impl
        elif backend == "numpy":
            self._proj_rows = _project_rows_simplex_numpy_impl
            self._proj_cols = _project_cols_simplex_numpy_impl
        elif backend == "auto":
            if _HAS_NUMBA and self.use_numba:
                self._proj_rows = _project_rows_simplex_numba_impl
                self._proj_cols = _project_cols_simplex_numba_impl
            else:
                self._proj_rows = _project_rows_simplex_numpy_impl
                self._proj_cols = _project_cols_simplex_numpy_impl
        else:
            raise ValueError("projection_backend must be 'auto', 'numba', or 'numpy'.")

    # ---- core math helpers ---------------------------------------------------

    def _stats(self, X, V, mask):
        # Compute masked ratios: R1 = (M*X)/V ; R0 = (M*(1-X))/(1-V)
        if self.use_numexpr and _HAS_NUMEXPR:
            M = 1.0 if mask is None else mask
            R1 = ne.evaluate("(M*X)/V")
            R0 = ne.evaluate("(M*(1.0 - X))/(1.0 - V)")
        else:
            M = 1.0 if mask is None else mask
            R1 = _safe_div(M * X, V)
            R0 = _safe_div(M * (1.0 - X), (1.0 - V))
        return R1, R0

    def _update_beta_factor(self, C, D):
        # Closed-form Bernoulli-Beta MM update: Z <- (C + a - 1) / (C + D + a + b - 2)
        a, b = float(self.alpha), float(self.beta)
        Z = (C + (a - 1.0)) / (C + D + (a + b - 2.0) + 1e-12)
        return np.clip(Z, 1e-9, 1.0 - 1e-9)

    def _update_simplex_rows(self, W, num, den):
        W = W * _safe_div(num, den)
        return self._proj_rows(W)

    def _update_simplex_cols(self, H, num, den):
        H = H * _safe_div(num, den)
        return self._proj_cols(H)

    # ---- public API ----------------------------------------------------------

    def _prior_penalty(self, W, H) -> float:
        if self.orientation == "dir-beta":
            return _beta_neglogprior(W, self.alpha, self.beta)
        else:
            return _beta_neglogprior(H, self.alpha, self.beta)

    def fit(self, X, y=None, *, mask=None):
        """
        Fit NBMF-MM to X with optional observation mask.
        Records the full MAP objective (NLL + negative Beta log-prior).
        """
        eps = 1e-9
        X, mask = _check_inputs(
            X, mask, self.alpha, self.beta, self.n_components, self.orientation, eps
        )
        X = X.astype(self.dtype, copy=False)
        mask = None if mask is None else mask.astype(self.dtype, copy=False)
        M, N = X.shape
        K = self.n_components

        self._select_projection_ops()
        rng = check_random_state(self.random_state) if _HAS_SKLEARN else np.random.default_rng(self.random_state)

        best_obj = np.inf
        best = None

        for run in range(max(1, int(self.n_init))):
            W, H = _init_factors(X, K, self.orientation, self.alpha, self.beta, rng, eps)
            obj_hist = []

            prev = np.inf
            for it in range(int(self.max_iter)):
                V = _clip01(W @ H, eps)
                R1, R0 = self._stats(X, V, mask)

                if self.orientation == "beta-dir":
                    # Update H (Beta prior), then W (row-simplex)
                    C = W.T @ R1
                    D = W.T @ R0
                    H = self._update_beta_factor(C, D)

                    num = R1 @ H.T
                    den = R0 @ H.T
                    W = self._update_simplex_rows(W, num, den)

                else:  # "dir-beta"
                    # Update W (Beta prior), then H (col-simplex)
                    C = R1 @ H.T
                    D = R0 @ H.T
                    W = self._update_beta_factor(C, D)

                    num = W.T @ R1
                    den = W.T @ R0
                    H = self._update_simplex_cols(H, num, den)

                V = _clip01(W @ H, eps)
                nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
                obj = nll + self._prior_penalty(W, H)
                obj_hist.append(obj)

                if self.verbose and (it % 50 == 0 or it == self.max_iter - 1):
                    print(f"[run {run+1}] iter {it:5d}  MAP={obj:.6f}  (NLL={nll:.6f})")

                if prev < np.inf:
                    rel = abs(prev - obj) / (prev + 1e-12)
                    if rel < self.tol:
                        break
                prev = obj

            if obj < best_obj:
                best_obj = obj
                best = (W.astype(self.dtype, copy=False), H.astype(self.dtype, copy=False), it + 1, obj_hist)

        self.W_, self.components_, self.n_iter_, self.objective_history_ = best
        return self

    def fit_transform(self, X, y=None, *, mask=None):
        """Fit the model and return W."""
        self.fit(X, mask=mask)
        return self.W_

    def transform(self, X, *, mask=None, max_iter: int = 500, tol: float = 1e-6):
        """
        Estimate W for new X with components_ fixed.
        """
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        eps = 1e-9
        X, mask = _check_inputs(
            X, mask, self.alpha, self.beta, self.n_components, self.orientation, eps
        )
        X = X.astype(self.dtype, copy=False)
        mask = None if mask is None else mask.astype(self.dtype, copy=False)

        H = self.components_
        M, N = X.shape
        K = H.shape[0]

        rng = check_random_state(self.random_state) if _HAS_SKLEARN else np.random.default_rng(self.random_state)
        if self.orientation == "beta-dir":
            W = rng.random((M, K)).astype(self.dtype, copy=False)
            W = self._proj_rows(W)
        else:
            W = _rand_beta((M, K), self.alpha, self.beta, rng, eps).astype(self.dtype, copy=False)

        prev = np.inf
        for _ in range(int(max_iter)):
            V = _clip01(W @ H, eps)
            R1, R0 = self._stats(X, V, mask)

            if self.orientation == "beta-dir":
                num = R1 @ H.T
                den = R0 @ H.T
                W = self._update_simplex_rows(W, num, den)
            else:
                C = R1 @ H.T
                D = R0 @ H.T
                W = self._update_beta_factor(C, D)

            V = _clip01(W @ H, eps)
            nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
            if prev < np.inf:
                rel = abs(prev - nll) / (prev + 1e-12)
                if rel < tol:
                    break
            prev = nll

        return W

    def inverse_transform(self, W):
        """Return reconstructed probabilities Xhat = W @ H in (0,1)."""
        eps = 1e-9
        H = self.components_
        V = W @ H
        return np.clip(V, eps, 1.0 - eps)

    def score(self, X, *, mask=None) -> float:
        """Negative NLL per observed entry (higher is better)."""
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        eps = 1e-9
        X = _to_dense(X).astype(self.dtype, copy=False)
        mask = None if mask is None else _to_dense(mask).astype(self.dtype, copy=False)
        W = self.transform(X, mask=mask, max_iter=1)
        V = self.inverse_transform(W)
        nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
        nobs = float(X.size if mask is None else np.sum(mask))
        return -nll / max(1.0, nobs)

    def perplexity(self, X, *, mask=None) -> float:
        """exp(NLL per observed entry). Lower is better."""
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        eps = 1e-9
        X = _to_dense(X).astype(self.dtype, copy=False)
        mask = None if mask is None else _to_dense(mask).astype(self.dtype, copy=False)
        W = self.transform(X, mask=mask, max_iter=1)
        V = self.inverse_transform(W)
        nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
        nobs = float(X.size if mask is None else np.sum(mask))
        return float(np.exp(nll / max(1.0, nobs)))
