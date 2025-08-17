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

References
----------
- P. Magron & C. Févotte (2022), "A majorization-minimization algorithm for
  nonnegative binary matrix factorization" (Algorithm 1; Eqs. 14-20).  # arXiv:2204.09741
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np

try:
    # Optional acceleration for elementwise expressions
    import numexpr as ne
    _HAS_NUMEXPR = True
except Exception:
    _HAS_NUMEXPR = False

try:
    # Optional acceleration for simplex projection
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    import scipy.sparse as sp
    _HAS_SPARSE = True
except Exception:
    _HAS_SPARSE = False

# scikit-learn integration like UMAP/HDBSCAN
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils import check_random_state
    from sklearn.utils.validation import check_array
    _HAS_SKLEARN = True
except Exception:
    # very small fallback so import still works without sklearn installed
    class BaseEstimator:  # type: ignore
        pass
    class TransformerMixin:  # type: ignore
        pass
    def check_random_state(seed):  # type: ignore
        return np.random.default_rng(seed)
    def check_array(X, **kwargs):  # type: ignore
        return np.asarray(X)
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


def _safe_div(num, den, eps=1e-12):
    return num / (den + eps)


# --- Simplex projections (Duchi et al. 2008) ---------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _project_simplex_row_numba(v):
        """
        Project 1D array v onto probability simplex.
        """
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
    def _project_rows_simplex_numba(W):
        M, K = W.shape
        out = np.empty_like(W)
        for m in range(M):
            out[m, :] = _project_simplex_row_numba(W[m, :])
        return out

else:
    def _project_rows_simplex_numba(W):
        # pure numpy fallback
        M, K = W.shape
        out = np.empty_like(W)
        for m in range(M):
            v = W[m, :]
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            j = np.arange(1, K + 1)
            t = (cssv - 1.0) / j
            # find rho = max { j | u_j - t_j > 0 }
            rho = np.nonzero(u - t > 0)[0]
            rho = rho[-1] if rho.size else 0
            theta = t[rho]
            out[m, :] = np.maximum(v - theta, 0.0)
        return out


def _project_cols_simplex(H):
    # project columns by transposing, using row projection
    return _project_rows_simplex_numba(H.T).T


# --- Initialization -----------------------------------------------------------

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
        # H: columns on simplex; W: Beta(α,β)
        H = rng.random((K, N))
        H = _project_cols_simplex(H)
        W = _rand_beta((M, K), alpha, beta, rng, eps)
    else:  # "beta-dir"
        # W: rows on simplex; H: Beta(α,β)
        W = rng.random((M, K))
        W = _project_rows_simplex_numba(W)
        H = _rand_beta((K, N), alpha, beta, rng, eps)
    return W, H


# --- Estimator ----------------------------------------------------------------

class BernoulliNMF_MM(BaseEstimator, TransformerMixin):
    """
    Mean-parameterized Bernoulli (Binary) Matrix Factorization via MM.

    Parameters
    ----------
    n_components : int
        Rank K.
    orientation : {"dir-beta","beta-dir"}, default="dir-beta"
        Which factor is simplex-constrained vs Beta-constrained.
    alpha, beta : float, default=1.2
        Beta prior hyperparameters for the Beta-constrained factor.
    max_iter : int, default=2000
        Maximum MM iterations.
    tol : float, default=1e-6
        Relative tolerance on NLL for convergence.
    n_init : int, default=1
        Number of restarts; best (lowest NLL) is kept.
    random_state : int|None, default=None
        RNG seed.
    use_numexpr : bool, default=True
        Use NumExpr for elementwise expressions, if available.
    use_numba : bool, default=True
        Use Numba-accelerated simplex projection, if available.
    dtype : {np.float64, np.float32}, default=np.float64
        Computation dtype.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    components_ : ndarray of shape (K, N)
        Learned `H` (basis).
    W_ : ndarray of shape (M, K)
        Learned `W` (loadings).
    n_iter_ : int
        Iterations run for the best initialization.
    objective_history_ : list[float]
        NLL per iteration for the best initialization.
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
        self.dtype = dtype
        self.verbose = verbose

    # --- core math helpers (vectorized) --------------------------------------

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
        if self.use_numba and _HAS_NUMBA:
            return _project_rows_simplex_numba(W)
        return _project_rows_simplex_numba(W)  # numpy fallback is same name

    def _update_simplex_cols(self, H, num, den):
        H = H * _safe_div(num, den)
        return _project_cols_simplex(H)

    # --- public API -----------------------------------------------------------

    def fit(self, X, y=None, *, mask=None):
        """
        Fit NBMF-MM to X with optional observation mask.

        Parameters
        ----------
        X : array-like or sparse, shape (n_samples, n_features), values in [0,1]
        mask : array-like or sparse, optional, same shape as X, values in [0,1]

        Returns
        -------
        self
        """
        eps = 1e-9
        X, mask = _check_inputs(
            X, mask, self.alpha, self.beta, self.n_components, self.orientation, eps
        )
        X = X.astype(self.dtype, copy=False)
        mask = None if mask is None else mask.astype(self.dtype, copy=False)
        M, N = X.shape
        K = self.n_components

        rng = check_random_state(self.random_state) if _HAS_SKLEARN else np.random.default_rng(self.random_state)

        best_nll = np.inf
        best = None

        for run in range(max(1, int(self.n_init))):
            W, H = _init_factors(X, K, self.orientation, self.alpha, self.beta, rng, eps)
            obj_hist = []

            prev = np.inf
            for it in range(int(self.max_iter)):
                V = _clip01(W @ H, eps)

                R1, R0 = self._stats(X, V, mask)

                if self.orientation == "beta-dir":
                    # Update H (Beta prior)
                    C = W.T @ R1
                    D = W.T @ R0
                    H = self._update_beta_factor(C, D)

                    # Update W (row-simplex)
                    num = R1 @ H.T
                    den = R0 @ H.T
                    W = self._update_simplex_rows(W, num, den)

                else:  # "dir-beta"
                    # Update W (Beta prior)
                    C = R1 @ H.T
                    D = R0 @ H.T
                    W = self._update_beta_factor(C, D)

                    # Update H (col-simplex)
                    num = W.T @ R1
                    den = W.T @ R0
                    H = self._update_simplex_cols(H, num, den)

                V = _clip01(W @ H, eps)
                nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
                obj_hist.append(nll)

                if self.verbose and (it % 50 == 0 or it == self.max_iter - 1):
                    print(f"[run {run+1}] iter {it:5d}  NLL={nll:.6f}")

                if prev < np.inf:
                    rel = abs(prev - nll) / (prev + 1e-12)
                    if rel < self.tol:
                        break
                prev = nll

            if nll < best_nll:
                best_nll = nll
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

        For orientation="beta-dir" (row-simplex W; Beta prior on H),
        we update W multiplicatively with simplex projection.

        For orientation="dir-beta" (Beta prior on W),
        we use the closed-form Beta update for W.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        mask : array-like, optional
        max_iter : int
        tol : float

        Returns
        -------
        W : ndarray, shape (n_samples, n_components)
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
        # Good starting point
        if self.orientation == "beta-dir":
            W = rng.random((M, K)).astype(self.dtype, copy=False)
            W = _project_rows_simplex_numba(W)
        else:
            W = _rand_beta((M, K), self.alpha, self.beta, rng, eps).astype(self.dtype, copy=False)

        prev = np.inf
        for it in range(int(max_iter)):
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

        V = self.inverse_transform(self.transform(X, mask=mask, max_iter=1))  # single step approx
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

        V = self.inverse_transform(self.transform(X, mask=mask, max_iter=1))  # single step approx
        nll = _bern_nll_masked(X, V, mask=mask, eps=eps)
        nobs = float(X.size if mask is None else np.sum(mask))
        return float(np.exp(nll / max(1.0, nobs)))