# SPDX-License-Identifier: BSD-3-Clause
"""
NBMF-MM estimator: Bernoulli (mean-parameterized) NMF solved by MM.

Implements Algorithm 1 of Magron & Févotte (2022) with a scikit-learn-style API
and two symmetric orientations:

- "beta-dir"  (Binary ICA):           W rows on the simplex; Beta prior on H
- "dir-beta"  (Aspect Bernoulli):     H columns on the simplex; Beta prior on W

Projection for the simplex-constrained factor:
- projection_method="normalize" (**default; theory-first**): multiplicative
  step followed by exact L1 renormalization (paper-faithful; monotone).
- projection_method="duchi": routed to the same paper-exact step for parity
  (the MM step already lands on the simplex).

References
----------
- P. Magron and C. Févotte (2022).
  “A majorization-minimization algorithm for nonnegative binary matrix
   factorization.” IEEE SPL. (arXiv:2204.09741)
- J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra (2008).
  “Efficient Projections onto the ℓ₁-Ball for Learning in High Dimensions.”
"""

# src/nbmf_mm/estimator.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None

from ._mm_exact import (
    bernoulli_nll,
    objective,
    mm_step_beta_dir,
    mm_step_dir_beta,
    _clip01,
)

Array = np.ndarray


def _is_sparse(x) -> bool:
    return sp is not None and sp.issparse(x)


def _to_ndarray(x) -> Array:
    if _is_sparse(x):
        return x.toarray()
    return np.asarray(x, dtype=float)


def _project_rows_to_simplex(W: Array, eps: float = 1e-12) -> Array:
    W = np.maximum(W, eps)
    W /= np.sum(W, axis=1, keepdims=True)
    return W


def _project_cols_to_simplex(H: Array, eps: float = 1e-12) -> Array:
    H = np.maximum(H, eps)
    H /= np.sum(H, axis=0, keepdims=True)
    return H


def _canonical_orientation(name: str) -> str:
    s = str(name).strip().lower().replace(" ", "").replace("_", "").replace("‑", "").replace("–", "")
    # Accept a few friendly aliases
    if s in {"dirbeta", "aspectbernoulli", "dir-beta"}:
        return "dir-beta"
    if s in {"betadir", "binaryica", "bica", "beta-dir"}:
        return "beta-dir"
    # If already canonical, return as is
    if name in {"dir-beta", "beta-dir"}:
        return name
    # Default to dir-beta for robustness
    return "dir-beta"


class NBMF:
    """
    Mean-parameterized Bernoulli NMF solved by Majorization–Minimization (MM).

    This estimator implements the two symmetric orientations:

    - ``orientation='dir-beta'`` (default; "Aspect Bernoulli"):
      * Columns of ``H`` lie on the probability simplex (Dirichlet-like constraint)
      * Entries of ``W`` lie in (0,1) and carry an i.i.d. Beta(α,β) prior

    - ``orientation='beta-dir'`` ("Binary ICA"):
      * Rows of ``W`` lie on the simplex
      * Entries of ``H`` lie in (0,1) and carry an i.i.d. Beta(α,β) prior

    The **paper-exact path** uses multiplicative updates plus **L1 renormalization**
    for the simplex-constrained block (“normalize” projection). Under this path,
    the MM objective (Bernoulli NLL + Beta prior) is non-increasing at every
    iteration (up to numerical round-off).

    Parameters
    ----------
    n_components : int
        Rank of the factorization.
    orientation : {'dir-beta', 'beta-dir'} or friendly alias, default='dir-beta'
        Which block is Beta-constrained vs. simplex-constrained.
    alpha, beta : float, default=1.2
        Hyperparameters of the Beta(α,β) prior applied to the Beta-constrained block.
    max_iter : int, default=300
        Maximum number of MM iterations.
    tol : float, default=1e-6
        Stop when the last objective decrease is smaller than ``tol``.
    random_state : int or None, default=None
        Seed for NumPy generator.
    n_init : int, default=1
        Number of restarts; best (lowest objective) is kept.
    projection_method : {'normalize','duchi'}, default='normalize'
        'normalize' is the theory-faithful path (MM majorizer). 'duchi' applies
        Euclidean projection to the simplex (fast; not strictly MM).
    projection_backend : {'numpy'}, default='numpy'
        Placeholder for parity with tests; only NumPy is implemented.
    use_numexpr, use_numba : bool, default=False
        Placeholders for API compatibility; no effect in this pure-NumPy version.
    init_W, init_H : array-like or None
        Optional initial factors (used as-is if shapes agree).

    Attributes
    ----------
    W_ : ndarray of shape (M, K)
        Learned left factor.
    components_ : ndarray of shape (K, N)
        Learned right factor (``H``).
    objective_history_ : list[float]
        One objective value per performed iteration. Length equals ``n_iter_``.
    n_iter_ : int
        Number of iterations run in the chosen (best) restart.
    reconstruction_err_ : float
        Final MAP objective value (same quantity as last entry in history).
    """

    def __init__(
        self,
        n_components: int,
        *,
        orientation: str = "dir-beta",
        alpha: float = 1.2,
        beta: float = 1.2,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        n_init: int = 1,
        projection_method: str = "normalize",
        projection_backend: str = "numpy",
        use_numexpr: bool = False,
        use_numba: bool = False,
        init_W: Optional[Array] = None,
        init_H: Optional[Array] = None,
        eps: float = 1e-12,
    ):
        self.n_components = int(n_components)
        self.orientation = _canonical_orientation(orientation)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.projection_method = str(projection_method).lower()
        self.projection_backend = projection_backend
        self.use_numexpr = bool(use_numexpr)
        self.use_numba = bool(use_numba)
        self.init_W = init_W
        self.init_H = init_H
        self.eps = float(eps)

        # learned attributes
        self.W_: Optional[Array] = None
        self.components_: Optional[Array] = None
        self.n_iter_: int = 0
        self.objective_history_: Optional[list[float]] = None
        self.reconstruction_err_: Optional[float] = None

    # ------------------------- public API -------------------------

    def fit(self, X: Array, mask: Optional[Array] = None):
        X = _to_ndarray(X)
        if mask is not None:
            mask = _to_ndarray(mask)
            if mask.shape != X.shape:
                raise ValueError("mask must have same shape as X")

        M, N = X.shape
        K = self.n_components
        rng = np.random.default_rng(self.random_state)

        best = dict(obj=np.inf, W=None, H=None, hist=None, iters=0)

        for _ in range(self.n_init):
            W, H = self._init_factors(rng, M, N, K)
            history: list[float] = []

            for it in range(self.max_iter):
                if self.orientation == "beta-dir":
                    # H (Beta) then W (row-simplex)
                    H = self._step_beta_block(X, W, H, mask)
                    W = self._step_simplex_block(X, W, H, mask)
                else:  # 'dir-beta'
                    # W (Beta) then H (col-simplex)
                    W = self._step_beta_block(X, W, H, mask)
                    H = self._step_simplex_block(X, W, H, mask)

                obj = objective(X, W, H, mask, self.orientation, self.alpha, self.beta, eps=self.eps)
                history.append(obj)

                if it > 0 and history[-2] - history[-1] < self.tol:
                    break

            final_obj = history[-1]
            if final_obj < best["obj"]:
                best.update(obj=final_obj, W=W.copy(), H=H.copy(), hist=history[:], iters=len(history))

        self.W_ = best["W"]
        self.components_ = best["H"]
        self.objective_history_ = best["hist"]
        self.n_iter_ = int(best["iters"])
        self.reconstruction_err_ = float(best["obj"])
        return self

    def fit_transform(self, X: Array, mask: Optional[Array] = None) -> Array:
        return self.fit(X, mask=mask).W_

    def transform(self, X: Array) -> Array:
        # Standard NMF-like API; here we simply return the learned W_
        if self.W_ is None:
            raise RuntimeError("Call fit() before transform().")
        return self.W_

    def inverse_transform(self, W: Array) -> Array:
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return _clip01(np.asarray(W) @ self.components_, self.eps)

    def score(self, X: Array, mask: Optional[Array] = None) -> float:
        """
        Negative average NLL on (X,mask) under the learned model (higher is better).
        """
        if self.W_ is None or self.components_ is None:
            raise RuntimeError("Call fit() before score().")
        X = _to_ndarray(X)
        if mask is not None:
            mask = _to_ndarray(mask)
        P = _clip01(self.W_ @ self.components_, self.eps)
        return -bernoulli_nll(X, P, mask=mask, average=True, eps=self.eps)

    # ------------------------- internals -------------------------

    def _init_factors(
        self, rng: np.random.Generator, M: int, N: int, K: int
    ) -> Tuple[Array, Array]:
        if self.init_W is not None and self.init_H is not None:
            W = _to_ndarray(self.init_W).copy()
            H = _to_ndarray(self.init_H).copy()
            if W.shape != (M, K) or H.shape != (K, N):
                raise ValueError("init_W/init_H shapes do not match (M,K)/(K,N)")
            # Ensure constraints
            if self.orientation == "beta-dir":
                W = _project_rows_to_simplex(W, self.eps)
                H = _clip01(H, self.eps)
            else:
                H = _project_cols_to_simplex(H, self.eps)
                W = _clip01(W, self.eps)
            return W, H

        # Random init respecting constraints
        if self.orientation == "beta-dir":
            W = rng.random((M, K))
            W = _project_rows_to_simplex(W, self.eps)
            H = _clip01(rng.random((K, N)), self.eps)
        else:  # dir-beta
            H = rng.random((K, N))
            H = _project_cols_to_simplex(H, self.eps)
            W = _clip01(rng.random((M, K)), self.eps)
        return W, H

    def _step_beta_block(self, X: Array, W: Array, H: Array, mask: Optional[Array]) -> Array:
        """
        Update the Beta-constrained block (H for 'beta-dir', W for 'dir-beta').
        """
        if self.orientation == "beta-dir":
            # Update H with Beta prior
            from ._mm_exact import _update_H_beta_dir  # local import to keep surface small
            return _update_H_beta_dir(X, W, H, self.alpha, self.beta, mask, self.eps)
        else:
            from ._mm_exact import _update_W_beta_dirbeta
            return _update_W_beta_dirbeta(X, W, H, self.alpha, self.beta, mask, self.eps)

    def _step_simplex_block(self, X: Array, W: Array, H: Array, mask: Optional[Array]) -> Array:
        """
        Update the simplex-constrained block with either MM-faithful 'normalize'
        (L1 renormalization) or 'duchi' Euclidean projection.
        """
        if self.projection_method == "normalize":
            if self.orientation == "beta-dir":
                # Row-simplex on W
                from ._mm_exact import _update_W_rows_simplex
                return _update_W_rows_simplex(X, W, H, mask, self.eps)
            else:
                from ._mm_exact import _update_H_cols_simplex
                return _update_H_cols_simplex(X, W, H, mask, self.eps)

        # 'duchi' path: multiplicative update + Euclidean projection to simplex
        # (not strictly MM; included for parity with tests)
        if self.orientation == "beta-dir":
            # unconstrained H already updated; here project W rows
            P = _clip01(W @ H, self.eps)
            A = X / P
            B = (1.0 - X) / (1.0 - P)
            num = A @ H.T
            den = B @ H.T
            W *= num / np.maximum(den, self.eps)
            W = _euclidean_project_rows_to_simplex(W, self.eps)
            return W
        else:
            # project H columns
            P = _clip01(W @ H, self.eps)
            A = W.T @ (X / P)
            B = W.T @ ((1.0 - X) / (1.0 - P))
            H *= A / np.maximum(B, self.eps)
            H = _euclidean_project_cols_to_simplex(H, self.eps)
            return H


# ------------ Euclidean (Duchi) projections to the simplex -------------------

def _euclidean_project_rows_to_simplex(W: Array, eps: float) -> Array:
    # Duchi et al. projection row-wise
    M, K = W.shape
    Wp = np.empty_like(W)
    for m in range(M):
        w = np.maximum(W[m], 0.0)
        if w.sum() == 0.0:
            w[:] = 1.0 / K
            Wp[m] = w
            continue
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, K + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(w - theta, eps)
        w /= w.sum()
        Wp[m] = w
    return Wp


def _euclidean_project_cols_to_simplex(H: Array, eps: float) -> Array:
    K, N = H.shape
    Hp = np.empty_like(H)
    for n in range(N):
        h = np.maximum(H[:, n], 0.0)
        if h.sum() == 0.0:
            h[:] = 1.0 / K
            Hp[:, n] = h
            continue
        u = np.sort(h)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, K + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        h = np.maximum(h - theta, eps)
        h /= h.sum()
        Hp[:, n] = h
    return Hp
