"""
NBMF-MM estimator: Bernoulli (mean-parameterized) NMF solved by MM.

Implements Algorithm 1 of Magron & Févotte (2022) with a scikit-learn-style API
and two symmetric orientations:

- "beta-dir"  (Binary ICA):    W rows on the simplex; Beta prior on H (ratio step)
- "dir-beta"  (Aspect Bernoulli): H columns on the simplex; Beta prior on W (ratio step)

Projection choices for the simplex-constrained factor:
- projection_method="normalize" (**default; theory-first**): multiplicative step
  followed by exact L1 renormalization (paper-faithful; monotone objective).
- projection_method="duchi": routed to the **same** paper-exact step for parity
  (Euclidean projection would be a no-op anyway since we already land on the simplex).

References
----------
- P. Magron and C. Févotte (2022).
  “A majorization-minimization algorithm for nonnegative binary matrix
   factorization.” IEEE SPL. (arXiv:2204.09741)
- J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra (2008).
  “Efficient Projections onto the ℓ₁-Ball for Learning in High Dimensions.”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._mm_exact import (
    mm_step_beta_dir,
    mm_step_dir_beta,
    mm_update_W_simplex_beta_dir,
    mm_update_W_dir_beta,
    objective as _objective,
    bernoulli_nll as _nll,
    _clip01,
)


_EPS = 1e-12


def _canon_orientation(o: str) -> str:
    s = str(o).strip().lower().replace("_", "-").replace(" ", "-")
    dirbeta = {"dir-beta", "dirbeta", "aspect-bernoulli", "aspect", "aspectbernoulli"}
    betadir = {"beta-dir", "betadir", "binary-ica", "binaryica", "bica"}
    if s in dirbeta:
        return "dir-beta"
    if s in betadir:
        return "beta-dir"
    if s in {"dir-beta", "beta-dir"}:
        return s
    raise ValueError('orientation must be "beta-dir" or "dir-beta"')


@dataclass
class NBMF:
    n_components: int
    orientation: str = "dir-beta"
    alpha: float = 1.2
    beta: float = 1.2
    max_iter: int = 2000
    tol: float = 1e-6
    random_state: Optional[int] = None
    n_init: int = 1
    projection_method: str = "normalize"   # theory-first default
    projection_backend: str = "auto"
    use_numexpr: bool = False
    use_numba: bool = False
    init_W: Optional[np.ndarray] = None
    init_H: Optional[np.ndarray] = None
    eps: float = _EPS

    # learned attributes
    W_: np.ndarray = None
    components_: np.ndarray = None
    n_iter_: int = 0
    objective_history_: list[float] = None
    reconstruction_err_: float = None

    def __post_init__(self):
        self.orientation = _canon_orientation(self.orientation)

    # ------------------ Public API ------------------

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None):
        # Accept SciPy sparse inputs (converted to dense)
        try:
            import scipy.sparse as sp  # type: ignore
            if sp.issparse(X):
                X = X.toarray()
            if mask is not None and sp.issparse(mask):
                mask = mask.toarray()
        except Exception:
            pass

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if mask is not None:
            mask = np.asarray(mask, dtype=float)
            if mask.shape != X.shape:
                raise ValueError("mask must have the same shape as X.")

        M, N = X.shape
        K = int(self.n_components)

        rng = np.random.default_rng(self.random_state)
        best_obj = np.inf
        best_W = None
        best_H = None
        best_hist: list[float] = []
        best_iter = 0

        for init_idx in range(max(1, int(self.n_init))):
            if self.init_W is not None and self.init_H is not None and init_idx == 0:
                W = np.asarray(self.init_W, dtype=float).copy()
                H = np.asarray(self.init_H, dtype=float).copy()
                if W.shape != (M, K) or H.shape != (K, N):
                    raise ValueError("init_W shape must be (M, K) and init_H shape must be (K, N).")
            else:
                W, H = self._random_init(rng, M, N, K)

            hist: list[float] = []
            obj_prev = np.inf
            it_done = 0

            for it in range(1, self.max_iter + 1):
                # === Paper-exact one full MM step (normalize/duchi both route here) ===
                if self.orientation == "beta-dir":
                    W, H = mm_step_beta_dir(X, W, H, self.alpha, self.beta, mask, self.eps)
                else:
                    W, H = mm_step_dir_beta(X, W, H, self.alpha, self.beta, mask, self.eps)

                # Log MAP objective (NLL − log prior)
                obj = _objective(X, W, H, mask, self.orientation, self.alpha, self.beta, eps=self.eps)
                hist.append(float(obj))
                it_done = it

                # Early stopping on relative decrease
                if obj_prev < np.inf and (obj_prev - obj) / max(1.0, abs(obj_prev)) <= self.tol:
                    break
                obj_prev = obj

            if obj < best_obj:
                best_obj = obj
                best_W = W
                best_H = H
                best_hist = hist
                best_iter = it_done

        # Store best solution
        self.W_ = best_W
        self.components_ = best_H
        self.objective_history_ = best_hist
        self.n_iter_ = best_iter

        # Reproducibility metric: average NLL on training data
        P = _clip01(self.W_ @ self.components_, self.eps)
        self.reconstruction_err_ = _nll(X, P, mask, average=True, eps=self.eps)
        return self

    def fit_transform(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, mask=mask).W_

    def transform(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Estimate W for new X with learned H fixed."""
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")

        # Accept SciPy sparse inputs at inference as well
        try:
            import scipy.sparse as sp  # type: ignore
            if sp.issparse(X):
                X = X.toarray()
            if mask is not None and sp.issparse(mask):
                mask = mask.toarray()
        except Exception:
            pass

        H = self.components_
        X = np.asarray(X, dtype=float)
        M, N = X.shape
        K = H.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.orientation == "beta-dir":
            # strictly positive row-simplex init
            W = rng.random((M, K))
            W = W / W.sum(axis=1, keepdims=True)
        else:
            W = np.clip(rng.random((M, K)), 0.0, 1.0)

        obj_prev = np.inf
        for _ in range(1, max_iter + 1):
            if self.orientation == "beta-dir":
                # One multiplicative+renorm step with H fixed
                W = mm_update_W_simplex_beta_dir(X, W, H, mask, self.eps)
            else:
                # One Beta-regularized ratio step with H fixed
                W = mm_update_W_dir_beta(X, W, H, self.alpha, self.beta, mask, self.eps)

            P = _clip01(W @ H, self.eps)
            obj = _nll(X, P, mask, average=False, eps=self.eps)
            if obj_prev < np.inf and (obj_prev - obj) / max(1.0, abs(obj_prev)) <= tol:
                break
            obj_prev = obj

        return W

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return _clip01(np.asarray(W, dtype=float) @ self.components_, self.eps)

    def score(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Return −NLL per observed entry (higher is better)."""
        P = _clip01(self.W_ @ self.components_, self.eps)
        return -_nll(np.asarray(X, dtype=float), P, mask, average=True, eps=self.eps)

    def perplexity(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Return exp(average NLL per observed entry) on X."""
        P = _clip01(self.W_ @ self.components_, self.eps)
        nll_avg = _nll(np.asarray(X, dtype=float), P, mask, average=True, eps=self.eps)
        return float(np.exp(nll_avg))

    # ------------------ Helpers ------------------

    def _random_init(
        self, rng: np.random.Generator, M: int, N: int, K: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random initialization that respects the orientation’s constraints."""
        if self.orientation == "beta-dir":
            # W: strictly positive rows normalized to 1
            W = rng.random((M, K))
            W = W / W.sum(axis=1, keepdims=True)
            # H: in (0,1)
            H = np.clip(rng.random((K, N)), 0.0, 1.0)
        elif self.orientation == "dir-beta":
            # H: strictly positive columns normalized to 1
            H = rng.random((K, N))
            H = H / H.sum(axis=0, keepdims=True)
            # W: in (0,1)
            W = np.clip(rng.random((M, K)), 0.0, 1.0)
        else:
            raise ValueError('orientation must be "beta-dir" or "dir-beta"')
        return W, H
