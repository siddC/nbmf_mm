"""
NBMF-MM: Bernoulli (mean-parameterized) nonnegative binary matrix factorization
solved by Majorization-Minimization (MM), with a scikit-learn-style API.

This estimator implements the **mean-parameterized Bernoulli** model
:math:`Y_{mn} \\sim \\mathrm{Bernoulli}(\\theta_{mn})` with a *probability*
matrix :math:`\\Theta = W H \\in (0,1)^{M\\times N}` and nonnegativity /
simplex constraints on the factors. It follows the MM algorithm of
**Magron & Févotte (2022)** and supports two *symmetric* orientations of the
constraints and priors:

- ``orientation="beta-dir"`` (paper's default):
  rows of :math:`W` lie on the probability simplex
  (:math:`\\sum_k W_{mk}=1,\\;W_{mk}\\ge 0`); entries of :math:`H` are in
  :math:`(0,1)` with a Beta(:math:`\\alpha,\\beta`) prior.
  In this orientation, **H** is updated by a ratio rule :math:`C/(C+D)`, and
  **W** is updated multiplicatively with the analytic **/N** normalizer that
  preserves the row simplex *without projection*.

- ``orientation="dir-beta"`` (swapped):
  columns of :math:`H` lie on the probability simplex
  (:math:`\\sum_k H_{kn}=1,\\;H_{kn}\\ge 0`); entries of :math:`W` are in
  :math:`(0,1)` with a Beta(:math:`\\alpha,\\beta`) prior.
  In this orientation, **W** uses the ratio rule and **H** uses the multiplicative
  update with the analytic **/M** normalizer to preserve the column simplex.

A **binary mask** can be supplied to perform masked training (matrix completion);
only observed entries contribute to the likelihood and gradients. The estimator
logs the negative log-posterior (up to constants) over iterations in
``objective_history_`` and exposes both a ``score`` (-NLL per observed entry;
higher is better) and ``perplexity`` (:math:`\\exp` of the average NLL).

**Projection choices for the simplex-constrained factor**

- ``projection_method="normalize"`` (paper-exact): uses the closed-form
  **/N** (or **/M**) normalizer inside the multiplicative update, which
  preserves the simplex exactly in exact arithmetic and yields the classical
  MM monotonicity guarantee.

- ``projection_method="duchi"``: applies a Euclidean projection to the
  simplex (Duchi et. al., 2008) *after* the multiplicative step. This is often
  faster in high dimensions and is numerically near-identical to the
  ``"normalize"`` path for practical purposes, but the strict MM monotonicity
  proof applies to the paper-exact path.

**Notes**
- We clip probabilities to :math:`[\\varepsilon, 1-\\varepsilon]` (default
  :math:`\\varepsilon=10^{-12}`) to avoid division by zero and log underflow.
- For the Beta prior, :math:`\\alpha,\\beta \\ge 1` are recommended; the
  MM-based bounds and in-[0,1] preservation rely on this regime.
- Two random-restart strategies (``n_init``) are supported; the best solution
  by final objective is retained.
- Masking is applied consistently in the A/B ratio terms
  :math:`A=Y/\\hat Y,\\; B=(1-Y)/(1-\\hat Y)` and in all averages.

Parameters
----------
n_components : int
    Rank of the factorization (number of Bernoulli components).

orientation : {"dir-beta", "beta-dir"}, default="dir-beta"
    Which factor carries the simplex constraint and which carries the Beta
    prior. See the orientation descriptions above.

alpha, beta : float, default=1.2
    Shape parameters of the Beta prior placed on the *non-simplex* factor.
    Values :math:`\\ge 1` are recommended.

max_iter : int, default=2000
    Maximum number of MM iterations per initialization.

tol : float, default=1e-6
    Relative objective-decrease tolerance for early stopping.

random_state : int or None, default=None
    Seed for reproducible initialization.

n_init : int, default=1
    Number of random initializations. The best run by final objective is kept.

projection_method : {"duchi", "normalize"}, default="duchi"
    Strategy for enforcing the simplex constraint on the simplex-constrained
    factor. ``"normalize"`` uses the **paper-exact** /N or /M normalizer;
    ``"duchi"`` uses an ℓ₁-simplex projection (Duchi et. al., 2008).

projection_backend : {"auto", ...}, default="auto"
    Backend selector for the projection (placeholder for e.g. "numpy"/"numba").

use_numexpr : bool, default=False
    Optional acceleration toggle (if supported elsewhere in the package).

init_W, init_H : array-like of shape (M, K) and (K, N), optional
    Exact initial factors for strict parity testing or warm starts.
    If provided, they are used for the first initialization.

eps : float, default=1e-12
    Numerical epsilon used for clipping probabilities.

Attributes
----------
W_ : ndarray of shape (M, K)
    Learned left factor. If ``orientation="beta-dir"`` its rows sum to 1.

components_ : ndarray of shape (K, N)
    Learned right factor. If ``orientation="dir-beta"`` its columns sum to 1.

n_iter_ : int
    Number of iterations run for the selected (best) initialization.

objective_history_ : list of float
    Trace of the negative log-posterior (up to constants) across iterations.

See Also
--------
NBMF (Bayesian) :
    Fully Bayesian mean-parameterized NBMF with Gibbs/variational inference
    (slower, but with posterior uncertainty).

Notes
-----
The implementation follows Algorithm 1 of Magron & Févotte (2022). In the
paper-exact path, the simplex-constrained factor is updated multiplicatively
with a **/N** (or **/M**) normalizer that *preserves the simplex by construction*;
the factor with the Beta prior is updated by the closed-form ratio
:math:`C/(C+D)`. The swapped orientation is obtained by exchanging the roles
of :math:`W` and :math:`H`, as discussed by the authors.

References
----------
- P. Magron and C. Févotte (2022).
  *A majorization-minimization algorithm for nonnegative binary matrix
  factorization.* IEEE Signal Processing Letters. (See also arXiv:2204.09741)

- J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra (2008).
  *Efficient Projections onto the ℓ₁-Ball for Learning in High Dimensions.*
  Proceedings of ICML.

- A. Lumbreras, L. Filstroff, and C. Févotte (2018).
  *Bayesian Mean-parameterized Nonnegative Binary Matrix Factorization.*
  (Preprint / arXiv).  Background on fully Bayesian NBMF.

Examples
--------
>>> from nbmf_mm import NBMF
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> Y = (rng.random((50, 80)) < 0.2).astype(float)
>>> model = NBMF(n_components=6, orientation="beta-dir",
...              alpha=1.2, beta=1.2, projection_method="normalize",
...              random_state=0).fit(Y)
>>> Yhat = model.W_ @ model.components_
>>> perp = model.perplexity(Y)
>>> round(perp, 6)  # doctest: +SKIP
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =========================
# Utilities / numerics
# =========================

_EPS = 1e-12


def _clip01(X: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Clip array to the open interval (eps, 1 - eps)."""
    return np.clip(X, a_min=eps, a_max=1.0 - eps)


def _compute_A_B(Y: np.ndarray, P: np.ndarray, mask: Optional[np.ndarray], eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    A = Y / P,  B = (1 - Y) / (1 - P), with masking applied (zeros where mask == 0).
    """
    P = _clip01(P, eps)
    if mask is None:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
        return A, B
    A = np.zeros_like(Y)
    B = np.zeros_like(Y)
    nz = mask.astype(bool)
    A[nz] = Y[nz] / P[nz]
    B[nz] = (1.0 - Y[nz]) / (1.0 - P[nz])
    return A, B


def _bernoulli_nll(Y: np.ndarray, P: np.ndarray, mask: Optional[np.ndarray], average: bool, eps: float) -> float:
    """
    Negative log-likelihood: -sum_{mn} [ y log p + (1-y) log(1-p) ].
    """
    P = _clip01(P, eps)
    if mask is None:
        nll = -(np.sum(Y * np.log(P)) + np.sum((1.0 - Y) * np.log1p(-P)))
        denom = Y.size
    else:
        nz = mask.astype(bool)
        nll = -(np.sum(Y[nz] * np.log(P[nz])) + np.sum((1.0 - Y[nz]) * np.log1p(-P[nz])))
        denom = float(np.sum(mask))
    return float(nll / denom if average else nll)


def _beta_log_prior(Z: np.ndarray, alpha: float, beta: float, eps: float) -> float:
    """
    Sum of elementwise log prior (up to constants): sum[(α-1)log Z + (β-1)log(1-Z)].
    """
    Z = _clip01(Z, eps)
    return float(np.sum((alpha - 1.0) * np.log(Z) + (beta - 1.0) * np.log1p(-Z)))


# =========================
# Simplex projections (Duchi)
# =========================

def _project_rows_to_simplex(W: np.ndarray) -> np.ndarray:
    """
    Project each row of W onto the probability simplex:
      S = { x in R^K : x >= 0, sum(x) = 1 }.
    """
    M, K = W.shape
    Wp = np.empty_like(W)
    for m in range(M):
        y = W[m]
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u + (1.0 - cssv) / (np.arange(K) + 1) > 0.0)[0]
        if rho_idx.size == 0:
            Wp[m].fill(1.0 / K)
            continue
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        x = y - theta
        x[x < 0.0] = 0.0
        s = x.sum()
        if s == 0.0:
            x.fill(1.0 / K)
        else:
            x /= s
        Wp[m] = x
    return _clip01(Wp)


def _project_cols_to_simplex(H: np.ndarray) -> np.ndarray:
    """
    Project each column of H onto the probability simplex.
    """
    K, N = H.shape
    Hp = np.empty_like(H)
    for n in range(N):
        y = H[:, n]
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u + (1.0 - cssv) / (np.arange(K) + 1) > 0.0)[0]
        if rho_idx.size == 0:
            Hp[:, n].fill(1.0 / K)
            continue
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        x = y - theta
        x[x < 0.0] = 0.0
        s = x.sum()
        if s == 0.0:
            x.fill(1.0 / K)
        else:
            x /= s
        Hp[:, n] = x
    return _clip01(Hp)


# =========================
# Estimator
# =========================

@dataclass
class NBMF:
    n_components: int
    orientation: str = "dir-beta"     # "dir-beta" (columns of H on simplex) or "beta-dir" (rows of W on simplex)
    alpha: float = 1.2
    beta: float = 1.2
    max_iter: int = 2000
    tol: float = 1e-6
    random_state: Optional[int] = None
    n_init: int = 1
    projection_method: str = "duchi"  # "duchi" or "normalize"
    projection_backend: str = "auto"  # placeholder; numba/numpy selection lives elsewhere if you add it
    use_numexpr: bool = False         # placeholder; not used in this reference
    # Optional exact init for strict parity tests
    init_W: Optional[np.ndarray] = None
    init_H: Optional[np.ndarray] = None
    # internal numeric epsilon
    eps: float = _EPS

    # learned attributes
    W_: np.ndarray = None
    components_: np.ndarray = None
    n_iter_: int = 0
    objective_history_: list[float] = None

    # ------------------ Public API ------------------

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Fit NBMF to binary/probabilistic data X in [0,1], optionally with a mask.
        Stores W_ and components_ (H).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if mask is not None:
            mask = np.asarray(mask, dtype=float)
            if mask.shape != X.shape:
                raise ValueError("mask must have the same shape as X.")

        M, N = X.shape
        K = int(self.n_components)
        best_obj = np.inf
        best_W = None
        best_H = None
        best_hist: list[float] = []

        rng = np.random.default_rng(self.random_state)

        for init_idx in range(max(1, int(self.n_init))):
            # ----- initialize -----
            if self.init_W is not None and self.init_H is not None and init_idx == 0:
                W = np.asarray(self.init_W, dtype=float).copy()
                H = np.asarray(self.init_H, dtype=float).copy()
                if W.shape != (M, K) or H.shape != (K, N):
                    raise ValueError("init_W shape must be (M, K) and init_H shape must be (K, N).")
            else:
                W, H = self._random_init(rng, M, N, K)

            hist: list[float] = []

            # ----- MM loop -----
            obj_prev = np.inf
            for it in range(1, self.max_iter + 1):
                # Current reconstruction and ratios
                P = _clip01(W @ H, self.eps)
                A, B = _compute_A_B(X, P, mask, self.eps)

                if self.orientation == "beta-dir":
                    # ---- H: ratio update (Beta prior on H)
                    C = H * (W.T @ A + (self.alpha - 1.0))
                    D = (1.0 - H) * (W.T @ B + (self.beta - 1.0))
                    H = C / (C + D + self.eps)
                    H = _clip01(H, self.eps)

                    # ---- W: simplex on rows
                    # paper-exact multiplicative step with **/N** (preserves row sums = 1)
                    numer = A @ H.T + B @ (1.0 - H).T   # MxK
                    W_new = W * (numer / float(N))       # **/N normalizer**
                    # For "duchi", project AFTER the paper-exact step (no-op if already on simplex)
                    if self.projection_method == "duchi":
                        W = _project_rows_to_simplex(W_new)
                    else:  # "normalize"
                        # numeric guard: exact row renorm (no-op ideally)
                        rs = W_new.sum(axis=1, keepdims=True)
                        rs[rs == 0.0] = 1.0
                        W = _clip01(W_new / rs, self.eps)

                elif self.orientation == "dir-beta":
                    # ---- W: ratio update (Beta prior on W)
                    C = W * (A @ H.T + (self.alpha - 1.0))
                    D = (1.0 - W) * (B @ (1.0 - H).T + (self.beta - 1.0))
                    W = C / (C + D + self.eps)
                    W = _clip01(W, self.eps)

                    # ---- H: simplex on columns
                    numer = W.T @ A + (1.0 - W).T @ B   # KxN
                    H_new = H * (numer / float(M))       # **/M normalizer**
                    if self.projection_method == "duchi":
                        H = _project_cols_to_simplex(H_new)
                    else:
                        cs = H_new.sum(axis=0, keepdims=True)
                        cs[cs == 0.0] = 1.0
                        H = _clip01(H_new / cs, self.eps)

                else:
                    raise ValueError('orientation must be "beta-dir" or "dir-beta"')

                # Objective (negative log-posterior up to constants)
                P = _clip01(W @ H, self.eps)
                nll = _bernoulli_nll(X, P, mask, average=False, eps=self.eps)
                if self.orientation == "beta-dir":
                    obj = nll - _beta_log_prior(H, self.alpha, self.beta, self.eps)
                else:  # dir-beta
                    obj = nll - _beta_log_prior(W, self.alpha, self.beta, self.eps)
                hist.append(float(obj))

                # Convergence: relative decrease
                if obj_prev < np.inf:
                    rel = (obj_prev - obj) / max(1.0, abs(obj_prev))
                    if rel <= self.tol:
                        break
                obj_prev = obj

            # keep best init
            if obj < best_obj:
                best_obj = obj
                best_W = W
                best_H = H
                best_hist = hist
                self.n_iter_ = it

        # store
        self.W_ = best_W
        self.components_ = best_H
        self.objective_history_ = best_hist
        return self

    def fit_transform(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, mask=mask).W_

    def transform(self, X: np.ndarray, mask: Optional[np.ndarray] = None, max_iter: int = 500, tol: float = 1e-6) -> np.ndarray:
        """
        Estimate W for new X with learned H fixed (common inference mode).
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        H = self.components_
        X = np.asarray(X, dtype=float)
        M, N = X.shape
        K = H.shape[0]
        rng = np.random.default_rng(self.random_state)

        # init W depending on orientation
        if self.orientation == "beta-dir":
            # rows on simplex
            W = rng.random((M, K))
            W = _project_rows_to_simplex(W)
        else:
            # Beta prior on W -> values in (0,1)
            W = _clip01(rng.random((M, K)))

        obj_prev = np.inf
        for it in range(1, max_iter + 1):
            P = _clip01(W @ H, self.eps)
            A, B = _compute_A_B(X, P, mask, self.eps)

            if self.orientation == "beta-dir":
                # update only W (simplex rows) with paper-exact step
                numer = A @ H.T + B @ (1.0 - H).T
                W_new = W * (numer / float(N))
                if self.projection_method == "duchi":
                    W = _project_rows_to_simplex(W_new)
                else:
                    rs = W_new.sum(axis=1, keepdims=True)
                    rs[rs == 0.0] = 1.0
                    W = _clip01(W_new / rs, self.eps)
            else:
                # ratio update for W (Beta prior)
                C = W * (A @ H.T + (self.alpha - 1.0))
                D = (1.0 - W) * (B @ (1.0 - H).T + (self.beta - 1.0))
                W = _clip01(C / (C + D + self.eps), self.eps)

            # early stop
            P = _clip01(W @ H, self.eps)
            obj = _bernoulli_nll(X, P, mask, average=False, eps=self.eps)
            if obj_prev < np.inf and (obj_prev - obj) / max(1.0, abs(obj_prev)) <= tol:
                break
            obj_prev = obj

        return W

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return _clip01(np.asarray(W, dtype=float) @ self.components_, self.eps)

    def score(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Return negative NLL per observed entry (higher is better).
        """
        P = _clip01(self.W_ @ self.components_, self.eps)
        return -_bernoulli_nll(np.asarray(X, dtype=float), P, mask, average=True, eps=self.eps)

    def perplexity(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Return exp(average NLL per observed entry) on X.
        """
        P = _clip01(self.W_ @ self.components_, self.eps)
        nll_avg = _bernoulli_nll(np.asarray(X, dtype=float), P, mask, average=True, eps=self.eps)
        return float(np.exp(nll_avg))

    # ------------------ Helpers ------------------

    def _random_init(self, rng: np.random.Generator, M: int, N: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random initialization that respects the orientation’s constraints.
        """
        if self.orientation == "beta-dir":
            # W rows on simplex; H in (0,1) (Beta-constrained in updates)
            W = rng.random((M, K))
            W = _project_rows_to_simplex(W)
            H = _clip01(rng.random((K, N)), self.eps)
        elif self.orientation == "dir-beta":
            # H columns on simplex; W in (0,1)
            H = rng.random((K, N))
            H = _project_cols_to_simplex(H)
            W = _clip01(rng.random((M, K)), self.eps)
        else:
            raise ValueError('orientation must be "beta-dir" or "dir-beta"')
        return W, H
