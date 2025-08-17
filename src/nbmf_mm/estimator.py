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

References
----------
- P. Magron & C. Févotte (2022), "A majorization-minimization algorithm for
  nonnegative binary matrix factorization" (Algorithm 1; Eqs. 14-20).  # arXiv:2204.09741
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Literal, Iterable
import numpy as np

ArrayLike = Union[np.ndarray]

# ---------- small utilities ----------

def _rng(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)

def _project_rows_to_simplex(W: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    W = np.maximum(W, eps)
    s = W.sum(axis=1, keepdims=True)
    bad = (s[:, 0] <= eps)
    if np.any(bad):
        W[bad] = 1.0 / W.shape[1]
        s = W.sum(axis=1, keepdims=True)
    return W / s

def _project_cols_to_simplex(H: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    H = np.maximum(H, eps)
    s = H.sum(axis=0, keepdims=True)
    bad = (s[0, :] <= eps)
    if np.any(bad):
        H[:, bad] = 1.0 / H.shape[0]
        s = H.sum(axis=0, keepdims=True)
    return H / s

def _dirichlet_rows(n_rows: int, n_cols: int, rng, concentration: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    alpha = np.full(n_cols, max(concentration, eps), dtype=np.float64)
    return _project_rows_to_simplex(rng.dirichlet(alpha, size=n_rows), eps=eps)

def _dirichlet_cols(n_rows: int, n_cols: int, rng, concentration: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    alpha = np.full(n_rows, max(concentration, eps), dtype=np.float64)
    H = rng.dirichlet(alpha, size=n_cols).T  # (n_rows, n_cols)
    return _project_cols_to_simplex(H, eps=eps)

def _beta_matrix(n_rows: int, n_cols: int, rng, a: float, b: float, eps: float) -> np.ndarray:
    X = rng.beta(max(a, eps), max(b, eps), size=(n_rows, n_cols))
    return np.clip(X, eps, 1.0 - eps)

def _safe_bern_nll(X: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray], eps: float) -> float:
    Vc = np.clip(V, eps, 1.0 - eps)
    if mask is None:
        return float(-(X * np.log(Vc) + (1.0 - X) * np.log(1.0 - Vc)).sum())
    M = mask.astype(X.dtype)
    return float(-((M * X) * np.log(Vc) + (M * (1.0 - X)) * np.log(1.0 - Vc)).sum())

def _safe_beta_neglogprior(Z: np.ndarray, alpha_vec: np.ndarray, beta_vec: np.ndarray, eps: float) -> float:
    Zc = np.clip(Z, eps, 1.0 - eps)
    return float(-(((alpha_vec - 1.0) * np.log(Zc)) + ((beta_vec - 1.0) * np.log(1.0 - Zc))).sum())

def _broadcast_hparam(z, name: str, K: int, dtype) -> np.ndarray:
    z = np.asarray(z, dtype=dtype).reshape(-1)
    if z.size == 1:
        out = np.full((K, 1), float(z.item()), dtype=dtype)
    elif z.size == K:
        out = z.reshape((K, 1)).astype(dtype, copy=False)
    else:
        raise ValueError(f"{name!r} must be scalar or have length n_components.")
    return out

def _normalize_orientation(s: str) -> str:
    key = str(s).strip().lower()
    ab = {"aspect bernoulli", "ab", "dir-beta", "dirichlet-beta", "dirichlet–beta", "dir – beta"}
    bica = {"binary ica", "bica", "beta-dir", "beta‑dir", "beta‑dirichlet", "beta-dirichlet"}
    blda = {"binary lda", "blda", "dir-dir", "dirichlet-dirichlet", "dirichlet–dirichlet", "dir – dir"}
    if key in ab: return "dir-beta"
    if key in bica: return "beta-dir"
    if key in blda: return "dir-dir"
    # accept the canonical tags as-is
    if key in {"dir-beta", "beta-dir"}:
        return key
    raise ValueError(f"Unknown orientation {s!r}.")

# ---------- estimator ----------

class BernoulliNMF_MM:
    """
    Mean‑parameterized Bernoulli NMF via MM (NBMF‑MM), scikit‑learn style.

    Factorizes a binary/[0,1] matrix X ∈ R^{M×N} as W ∈ R_+^{M×K}, H ∈ R_+^{K×N}
    with mean parameter X̂ = W @ H ∈ [0,1]^{M×N}. One factor is constrained to be
    row/column‑simplex (Dirichlet‑like), the other gets a Beta(α,β) prior.

    Parameters
    ----------
    n_components : int
        Factorization rank K.
    alpha, beta : float or array-like of length K, default=1.0
        Beta prior parameters for the factor carrying the Beta prior; broadcast if scalar.
    orientation : str, default='dir-beta'
        Which factor is simplex‑constrained:
          - 'dir-beta' (Aspect Bernoulli; **default**): H columns on simplex, Beta prior on W.
          - 'beta-dir'  (binary ICA): W rows on simplex,     Beta prior on H.
          - 'dir-dir'   (binary LDA): **not supported** here; raises NotImplementedError.
        Synonyms accepted (case‑insensitive): {"Aspect Bernoulli","AB","Dir-Beta",...}
        and {"binary ICA","bICA","Beta-Dir",...}.
    init : {'random','nndsvd','custom'}, default='random'
        Factor initialization strategy; 'nndsvd' uses sklearn NMF then projects/clips.
    max_iter : int, default=200
        Maximum alternating MM iterations.
    tol : float, default=1e-4
        Relative tolerance on objective decrease for early stopping.
    random_state : int or numpy.random.Generator, optional
        Seed/Generator for reproducibility.
    verbose : int, default=0
        >=2 prints per‑iteration objective.
    eps : float, default=1e-9
        Numerical epsilon for clipping probabilities/divisions.
    dtype : numpy dtype, default=np.float64
        Internal floating dtype.

    Attributes
    ----------
    W_ : (M,K) ndarray
        Learned W. For 'beta-dir', rows sum to 1.
    components_ : (K,N) ndarray
        Learned H. For 'dir-beta', columns sum to 1.
    reconstruction_err_ : float
        Final (masked) objective value = Bernoulli NLL + Beta negative log‑prior.
    objective_history_ : list[float]
        Objective trajectory.
    n_iter_ : int
        Number of iterations run.

    Notes
    -----
    - 'beta-dir' orientation matches Algorithm 1 updates in Magron & Févotte (2022).
    - Masked training: entries with mask=0 are ignored in R1/R0 statistics and in the
      NLL; the W‑update normalization uses the number of observed entries per row
      (masked analogue of Eq. (19) ⇒ Eq. (20)).  # arXiv:2204.09741
    """

    def __init__(self,
                 n_components: int,
                 *,
                 alpha: Union[float, ArrayLike] = 1.0,
                 beta:  Union[float, ArrayLike] = 1.0,
                 orientation: str = "dir-beta",
                 init: str = "random",
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 random_state: Optional[Union[int, np.random.Generator]] = None,
                 verbose: int = 0,
                 eps: float = 1e-9,
                 dtype=np.float64):
        if not isinstance(n_components, (int, np.integer)) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.alpha = alpha
        self.beta = beta
        self.orientation = orientation
        self.init = init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.eps = float(eps)
        self.dtype = dtype

        # learned
        self.W_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.reconstruction_err_: float = np.nan
        self.objective_history_: list[float] = []
        self.n_iter_: int = 0

    # ---- API ----

    def fit(self, X: ArrayLike, y=None,
            *,
            mask: Optional[ArrayLike] = None,
            W_init: Optional[ArrayLike] = None,
            H_init: Optional[ArrayLike] = None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (M,N)
            Binary or [0,1]-valued matrix.
        mask : array-like of shape (M,N), optional
            Binary mask (1 = observed, 0 = ignored) for matrix completion training.
        W_init, H_init : array-like, optional
            Custom initial factors (used when init='custom').

        Returns
        -------
        self
        """
        Xf = np.asarray(X, dtype=self.dtype, order="C")
        M, N = Xf.shape
        Mask = None if mask is None else np.asarray(mask, dtype=self.dtype, order="C")
        if (Mask is not None) and Mask.shape != (M, N):
            raise ValueError("mask shape must match X.")

        ori = _normalize_orientation(self.orientation)

        # broadcast α, β to (K,1)
        self._alpha_vec = _broadcast_hparam(self.alpha, "alpha", self.n_components, self.dtype)
        self._beta_vec  = _broadcast_hparam(self.beta,  "beta",  self.n_components, self.dtype)

        rng = _rng(self.random_state)

        if ori == "beta-dir":
            W, H, hist = self._fit_beta_dir_core(Xf, Mask, rng, W_init=W_init, H_init=H_init)
            self.W_, self.components_, self.objective_history_ = W, H, hist
        elif ori == "dir-beta":
            # solve beta‑dir on X^T, then map back
            Wp, Hp, hist = self._fit_beta_dir_core(
                Xf.T, None if Mask is None else Mask.T, rng,
                W_init=None if H_init is None else H_init.T,
                H_init=None if W_init is None else W_init.T
            )
            self.W_ = Hp.T          # Beta on W
            self.components_ = Wp.T # column‑simplex H
            self.objective_history_ = hist
        else:
            raise NotImplementedError("Dir–Dir ('binary LDA') is not implemented in this MM solver.")

        self.reconstruction_err_ = float(self.objective_history_[-1]) if self.objective_history_ else np.nan
        self.n_iter_ = len(self.objective_history_)
        return self

    def fit_transform(self, X: ArrayLike, y=None, **fit_kwargs) -> np.ndarray:
        """Fit and return W (as in scikit‑learn)."""
        return self.fit(X, **fit_kwargs).W_

    def transform(self, X: ArrayLike, *, max_iter_W: int = 200, tol_W: float = 1e-5) -> np.ndarray:
        """Given learned components, estimate W for new X (unmasked)."""
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")

        Xf = np.asarray(X, dtype=self.dtype, order="C")
        M, N = Xf.shape
        K = self.n_components
        eps = self.eps
        rng = _rng(self.random_state)

        ori = _normalize_orientation(self.orientation)
        if ori == "beta-dir":
            H = self.components_
            W = _project_rows_to_simplex(np.maximum(rng.random((M, K)) * 0.5, eps), eps=eps)
            prev = np.inf
            for _ in range(max_iter_W):
                V  = np.clip(W @ H, eps, 1.0 - eps)
                R1 = Xf / V
                R0 = (1.0 - Xf) / (1.0 - V)
                W *= (R1 @ H.T) + (R0 @ (1.0 - H).T)
                W /= float(N)
                W  = _project_rows_to_simplex(W, eps=eps)
                V  = np.clip(W @ H, eps, 1.0 - eps)
                curr = _safe_bern_nll(Xf, V, mask=None, eps=eps)
                if abs(prev - curr) / max(1.0, abs(prev)) < tol_W:
                    break
                prev = curr
            return W.astype(self.dtype, copy=False)

        else:  # dir‑beta
            H = self.components_  # (K,N), column‑simplex
            W = np.clip(rng.random((M, K)), eps, 1.0 - eps)
            prev = np.inf
            for _ in range(max_iter_W):
                V  = np.clip(W @ H, eps, 1.0 - eps)
                R1 = Xf / V
                R0 = (1.0 - Xf) / (1.0 - V)
                C = W * (R1 @ H.T) + self._alpha_vec.T
                D = (1.0 - W) * (R0 @ (1.0 - H).T) + self._beta_vec.T
                W = np.clip(C / (C + D), eps, 1.0 - eps)
                V  = np.clip(W @ H, eps, 1.0 - eps)
                curr = _safe_bern_nll(Xf, V, mask=None, eps=eps) + _safe_beta_neglogprior(W, self._alpha_vec.T, self._beta_vec.T, eps)
                if abs(prev - curr) / max(1.0, abs(prev)) < tol_W:
                    break
                prev = curr
            return W.astype(self.dtype, copy=False)

    def inverse_transform(self, W: ArrayLike) -> np.ndarray:
        """Return mean reconstruction X̂ = W @ H in [0,1]."""
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return np.clip(np.asarray(W, dtype=self.dtype) @ self.components_, self.eps, 1.0 - self.eps)

    # ---------- core solver for Beta–Dir (binary ICA) ----------

    def _fit_beta_dir_core(self, Xf: np.ndarray, Mask: Optional[np.ndarray], rng: np.random.Generator,
                           W_init: Optional[ArrayLike] = None,
                           H_init: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        M, N = Xf.shape; K = self.n_components; eps = self.eps; dtype = self.dtype

        # init
        if self.init == "custom":
            if W_init is None or H_init is None:
                raise ValueError("init='custom' requires W_init and H_init.")
            W = np.asarray(W_init, dtype=dtype, order="C");  H = np.asarray(H_init, dtype=dtype, order="C")
            if W.shape != (M, K) or H.shape != (K, N):
                raise ValueError("W_init/H_init shapes are incompatible with X.")
            W = _project_rows_to_simplex(W, eps=eps);  H = np.clip(H, eps, 1.0 - eps)
        elif self.init == "nndsvd":
            try:
                from sklearn.decomposition import NMF as _SKNMF
                sk = _SKNMF(n_components=K, init="nndsvd", solver="cd",
                            beta_loss="frobenius", max_iter=400,
                            random_state=rng.integers(0, 2**31-1))
                W0 = sk.fit_transform(Xf); H0 = sk.components_
                W  = _project_rows_to_simplex(W0.astype(dtype, copy=False), eps=eps)
                H  = np.clip(H0.astype(dtype, copy=False), eps, 1.0 - eps)
                q  = np.quantile(H, 0.99, axis=1, keepdims=True); q = np.where(q <= eps, 1.0, q)
                H  = np.clip(H / q, eps, 1.0 - eps)
            except Exception:
                W = _dirichlet_rows(M, K, rng, concentration=1.0, eps=eps).astype(dtype, copy=False)
                a = float(np.mean(np.asarray(self.alpha))); b = float(np.mean(np.asarray(self.beta)))
                H = _beta_matrix(K, N, rng, a=a, b=b, eps=eps).astype(dtype, copy=False)
        else:
            W = _dirichlet_rows(M, K, rng, concentration=1.0, eps=eps).astype(dtype, copy=False)
            a = float(np.mean(np.asarray(self.alpha))); b = float(np.mean(np.asarray(self.beta)))
            H = _beta_matrix(K, N, rng, a=a, b=b, eps=eps).astype(dtype, copy=False)

        hist: list[float] = []
        prev = np.inf

        # constants for masked updates
        if Mask is None:
            Mask = np.ones_like(Xf, dtype=dtype)
        row_obs = np.maximum(Mask.sum(axis=1, keepdims=True), 1.0)

        for it in range(1, self.max_iter + 1):
            V  = np.clip(W @ H, eps, 1.0 - eps)
            R1 = (Mask * Xf) / V
            R0 = (Mask * (1.0 - Xf)) / (1.0 - V)

            # H‑update (closed‑form C/(C+D)) with Beta prior on H
            WT_R1 = W.T @ R1
            WT_R0 = W.T @ R0
            C = H * WT_R1 + self._alpha_vec
            D = (1.0 - H) * WT_R0 + self._beta_vec
            H = np.clip(C / (C + D), eps, 1.0 - eps)

            # W‑update (multiplicative) + row‑simplex projection
            W *= (R1 @ H.T) + (R0 @ (1.0 - H).T)
            W /= row_obs  # masked analogue of division by N (Eq. 20 via Eq. 19 with mask)
            W  = _project_rows_to_simplex(W, eps=eps)

            V  = np.clip(W @ H, eps, 1.0 - eps)
            obj = _safe_bern_nll(Xf, V, mask=Mask, eps=eps) + _safe_beta_neglogprior(H, self._alpha_vec, self._beta_vec, eps)
            hist.append(float(obj))

            if self.verbose >= 2:
                print(f"[it={it:04d}] obj={obj:.6f}")
            if abs(prev - obj) / max(1.0, abs(prev)) < self.tol:
                break
            prev = obj

        return W.astype(dtype, copy=False), H.astype(dtype, copy=False), hist
