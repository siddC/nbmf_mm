# SPDX-License-Identifier: BSD-3-Clause
"""
Exact MM helpers for mean-parameterized Bernoulli NMF (NBMF-MM).

This module implements the *paper-faithful* Majorization-Minimization (MM)
updates from:

  - P. Magron and C. Févotte (2022),
    "A majorization-minimization algorithm for nonnegative binary matrix
     factorization", IEEE Signal Processing Letters.
    arXiv:2204.09741 (Algorithm 1, Eqs. (14)-(20)).

Model:
    Y ~ Bernoulli(P),   with  P = W @ H

Two symmetric orientations are supported:

- "beta-dir"  (Binary ICA in the paper's nomenclature):
    • W: rows on the probability simplex (sum to 1, nonnegative).
    • H: entries in (0, 1) with Beta(α, β) prior.
    • Updates: H via C/(C+D) (Eqs. (14)-(16)), then W via multiplicative
      simplex update (Eq. (20)).

- "dir-beta"  (Aspect Bernoulli):
    • H: columns on the probability simplex.
    • W: entries in (0, 1) with Beta(α, β) prior.
    • Updates: W via C/(C+D) (transpose-symmetric of Eqs. (14)-(16)),
      then H via multiplicative simplex update (transpose of Eq. (20)).

Masking:
    If a binary mask M ∈ {0,1}^{MxN} is supplied, all sums are taken over
    observed entries only. In the multiplicative simplex steps the normalizer
    N (resp. M) from Eq. (20) becomes the number of observed entries per row
    (resp. column). This keeps the MM majorizer tight and preserves monotonicity.

All computations are purely NumPy and numerically safe (values clipped to
(ε, 1-ε)). These helpers are used by the estimator for the "normalize"
(theory-first) path and by the tests for strict parity.

Notes
-----
- The objective minimized here is the negative log-posterior up to
  constants:  f(W,H) - log p_beta(Z), where Z is the Beta-regularized factor.
- We keep α, β scalars; broadcasting to matrices is supported.

"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _clip01(X: Array, eps: float = 1e-12) -> Array:
    """Clip probabilities into (eps, 1-eps) without kwargs ambiguity."""
    # NB: avoid passing a_min/a_max twice (positional + kwargs) to np.clip.
    return np.clip(X, eps, 1.0 - eps, out=np.empty_like(X))


def bernoulli_nll(
    Y: Array,
    P: Array,
    mask: Optional[Array] = None,
    *,
    average: bool = False,
    eps: float = 1e-12,
) -> float:
    """Negative log-likelihood:  -∑ [ y log p + (1-y) log(1-p) ] over observed entries.

    Parameters
    ----------
    Y : array, shape (M, N)
    P : array, shape (M, N)
        Bernoulli means (probabilities), clipped internally to (eps, 1-eps).
    mask : array, optional
        Binary mask of observed entries (same shape as Y). If None, all entries observed.
    average : bool
        If True, divide by the number of observed entries.
    eps : float
        Numerical safety for logs.

    Returns
    -------
    float
        Negative log-likelihood (masked if provided).
    """
    P = _clip01(P, eps)
    if mask is not None:
        Y = Y * mask
        one_minus_Y = (1.0 - Y) * mask
    else:
        one_minus_Y = 1.0 - Y
    ll = Y * np.log(P) + one_minus_Y * np.log(1.0 - P)
    nobs = float(mask.sum()) if mask is not None else Y.size
    val = -float(ll.sum())
    return val / nobs if average else val


def beta_log_prior(Z: Array, alpha: float, beta: float, eps: float = 1e-12) -> float:
    """(Unnormalized) log Beta prior contribution: ∑[(α-1)log Z + (β-1)log(1-Z)]."""
    Z = _clip01(Z, eps)
    return float((alpha - 1.0) * np.log(Z).sum() + (beta - 1.0) * np.log(1.0 - Z).sum())


def objective(
    Y: Array,
    W: Array,
    H: Array,
    mask: Optional[Array],
    orientation: str,
    alpha: float,
    beta: float,
    *,
    eps: float = 1e-12,
) -> float:
    """MAP objective (negative log-posterior, ignoring constants)."""
    P = _clip01(W @ H, eps)
    obj = bernoulli_nll(Y, P, mask=mask, average=False, eps=eps)
    if orientation == "beta-dir":
        obj -= beta_log_prior(H, alpha, beta, eps=eps)
    elif orientation == "dir-beta":
        obj -= beta_log_prior(W, alpha, beta, eps=eps)
    else:
        raise ValueError('orientation must be "beta-dir" or "dir-beta"')
    return obj


def _masked_ratios(
    Y: Array, P: Array, mask: Optional[Array], eps: float
) -> Tuple[Array, Array]:
    """Return A=Y/P and B=(1-Y)/(1-P), masked if provided."""
    P = _clip01(P, eps)
    if mask is None:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
    else:
        A = (Y * mask) / P
        B = ((1.0 - Y) * mask) / (1.0 - P)
    return A, B


def _row_norm(mask: Optional[Array], N: int, eps: float) -> Array:
    """Row-wise observed counts (shape M×1); defaults to N if mask is None."""
    if mask is None:
        return np.full((N, ), np.nan)  # placeholder never used
    raise RuntimeError("internal misuse of _row_norm")


def _row_counts(mask: Optional[Array], M: int, N: int, eps: float) -> Array:
    """Row-wise normalizer: number of observed columns per row, shape (M, 1)."""
    if mask is None:
        return np.full((M, 1), float(N))
    cnt = mask.sum(axis=1, keepdims=True).astype(float)
    return np.maximum(cnt, eps)


def _col_counts(mask: Optional[Array], M: int, N: int, eps: float) -> Array:
    """Column-wise normalizer: number of observed rows per column, shape (1, N)."""
    if mask is None:
        return np.full((1, N), float(M))
    cnt = mask.sum(axis=0, keepdims=True).astype(float)
    return np.maximum(cnt, eps)


# ---------------------------------------------------------------------
# Paper-faithful MM updates (Algorithm 1 and its transpose-symmetric)
# ---------------------------------------------------------------------

def mm_update_H_beta_dir(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """Eq. (14) with the matrix form of Algorithm 1 (lines 4-6).

    C = H ⊙ ( Wᵀ (Y ./ (WH)) + α - 1 )
    D = (1 - H) ⊙ ( Wᵀ ((1 - Y) ./ (1 - WH)) + β - 1 )
    H ← C / (C + D)
    """
    P = _clip01(W @ H, eps)
    A, B = _masked_ratios(Y, P, mask, eps)
    # Broadcast α, β if scalars; allow numpy broadcasting against (K,N)
    C = H * (W.T @ A + (alpha - 1.0))
    D = (1.0 - H) * (W.T @ B + (beta - 1.0))
    H = C / _clip01(C + D, eps)
    return _clip01(H, eps)


def mm_update_W_dir_beta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """Transpose-symmetric of Eq. (14): Beta prior on W (C/(C+D) for W).

    Cw = W ⊙ ( (Y./P) Hᵀ + α - 1 )
    Dw = (1 - W) ⊙ ( ((1-Y)/(1-P)) (1 - H)ᵀ + β - 1 )
    W ← Cw / (Cw + Dw)
    """
    P = _clip01(W @ H, eps)
    A, B = _masked_ratios(Y, P, mask, eps)
    Cw = W * (A @ H.T + (alpha - 1.0))
    Dw = (1.0 - W) * (B @ (1.0 - H).T + (beta - 1.0))
    W = Cw / _clip01(Cw + Dw, eps)
    return _clip01(W, eps)


def mm_update_W_simplex_beta_dir(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """Eq. (20) (row-simplex multiplicative step for W) with mask-aware normalizer.

    W ← W ⊙ ( A Hᵀ + B (1 - H)ᵀ ) / N_row,
    where N_row[m] = number of observed entries in row m (defaults to N if unmasked).
    """
    M, N = Y.shape
    P = _clip01(W @ H, eps)
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (A @ H.T) + (B @ (1.0 - H).T)  # shape (M,K)
    # Mask-aware N_row (Eq. (20) uses N if fully observed)
    Nrow = _row_counts(mask, M, N, eps)  # (M,1)
    W = W * (numer / Nrow)
    # This multiplicative step preserves each row sum exactly.
    return _clip01(W, eps)


def mm_update_H_simplex_dir_beta(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """Transpose of Eq. (20) (column-simplex multiplicative step for H) with mask-aware normalizer.

    H ← H ⊙ ( Wᵀ A + (1 - W)ᵀ B ) / M_col,
    where M_col[n] = number of observed entries in column n (defaults to M if unmasked).
    """
    M, N = Y.shape
    P = _clip01(W @ H, eps)
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (W.T @ A) + ((1.0 - W).T @ B)  # shape (K,N)
    Mcol = _col_counts(mask, M, N, eps)    # (1,N)
    H = H * (numer / Mcol)
    return _clip01(H, eps)


# ---------------------------------------------------------------------
# One full MM step for each orientation (helpers used by tests & estimator)
# ---------------------------------------------------------------------

def mm_step_beta_dir(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array] = None,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    """One paper-exact MM iteration for orientation='beta-dir'.

    Order (Algorithm 1): update H via C/(C+D), then W via multiplicative step.
    """
    # 1) H : Beta-regularized via C/(C+D)
    H = mm_update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
    # 2) W : row-simplex multiplicative update (Eq. 20) with mask-aware normalizer
    W = mm_update_W_simplex_beta_dir(Y, W, H, mask, eps)
    return W, H


def mm_step_dir_beta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array] = None,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    """One paper-exact MM iteration for orientation='dir-beta'.

    Order (transpose-symmetric): update W via C/(C+D), then H via multiplicative step.
    """
    # 1) W : Beta-regularized via C/(C+D)
    W = mm_update_W_dir_beta(Y, W, H, alpha, beta, mask, eps)
    # 2) H : column-simplex multiplicative update with mask-aware normalizer
    H = mm_update_H_simplex_dir_beta(Y, W, H, mask, eps)
    return W, H
