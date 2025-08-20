"""
Exact MM helpers for mean-parameterized Bernoulli NMF (NBMF‑MM).

This module implements the paper‑faithful Majorization–Minimization (MM)
updates from:

  • P. Magron and C. Févotte (2022),
    “A majorization–minimization algorithm for nonnegative binary matrix
     factorization”, IEEE Signal Processing Letters. (arXiv:2204.09741)
    — Algorithm 1, Eqs. (14)–(20).

Model
-----
Y ~ Bernoulli(P), with  P = W @ H  and probabilities P ∈ (0,1)^{M×N}.

Orientations
------------
- "beta-dir"  (Binary ICA in the paper’s nomenclature):
    • W: rows on the probability simplex (∑_k W_{mk} = 1, W ≥ 0)
    • H: entries in (0,1) with Beta(α,β) prior
    • Step order: update H via C/(C+D), then W via multiplicative + L1 row‑renorm

- "dir-beta"  (Aspect Bernoulli):
    • H: columns on the probability simplex (∑_k H_{kn} = 1, H ≥ 0)
    • W: entries in (0,1) with Beta(α,β) prior
    • Step order: update W via C/(C+D), then H via multiplicative + L1 col‑renorm

Masking
-------
If a binary mask M ∈ {0,1}^{M×N} is supplied, sums are taken over observed
entries only. In the simplex steps we still use multiplicative + **L1
renormalization** (not a fixed constant), which preserves the simplex exactly
even with missingness.

Important implementation notes
------------------------------
• We do **not** clip W/H during the updates (to keep sums exact and preserve
  the MM identities). We only clip probabilities **when evaluating** the NLL.
• The ratio (C/(C+D)) updates keep the regularized factor strictly in (0,1).
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _clip01(X: Array, eps: float = 1e-12) -> Array:
    """Clip into (eps, 1−eps) without kwarg duplication issues."""
    return np.clip(X, eps, 1.0 - eps, out=np.empty_like(X))


def bernoulli_nll(
    Y: Array,
    P: Array,
    mask: Optional[Array] = None,
    *,
    average: bool = False,
    eps: float = 1e-12,
) -> float:
    """Negative log-likelihood −∑ [ y log p + (1−y) log(1−p) ] over observed entries."""
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
    """(Unnormalized) log Beta prior: ∑[(α−1)log Z + (β−1)log(1−Z)]."""
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
    """MAP objective (negative log-posterior up to constants)."""
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
    """Return A=Y/P and B=(1−Y)/(1−P), masked if provided (safe denominators)."""
    P = np.clip(P, eps, 1.0 - eps)
    if mask is None:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
    else:
        A = (Y * mask) / P
        B = ((1.0 - Y) * mask) / (1.0 - P)
    return A, B


def _row_normalize(W: Array, eps: float) -> Array:
    s = W.sum(axis=1, keepdims=True)
    s[s <= 0.0] = 1.0  # safeguard
    return W / s


def _col_normalize(H: Array, eps: float) -> Array:
    s = H.sum(axis=0, keepdims=True)
    s[s <= 0.0] = 1.0  # safeguard
    return H / s


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
    """Eq. (14), Algorithm 1 lines 4–6 (Beta prior on H; ratio C/(C+D))."""
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    C = H * (W.T @ A + (alpha - 1.0))
    D = (1.0 - H) * (W.T @ B + (beta - 1.0))
    return C / (C + D)


def mm_update_W_dir_beta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """Transpose-symmetric of Eq. (14) (Beta prior on W; ratio C/(C+D))."""
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    Cw = W * (A @ H.T + (alpha - 1.0))
    # IMPORTANT: uses B @ H.T (not B @ (1−H).T)
    Dw = (1.0 - W) * (B @ H.T + (beta - 1.0))
    return Cw / (Cw + Dw)


def mm_update_W_simplex_beta_dir(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """Eq. (20) multiplicative part + exact L1 row‑renormalization (simplex)."""
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (A @ H.T) + (B @ (1.0 - H).T)  # (M,K)
    W_tmp = W * numer
    return _row_normalize(W_tmp, eps)


def mm_update_H_simplex_dir_beta(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """Transpose of Eq. (20): multiplicative part + exact L1 col‑renormalization."""
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (W.T @ A) + ((1.0 - W).T @ B)  # (K,N)
    H_tmp = H * numer
    return _col_normalize(H_tmp, eps)


# ---------------------------------------------------------------------
# One full MM step for each orientation (used by tests & estimator)
# ---------------------------------------------------------------------

def mm_step_beta_dir(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array] = None,
    eps: float = 1e-12,
):
    """One paper‑exact MM iteration for orientation='beta-dir'."""
    H = mm_update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
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
):
    """One paper‑exact MM iteration for orientation='dir-beta'."""
    W = mm_update_W_dir_beta(Y, W, H, alpha, beta, mask, eps)
    H = mm_update_H_simplex_dir_beta(Y, W, H, mask, eps)
    return W, H
