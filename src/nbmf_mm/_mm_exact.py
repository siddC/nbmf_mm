# SPDX-License-Identifier: BSD-3-Clause
"""
Exact MM helpers for mean-parameterized Bernoulli NMF (NBMF‑MM).

This module implements the MM updates from:

  • P. Magron and C. Févotte (2022),
    “A majorization–minimization algorithm for nonnegative binary matrix
     factorization”, IEEE Signal Processing Letters. (arXiv:2204.09741)

We follow Algorithm 1 (and its transpose‑symmetric counterpart) with:

Y ~ Bernoulli(P),  P = W @ H,   P ∈ (0,1)^{M×N}.

Orientations
------------
- "beta-dir"  (Binary ICA):
    • W: rows on simplex (Σ_k W_{mk} = 1, W ≥ 0)
    • H: entries in (0,1) with Beta(α,β) prior
    • Step order: update H (Beta‑regularized) then W (simplex)

- "dir-beta"  (Aspect Bernoulli):
    • H: columns on simplex (Σ_k H_{kn} = 1, H ≥ 0)
    • W: entries in (0,1) with Beta(α,β) prior
    • Step order: update W (Beta‑regularized) then H (simplex)

Masking
-------
If mask ∈ {0,1}^{M×N} is supplied, sums are restricted to observed entries
(via elementwise multiplication inside A and B). The simplex steps are
multiplicative followed by exact L1 renormalization (no clipping), which
preserves the simplex exactly even with missingness.

Implementation notes
--------------------
• We never clip W/H during the updates (to preserve exact simplex sums and the
  MM identities). We only clip probabilities when evaluating the NLL.
• Beta‑regularized updates use the *logistic* closed form whose denominator is
  independent of the current iterate; this is the form that ensures monotonic
  decrease of the MAP objective under the “normalize” path.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _clip01(X: Array, eps: float = 1e-12) -> Array:
    """Clip into (eps, 1−eps) with positional args (no kwarg duplication)."""
    return np.clip(X, eps, 1.0 - eps, out=np.empty_like(X))


def bernoulli_nll(
    Y: Array,
    P: Array,
    mask: Optional[Array] = None,
    *,
    average: bool = False,
    eps: float = 1e-12,
) -> float:
    """Negative log-likelihood: −∑_{mn} [ y log p + (1−y) log(1−p) ] (masked if given)."""
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
    """Unnormalized log Beta prior: ∑[(α−1)log Z + (β−1)log(1−Z)]."""
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
    """MAP objective (negative log posterior up to constants)."""
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
    """
    A = Y/P,  B = (1−Y)/(1−P), masked if provided.
    Denominators are safely clipped to (eps, 1−eps).
    """
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
    s[s <= 0.0] = 1.0
    return W / s


def _col_normalize(H: Array, eps: float) -> Array:
    s = H.sum(axis=0, keepdims=True)
    s[s <= 0.0] = 1.0
    return H / s


# ---------------------------------------------------------------------
# Paper‑faithful MM updates
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
    """
    Beta‑regularized update for H in orientation='beta-dir'.

    Logistic closed form (denominator independent of H):
        S1 = W^T A
        S2 = W^T (A + B)
        H_new = [ H ⊙ S1 + (α − 1) ] / [ S2 + (α + β − 2) ]
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    S1 = W.T @ A                       # (K,N)
    S2 = W.T @ (A + B)                 # (K,N)
    H_new = (H * S1 + (alpha - 1.0)) / (S2 + (alpha + beta - 2.0))
    return H_new


def mm_update_W_dir_beta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """
    Beta‑regularized update for W in orientation='dir-beta'.

    Transpose‑symmetric logistic form:
        T1 = A H^T + B (1 − H)^T
        W_new = [ W ⊙ T_num + (α − 1) ] / [ T1 + (α + β − 2) ]
    with   T_num = A H^T
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    T_num = A @ H.T                    # (M,K)
    T_den = T_num + (B @ (1.0 - H).T)  # (M,K)  == (A H^T + B (1−H)^T)
    W_new = (W * T_num + (alpha - 1.0)) / (T_den + (alpha + beta - 2.0))
    return W_new


def mm_update_W_simplex_beta_dir(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """
    Simplex step for W (orientation='beta-dir'):
        W_tmp = W ⊙ [ A H^T + B (1 − H)^T ]
        W_new = row_normalize(W_tmp)
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (A @ H.T) + (B @ (1.0 - H).T)   # (M,K)
    W_tmp = W * numer
    return _row_normalize(W_tmp, eps)


def mm_update_H_simplex_dir_beta(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """
    Simplex step for H (orientation='dir-beta'):
        H_tmp = H ⊙ [ W^T A + (1 − W)^T B ]
        H_new = column_normalize(H_tmp)
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)
    numer = (W.T @ A) + ((1.0 - W).T @ B)   # (K,N)
    H_tmp = H * numer
    return _col_normalize(H_tmp, eps)


# ---------------------------------------------------------------------
# One full MM step per orientation
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
