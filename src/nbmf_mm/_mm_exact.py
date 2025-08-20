# src/nbmf_mm/_mm_exact.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

Array = np.ndarray


def _clip01(X: Array, eps: float = 1e-12) -> Array:
    """Clip to (eps, 1-eps) to avoid log/ratio singularities."""
    return np.clip(X, eps, 1.0 - eps)


def bernoulli_nll(
    Y: Array,
    P: Array,
    mask: Optional[Array] = None,
    *,
    average: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Negative log-likelihood of a Bernoulli mean model.

    Parameters
    ----------
    Y : array-like, shape (M, N)
        Binary observations (0/1).
    P : array-like, shape (M, N)
        Mean-parameter (probabilities) in (0, 1).
    mask : array-like of {0,1}, shape (M, N), optional (default=None)
        If provided, entries with mask==0 are ignored.
    average : bool, optional (default=False)
        If True, return average per observed entry; else return sum.
    eps : float
        Numerical epsilon for clipping.

    Returns
    -------
    nll : float
        Negative log-likelihood (sum or average).
    """
    P = _clip01(P, eps)
    if mask is None:
        ll = Y * np.log(P) + (1.0 - Y) * np.log(1.0 - P)
        denom = Y.size
    else:
        Wmask = mask.astype(P.dtype, copy=False)
        ll = Wmask * (Y * np.log(P) + (1.0 - Y) * np.log(1.0 - P))
        denom = np.maximum(Wmask.sum(), 1.0)  # avoid div by 0

    nll = -float(ll.sum())
    if average:
        nll /= float(denom)
    return nll


def _masked_ratios(
    Y: Array, P: Array, mask: Optional[Array], eps: float
) -> Tuple[Array, Array]:
    """
    Compute A = (mask*Y)/P and B = (mask*(1-Y))/(1-P) (or without mask).
    Shapes: Y,P,mask -> (M,N); returns (M,N).
    """
    P = _clip01(P, eps)
    if mask is None:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
    else:
        Wmask = mask.astype(P.dtype, copy=False)
        A = Wmask * (Y / P)
        B = Wmask * ((1.0 - Y) / (1.0 - P))
    return A, B


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
    """
    MAP objective = Bernoulli NLL + negative Beta log-prior
    (prior applies only to the Beta-constrained block).
    """
    P = W @ H
    nll = bernoulli_nll(Y, P, mask=mask, average=False, eps=eps)

    if orientation == "beta-dir":
        # Beta prior on H
        Hp = _clip01(H, eps)
        prior = - (alpha - 1.0) * np.log(Hp) - (beta - 1.0) * np.log(1.0 - Hp)
        prior = float(prior.sum())
    elif orientation == "dir-beta":
        # Beta prior on W
        Wp = _clip01(W, eps)
        prior = - (alpha - 1.0) * np.log(Wp) - (beta - 1.0) * np.log(1.0 - Wp)
        prior = float(prior.sum())
    else:
        raise ValueError('orientation must be "beta-dir" or "dir-beta"')

    return nll + prior


# ---- Block updates for one MM step (paper-exact path: "normalize") ---------

def _update_W_rows_simplex(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """
    Dirichlet-like constraint on rows of W (row-simplex).
    Multiplicative update + exact L1 row normalization.
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)  # (M,N)
    num = A @ H.T                           # (M,K)
    den = B @ H.T                           # (M,K)
    W *= num / np.maximum(den, eps)
    W = np.maximum(W, eps)
    W /= np.sum(W, axis=1, keepdims=True)
    return W


def _update_H_cols_simplex(
    Y: Array, W: Array, H: Array, mask: Optional[Array], eps: float
) -> Array:
    """
    Dirichlet-like constraint on columns of H (column-simplex).
    Multiplicative update + exact L1 column normalization.
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)  # (M,N)
    num = W.T @ A                            # (K,N)
    den = W.T @ B                            # (K,N)
    H *= num / np.maximum(den, eps)
    H = np.maximum(H, eps)
    H /= np.sum(H, axis=0, keepdims=True)
    return H


def _update_H_beta_dir(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """
    Beta-constrained H update for orientation 'beta-dir'.
    Multiplicative MM with (alpha-1)/H and (beta-1)/(1-H) terms.
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)  # (M,N)
    num = (W.T @ A) + (alpha - 1.0) / _clip01(H, eps)
    den = (W.T @ B) + (beta - 1.0) / _clip01(1.0 - H, eps)
    H *= num / np.maximum(den, eps)
    H = _clip01(H, eps)
    return H


def _update_W_beta_dirbeta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array],
    eps: float,
) -> Array:
    """
    Beta-constrained W update for orientation 'dir-beta'.
    Multiplicative MM with (alpha-1)/W and (beta-1)/(1-W) terms.
    """
    P = W @ H
    A, B = _masked_ratios(Y, P, mask, eps)  # (M,N)
    num = (A @ H.T) + (alpha - 1.0) / _clip01(W, eps)
    den = (B @ H.T) + (beta - 1.0) / _clip01(1.0 - W, eps)
    W *= num / np.maximum(den, eps)
    W = _clip01(W, eps)
    return W


# Public one-step helpers used in tests
def mm_step_beta_dir(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array] = None,
    *,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    """
    One MM step for orientation 'beta-dir':
      - Update H with Beta prior (multiplicative)
      - Update W by multiplicative + row L1-normalization
    """
    H = _update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
    W = _update_W_rows_simplex(Y, W, H, mask, eps)
    return W, H


def mm_step_dir_beta(
    Y: Array,
    W: Array,
    H: Array,
    alpha: float,
    beta: float,
    mask: Optional[Array] = None,
    *,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    """
    One MM step for orientation 'dir-beta':
      - Update W with Beta prior (multiplicative)
      - Update H by multiplicative + column L1-normalization
    """
    W = _update_W_beta_dirbeta(Y, W, H, alpha, beta, mask, eps)
    H = _update_H_cols_simplex(Y, W, H, mask, eps)
    return W, H
