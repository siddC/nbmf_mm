# Copyright (c) 2025
# nbmf_mm: Mean-parameterized Bernoulli NMF via Majorization–Minimization
# Paper-exact MM helpers and objective.
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


# ----------------------------- small utilities ----------------------------- #

def _clip01(X: Array, eps: float = 1e-12) -> Array:
    """Clip to (eps, 1-eps) to avoid log/ratio blow-ups."""
    return np.clip(X, eps, 1.0 - eps)


def _normalize_rows_to_one(X: Array) -> Array:
    """Exact row L1 normalization (sum exactly 1.0)."""
    s = X.sum(axis=1, keepdims=True)
    # Safe divide: rows that are (numerically) zero become uniform
    zero = (s <= 0.0)
    X = X / np.where(s > 0.0, s, 1.0)
    if np.any(zero):
        m, k = X.shape
        X[zero.ravel()] = 1.0 / k
    return X


def _normalize_cols_to_one(X: Array) -> Array:
    """Exact column L1 normalization (sum exactly 1.0)."""
    s = X.sum(axis=0, keepdims=True)
    zero = (s <= 0.0)
    X = X / np.where(s > 0.0, s, 1.0)
    if np.any(zero):
        k, n = X.shape
        X[:, zero.ravel()] = 1.0 / k
    return X


# Exposed to tests for deterministic initializations
def _dirichlet_rows(rng: np.random.Generator, M: int, K: int) -> Array:
    """Draw M rows on the simplex of size K."""
    X = rng.random((M, K))
    return _normalize_rows_to_one(X)


def _dirichlet_cols(rng: np.random.Generator, K: int, N: int) -> Array:
    """Draw N columns on the simplex of size K."""
    X = rng.random((K, N))
    return _normalize_cols_to_one(X)


# ----------------------------- likelihood & priors ----------------------------- #

def bernoulli_nll(Y: Array, WH: Array, mask: Optional[Array],
                  average: bool = False, eps: float = 1e-12) -> float:
    """
    Binary (Bernoulli) negative log-likelihood:
        L(Y|P) = - sum_{m,n} [ y log p + (1-y) log (1-p) ]   (masked if provided)
    """
    P = _clip01(WH, eps)
    if mask is not None:
        Y = Y * mask
        one_minus_Y = (1.0 - Y) * mask
    else:
        one_minus_Y = 1.0 - Y
    nll = - (Y * np.log(P) + one_minus_Y * np.log(1.0 - P)).sum()
    if average:
        denom = mask.sum() if mask is not None else Y.size
        return float(nll / max(denom, 1))
    return float(nll)


def _beta_neglog_prior(X: Array, alpha: float, beta: float, eps: float) -> float:
    """
    Negative log Beta prior (sum over entries; constant terms dropped):
        - (alpha - 1) * sum log X - (beta - 1) * sum log(1 - X)
    """
    Xc = _clip01(X, eps)
    return float(-(alpha - 1.0) * np.log(Xc).sum() - (beta - 1.0) * np.log(1.0 - Xc).sum())


def objective(Y: Array, W: Array, H: Array, mask: Optional[Array],
              orientation: str, alpha: float, beta: float,
              eps: float = 1e-12) -> float:
    """
    MAP objective used in the paper (NLL + negative Beta prior on the unconstrained factor).
    """
    WH = W @ H
    obj = bernoulli_nll(Y, WH, mask, average=False, eps=eps)
    orientation = orientation.lower()
    if orientation == "beta-dir":
        # W rows on simplex; Beta(α,β) prior on H entries
        obj += _beta_neglog_prior(H, alpha, beta, eps)
    elif orientation == "dir-beta":
        # H columns on simplex; Beta(α,β) prior on W entries
        obj += _beta_neglog_prior(W, alpha, beta, eps)
    else:
        raise ValueError('orientation must be "beta-dir" or "dir-beta"')
    return obj


# ----------------------------- MM core ratios ----------------------------- #

def _masked_ratios(Y: Array, WH: Array, mask: Optional[Array], eps: float
                   ) -> Tuple[Array, Array]:
    """
    Compute A = Y / P, B = (1-Y) / (1-P), with optional masking.
    """
    P = _clip01(WH, eps)
    if mask is not None:
        A = (Y * mask) / P
        B = ((1.0 - Y) * mask) / (1.0 - P)
    else:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
    return A, B


# ----------------------------- Paper-exact updates ----------------------------- #
# IMPORTANT: These implement the MM updates exactly as in Magron & Févotte (2022).
# The split into A (successes) and B (failures) must be respected. The simplex
# factor uses exact L1 renormalization ("normalize" projection).


def mm_update_W_beta_dir(Y: Array, W: Array, H: Array,
                         mask: Optional[Array], eps: float) -> Array:
    """
    Simplex-constrained W update (orientation: beta-dir). No Beta prior here.
    W <- W * ((A @ H^T) / (B @ H^T)), then renormalize each row to sum 1.
    """
    A, B = _masked_ratios(Y, W @ H, mask, eps)
    num = A @ H.T
    den = B @ H.T
    W = W * (num / np.maximum(den, eps))
    W = _normalize_rows_to_one(np.maximum(W, eps))
    return W


def mm_update_H_dir_beta(Y: Array, W: Array, H: Array,
                         mask: Optional[Array], eps: float) -> Array:
    """
    Simplex-constrained H update (orientation: dir-beta). No Beta prior here.
    H <- H * ((W^T @ A) / (W^T @ B)), then renormalize each column to sum 1.
    """
    A, B = _masked_ratios(Y, W @ H, mask, eps)
    num = W.T @ A
    den = W.T @ B
    H = H * (num / np.maximum(den, eps))
    H = _normalize_cols_to_one(np.maximum(H, eps))
    return H


def mm_update_H_beta_dir(Y: Array, W: Array, H: Array,
                         alpha: float, beta: float,
                         mask: Optional[Array], eps: float) -> Array:
    """
    Beta-regularized H update (orientation: beta-dir).
    H <- H * ( W^T A + (α-1)/H ) / ( W^T B + (β-1)/(1-H) )
    """
    A, B = _masked_ratios(Y, W @ H, mask, eps)
    Hc = _clip01(H, eps)
    num = (W.T @ A) + (alpha - 1.0) / Hc
    den = (W.T @ B) + (beta - 1.0) / (1.0 - Hc)
    H = Hc * (num / np.maximum(den, eps))
    H = _clip01(H, eps)
    return H


def mm_update_W_dir_beta(Y: Array, W: Array, H: Array,
                         alpha: float, beta: float,
                         mask: Optional[Array], eps: float) -> Array:
    """
    Beta-regularized W update (orientation: dir-beta).
    W <- W * ( A H^T + (α-1)/W ) / ( B H^T + (β-1)/(1-W) )
    """
    A, B = _masked_ratios(Y, W @ H, mask, eps)
    Wc = _clip01(W, eps)
    num = (A @ H.T) + (alpha - 1.0) / Wc
    den = (B @ H.T) + (beta - 1.0) / (1.0 - Wc)
    W = Wc * (num / np.maximum(den, eps))
    W = _clip01(W, eps)
    return W


def mm_step_beta_dir(Y: Array, W: Array, H: Array,
                     alpha: float, beta: float,
                     mask: Optional[Array] = None, eps: float = 1e-12
                     ) -> Tuple[Array, Array]:
    """
    One full MM step for orientation='beta-dir' (Binary ICA):
      1) W: simplex multiplicative step + exact row normalization
      2) H: Beta-regularized multiplicative step
    This ordering matches the helper used in tests; keep it **exact** for parity.
    """
    W = mm_update_W_beta_dir(Y, W, H, mask, eps)
    # Use the updated W when updating H (Gauss–Seidel), as in the paper.
    H = mm_update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
    return W, H


def mm_step_dir_beta(Y: Array, W: Array, H: Array,
                     alpha: float, beta: float,
                     mask: Optional[Array] = None, eps: float = 1e-12
                     ) -> Tuple[Array, Array]:
    """
    One full MM step for orientation='dir-beta' (Aspect Bernoulli):
      1) W: Beta-regularized multiplicative step
      2) H: simplex multiplicative step + exact column normalization
    This ordering matches the helper used in tests; keep it **exact** for parity.
    """
    W = mm_update_W_dir_beta(Y, W, H, alpha, beta, mask, eps)
    H = mm_update_H_dir_beta(Y, W, H, mask, eps)
    return W, H
