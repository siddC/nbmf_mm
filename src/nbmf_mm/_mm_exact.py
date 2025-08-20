"""
Exact MM (paper-exact) update helpers for mean-parameterized Bernoulli NBMF.

Implements the updates from Magron & Févotte (2022), Algorithm 1, with
two symmetric orientations:

- "beta-dir": rows of W lie on the probability simplex (sum to 1), Beta prior on H.
- "dir-beta": columns of H lie on the probability simplex (sum to 1), Beta prior on W.

Masking is supported by replacing the global denominators (/N or /M) by
per-row or per-column observed counts, respectively, which preserves the
simplex constraints under partial observation.
"""
from __future__ import annotations
import numpy as np

Array = np.ndarray


def _clip01(X: Array, eps: float = 1e-12) -> Array:
    return np.clip(X, a_min=eps, a_max=1.0 - eps)


def _masked_ratios(Y: Array, WH: Array, mask: Array | None, eps: float):
    P = _clip01(WH, eps)
    if mask is None:
        return Y / P, (1.0 - Y) / (1.0 - P)
    A = np.zeros_like(Y)
    B = np.zeros_like(Y)
    nz = mask.astype(bool)
    A[nz] = Y[nz] / P[nz]
    B[nz] = (1.0 - Y[nz]) / (1.0 - P[nz])
    return A, B


def bernoulli_nll(Y: Array, WH: Array, mask: Array | None = None, average: bool = False, eps: float = 1e-12) -> float:
    P = _clip01(WH, eps)
    if mask is None:
        nll = -(np.sum(Y * np.log(P)) + np.sum((1.0 - Y) * np.log1p(-P)))
        denom = Y.size
    else:
        nz = mask.astype(bool)
        nll = -(np.sum(Y[nz] * np.log(P[nz])) + np.sum((1.0 - Y[nz]) * np.log1p(-P[nz])))
        denom = float(np.sum(mask))
    return float(nll / denom if average else nll)


def beta_log_prior(Z: Array, alpha: float, beta: float, eps: float = 1e-12) -> float:
    Z = _clip01(Z, eps)
    return float(np.sum((alpha - 1.0) * np.log(Z) + (beta - 1.0) * np.log1p(-Z)))


def objective(Y: Array, W: Array, H: Array, mask: Array | None, orientation: str, alpha: float, beta: float, eps: float = 1e-12) -> float:
    WH = W @ H
    nll = bernoulli_nll(Y, WH, mask, average=False, eps=eps)
    if orientation == "beta-dir":
        return float(nll - beta_log_prior(H, alpha, beta, eps))
    elif orientation == "dir-beta":
        return float(nll - beta_log_prior(W, alpha, beta, eps))
    else:
        raise ValueError("orientation must be 'beta-dir' or 'dir-beta'")


# -------- beta-dir
def mm_update_H_beta_dir(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> Array:
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    # ADD pseudo-counts
    C = H * (W.T @ A) + (alpha - 1.0)
    D = (1.0 - H) * (W.T @ B) + (beta - 1.0)
    return _clip01(C / (C + D + eps), eps)


def mm_update_W_beta_dir(Y: Array, W: Array, H: Array, mask: Array | None = None, eps: float = 1e-12) -> Array:
    M, N = Y.shape
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    numer = A @ H.T + B @ (1.0 - H).T  # MxK
    if mask is None:
        return W * (numer / float(N))
    row_counts = np.asarray(mask.sum(axis=1), dtype=float).reshape(-1, 1)
    Wn = W.copy()
    rows = (row_counts.squeeze() > 0)
    if np.any(rows):
        Wn[rows] = W[rows] * (numer[rows] / row_counts[rows])
    return Wn


def mm_step_beta_dir(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12):
    H = mm_update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
    W = mm_update_W_beta_dir(Y, W, H, mask, eps)
    return W, H


# -------- dir-beta
def mm_update_W_dir_beta(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> Array:
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    # ADD pseudo-counts
    C = W * (A @ H.T) + (alpha - 1.0)
    D = (1.0 - W) * (B @ (1.0 - H).T) + (beta - 1.0)
    return _clip01(C / (C + D + eps), eps)


def mm_update_H_dir_beta(Y: Array, W: Array, H: Array, mask: Array | None = None, eps: float = 1e-12) -> Array:
    M, N = Y.shape
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    numer = W.T @ A + (1.0 - W).T @ B  # KxN
    if mask is None:
        return H * (numer / float(M))
    col_counts = np.asarray(mask.sum(axis=0), dtype=float).reshape(1, -1)
    Hn = H.copy()
    cols = (col_counts.squeeze() > 0)
    if np.any(cols):
        Hn[:, cols] = H[:, cols] * (numer[:, cols] / col_counts[:, cols])
    return Hn


def mm_step_dir_beta(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12):
    W = mm_update_W_dir_beta(Y, W, H, alpha, beta, mask, eps)
    H = mm_update_H_dir_beta(Y, W, H, mask, eps)
    return W, H
