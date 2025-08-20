"""
Exact MM (paper-exact) update helpers for mean-parameterized Bernoulli NBMF.

Implements the updates from Magron & Févotte (2022), Algorithm 1, with
the two symmetric orientations used in `nbmf_mm`:

- orientation="beta-dir": rows of W lie on the probability simplex (sum to 1),
  Beta prior on H.  (This matches the paper's main orientation.)
- orientation="dir-beta": columns of H lie on the probability simplex (sum to 1),
  Beta prior on W.  (Swapped / symmetric orientation.)

These helpers are intentionally minimal, NumPy-only, and "paper-exact":
  * The simplex-constrained factor is updated with the *closed-form normalizer*
    (/N or /M), not with a projection.
  * The Beta-constrained factor uses the (C/(C+D)) ratio update.

They are suitable for unit tests and reference parity checks.  They are *not*
optimized or fused; production code should use your library's vectorized
kernels, numexpr, numba, etc.

References:
  P. Magron and C. Févotte (2022). "A majorization–minimization algorithm
  for nonnegative binary matrix factorization".
"""
from __future__ import annotations
import numpy as np

Array = np.ndarray

def _clip01(X: Array, eps: float = 1e-12) -> Array:
    return np.clip(X, eps, 1.0 - eps, out=np.empty_like(X), a_min=eps, a_max=1.0 - eps)

def _apply_mask(Y: Array, mask: Array | None) -> tuple[Array, Array, float]:
    """
    Returns (Y_masked, one_minus_Y_masked, denom).
    denom is the number of observed entries if mask is provided, else Y.size.
    """
    if mask is None:
        return Y, (1.0 - Y), float(Y.size)
    Y_mask = Y * mask
    return Y_mask, (1.0 - Y) * mask, float(np.sum(mask))

def bernoulli_nll(Y: Array, WH: Array, mask: Array | None = None, average: bool = False, eps: float = 1e-12) -> float:
    """
    Bernoulli negative log-likelihood: -sum_{mn} [ y log p + (1-y) log (1-p) ].
    Uses natural logs. If mask is provided, only masked entries contribute.
    """
    P = _clip01(WH, eps)
    Y_m, one_m_Y, denom = _apply_mask(Y, mask)
    nll = -(np.sum(Y_m * np.log(P)) + np.sum(one_m_Y * np.log1p(-P)))
    return nll / denom if average else nll

def beta_log_prior(Z: Array, alpha: float, beta: float) -> float:
    """
    Sum of elementwise log Beta(alpha, beta) prior terms (up to constant):
      sum [ (alpha-1) log z + (beta-1) log (1-z) ]
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0 for Beta prior.")
    Z = _clip01(Z)
    return np.sum((alpha - 1.0) * np.log(Z) + (beta - 1.0) * np.log1p(-Z))

def objective(Y: Array, W: Array, H: Array, mask: Array | None, orientation: str, alpha: float, beta: float, eps: float = 1e-12) -> float:
    """
    Negative log-posterior (up to additive constants):
      NLL(Y | W H)  -  log_prior(Beta-constrained factor).
    """
    WH = W @ H
    obj = bernoulli_nll(Y, WH, mask, average=False, eps=eps)
    if orientation == "beta-dir":
        # Beta prior on H
        obj -= beta_log_prior(H, alpha, beta)
    elif orientation == "dir-beta":
        # Beta prior on W
        obj -= beta_log_prior(W, alpha, beta)
    else:
        raise ValueError("orientation must be 'beta-dir' or 'dir-beta'")
    return float(obj)

def _masked_ratios(Y: Array, WH: Array, mask: Array | None, eps: float) -> tuple[Array, Array]:
    """
    Returns A = (Y / WH) and B = ((1-Y) / (1-WH)), both masked (zeros where mask==0).
    """
    P = _clip01(WH, eps)
    if mask is None:
        A = Y / P
        B = (1.0 - Y) / (1.0 - P)
    else:
        A = np.zeros_like(Y)
        B = np.zeros_like(Y)
        nz = mask.astype(bool)
        A[nz] = Y[nz] / P[nz]
        B[nz] = (1.0 - Y[nz]) / (1.0 - P[nz])
    return A, B

# -----------------------------
# beta-dir (paper orientation)
# -----------------------------
def mm_update_H_beta_dir(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> Array:
    """
    Paper Eq. (14)-(16): H <- C / (C + D)
      C = H ⊙ ( W^T (Y/WH) + alpha - 1 )
      D = (1-H) ⊙ ( W^T ((1-Y)/(1-WH)) + beta - 1 )
    """
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    C = H * (W.T @ A + (alpha - 1.0))
    D = (1.0 - H) * (W.T @ B + (beta - 1.0))
    H_new = C / (C + D + eps)
    return _clip01(H_new, eps)

def mm_update_W_beta_dir(Y: Array, W: Array, H: Array, mask: Array | None = None, eps: float = 1e-12) -> Array:
    """
    Paper Eq. (20), multiplicative with /N normalizer (N = number of columns of Y):
      W <- W ⊙ [ (Y/WH) H^T + ((1-Y)/(1-WH)) (1-H)^T ] / N
    This preserves the per-row simplex (row sums stay exactly 1 in exact arithmetic).
    """
    M, N = Y.shape
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)  # MxN
    numer = A @ H.T + B @ (1.0 - H).T       # MxK
    W_new = W * (numer / float(N))
    # Numerical guard: renormalize the rows to sum to 1 exactly.
    row_sums = np.sum(W_new, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0  # avoid division by zero in pathological cases
    W_new = W_new / row_sums
    return _clip01(W_new, eps)

def mm_step_beta_dir(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> tuple[Array, Array]:
    """
    One full MM iteration for the beta-dir orientation: update H then W.
    """
    H = mm_update_H_beta_dir(Y, W, H, alpha, beta, mask, eps)
    W = mm_update_W_beta_dir(Y, W, H, mask, eps)
    return W, H

# -----------------------------
# dir-beta (swapped orientation)
# -----------------------------
def mm_update_W_dir_beta(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> Array:
    """
    Swapped ratio update for W (Beta prior on W):
      W <- C / (C + D)
      C = W ⊙ ( (Y/WH) H^T + alpha - 1 )
      D = (1-W) ⊙ ( ((1-Y)/(1-WH)) (1-H)^T + beta - 1 )
    """
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    C = W * (A @ H.T + (alpha - 1.0))
    D = (1.0 - W) * (B @ (1.0 - H).T + (beta - 1.0))
    W_new = C / (C + D + eps)
    return _clip01(W_new, eps)

def mm_update_H_dir_beta(Y: Array, W: Array, H: Array, mask: Array | None = None, eps: float = 1e-12) -> Array:
    """
    Swapped multiplicative update for H with /M normalizer (M = number of rows of Y):
      H <- H ⊙ [ W^T (Y/WH) + (1-W)^T ((1-Y)/(1-WH)) ] / M
    This preserves the per-column simplex of H (column sums == 1).
    """
    M, N = Y.shape
    WH = W @ H
    A, B = _masked_ratios(Y, WH, mask, eps)
    numer = W.T @ A + (1.0 - W).T @ B     # KxN
    H_new = H * (numer / float(M))
    # Numerical guard: renormalize columns to sum to 1 exactly.
    col_sums = np.sum(H_new, axis=0, keepdims=True)  # shape 1xN
    col_sums[col_sums == 0.0] = 1.0
    H_new = H_new / col_sums
    return _clip01(H_new, eps)

def mm_step_dir_beta(Y: Array, W: Array, H: Array, alpha: float, beta: float, mask: Array | None = None, eps: float = 1e-12) -> tuple[Array, Array]:
    """
    One full MM iteration for the dir-beta orientation: update W then H.
    """
    W = mm_update_W_dir_beta(Y, W, H, alpha, beta, mask, eps)
    H = mm_update_H_dir_beta(Y, W, H, mask, eps)
    return W, H
