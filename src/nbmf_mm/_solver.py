# src/nbmf_mm/_solver.py
from typing import Tuple, Optional, List
import time
import numpy as np


def _mask_views(
    Y: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return masked positives/negatives and their transposes.

    Y_obs = Y * mask
    Z_obs = (1 - Y) * mask

    If mask is None, use full data.
    """
    if mask is not None and hasattr(mask, "toarray"):
        mask = mask.toarray()

    if mask is None:
        Y_obs = Y
        Z_obs = 1.0 - Y
        Y_T = Y.T
        Z_T = Z_obs.T
    else:
        Y_obs = Y * mask
        Z_obs = (1.0 - Y) * mask
        Y_T = Y_obs.T
        Z_T = Z_obs.T

    return Y_obs, Z_obs, Y_T, Z_T


def nbmf_mm_update_beta_dir(
    Y: np.ndarray,
    W: np.ndarray,  # shape (k, m); columns sum to 1 (Dirichlet on rows of external W)
    H: np.ndarray,  # shape (k, n); entries in (0,1) (Beta prior)
    mask: Optional[np.ndarray],
    alpha: float,
    beta: float,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One MM iteration for the Beta-Dir (binary-ICA) orientation:
      - W columns (internal) sum to 1  <=> rows of external W on the simplex
      - H is Beta-distributed in (0,1)

    Masked extension of Magron & Févotte (2022), Alg. 1:
      - Replace Y by Y_obs = Y * mask and (1-Y) by Z_obs = (1-Y) * mask
      - Preserve the simplex with per-column Lagrange normalization (λ)
    """
    # Observed positives/negatives and their transposes
    Y_obs, Z_obs, Y_T, Z_T = _mask_views(Y, mask)

    # Prior parameters (broadcasted)
    A = (alpha - 1.0)
    B = (beta - 1.0)

    # ======================== H update (elementwise in (0,1)) ========================
    WH = W.T @ H  # (m, n)

    num_H = H * (W @ (Y_obs / (WH + eps))) + A
    den_H = (1.0 - H) * (W @ (Z_obs / (1.0 - WH + eps))) + B

    H_new = num_H / (num_H + den_H + eps)
    H_new = np.clip(H_new, eps, 1.0 - eps)

    # ======================== W update (columns sum to 1) ============================
    # Multiplicative pre-update
    HW_T = H_new.T @ W  # (n, m)
    F = H_new @ (Y_T / (HW_T + eps)) + (1.0 - H_new) @ (Z_T / (1.0 - HW_T + eps))

    W_raw = W * F

    # Lagrange normalization: λ_m = sum_k W_raw[k, m]
    lam = W_raw.sum(axis=0, keepdims=True)
    W_new = W_raw / (lam + eps)  # => columns sum to 1 exactly (up to eps)

    return W_new, H_new


def nbmf_mm_solver(
    Y: np.ndarray,
    n_components: int,
    max_iter: int = 500,
    tol: float = 1e-5,
    alpha: float = 1.2,
    beta: float = 1.2,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
    orientation: str = "beta-dir",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, List[float], float, int]:
    """
    NBMF-MM solver supporting both orientations.

    Parameters
    ----------
    Y : array-like, shape (m, n)
        Binary (or [0,1]) data matrix.
    n_components : int
        Number of components (latent dimension k).
    orientation : {"beta-dir", "dir-beta"}
        - "beta-dir": W rows simplex, H in (0,1)  (Binary ICA)
        - "dir-beta": H columns simplex, W in (0,1) (Aspect Bernoulli)

    Returns
    -------
    W : array-like, shape (m, k)
    H : array-like, shape (k, n)
    losses : list[float]
    time_elapsed : float
    n_iter : int
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Dense mask if needed
    if mask is not None and hasattr(mask, "toarray"):
        mask = mask.toarray()

    m, n = Y.shape
    k = n_components

    # Handle orientation by transposing if needed.
    # Internally we always solve the Beta-Dir case (simplex on internal W).
    transposed = (orientation == "dir-beta")
    if transposed:
        Y = Y.T
        m, n = n, m
        if mask is not None:
            mask = mask.T
        if W_init is not None and H_init is not None:
            W_init, H_init = H_init.T, W_init.T

    # Initialization
    if W_init is None:
        W_init = np.random.uniform(0.1, 0.9, (m, k))
    if H_init is None:
        H_init = np.random.uniform(0.1, 0.9, (k, n))

    # Convert to internal shapes: W is (k, m) with column-simplex; H is (k, n)
    W = W_init.T
    H = H_init.copy()
    W = W / (W.sum(axis=0, keepdims=True) + eps)

    # Main loop
    t0 = time.time()
    losses: List[float] = []
    loss_prev = np.inf

    for it in range(max_iter):
        # One MM step
        W, H = nbmf_mm_update_beta_dir(Y, W, H, mask, alpha, beta, eps)

        # ---- Compute masked log-likelihood + priors (for monitoring) ----
        WH = W.T @ H

        if mask is None:
            Y_obs = Y
            Z_obs = 1.0 - Y
            n_obs = Y.size
        else:
            Y_obs = Y * mask
            Z_obs = (1.0 - Y) * mask
            n_obs = int(mask.sum())

        log_lik = Y_obs * np.log(WH + eps) + Z_obs * np.log(1.0 - WH + eps)
        prior = (alpha - 1.0) * np.sum(np.log(H + eps)) + (beta - 1.0) * np.sum(np.log(1.0 - H + eps))

        loss = - (np.sum(log_lik) + prior) / max(n_obs, 1)
        losses.append(float(loss))

        if verbose and (it % 10 == 0):
            print(f"Iter {it:4d}: loss={loss:.6f}")

        if np.isfinite(loss_prev):
            rel = abs(loss_prev - loss) / (abs(loss_prev) + eps)
            if rel < tol:
                break
        loss_prev = loss

    time_elapsed = time.time() - t0
    n_iter = it + 1

    # Back to external shapes: W_ext is (m,k), H_ext is (k,n)
    W_ext = W.T
    H_ext = H

    if transposed:
        # dir-beta: swap back
        W_ext, H_ext = H_ext.T, W_ext.T  # W continuous, H columns simplex externally

    # Final tiny safeguard (should already be exact to ~1e-12)
    if orientation == "beta-dir":
        # rows of W sum to 1
        s = W_ext.sum(axis=1, keepdims=True)
        W_ext = W_ext / (s + eps)
    else:
        # columns of H sum to 1
        s = H_ext.sum(axis=0, keepdims=True)
        H_ext = H_ext / (s + eps)

    return W_ext, H_ext, losses, time_elapsed, n_iter
