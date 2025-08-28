import numpy as np
import time
from typing import Tuple, Optional, List

def nbmf_mm_update_beta_dir(
    Y: np.ndarray,
    W: np.ndarray,  # Shape (k, m), columns sum to 1
    H: np.ndarray,  # Shape (k, n), values in (0, 1)
    mask: Optional[np.ndarray],
    alpha: float,
    beta: float,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One MM iteration for beta-dir orientation.
    W columns sum to 1, H has Beta prior.
    This is the core algorithm from Magron & Févotte (2022).
    """
    m, n = Y.shape
    
    # Precompute masked versions
    if mask is None:
        Y_masked = Y
        Y_T = Y.T
        OneminusY_T = (1 - Y).T
    else:
        # Handle sparse masks
        if hasattr(mask, 'toarray'):
            mask = mask.toarray()
        Y_masked = Y * mask
        Y_T = Y.T * mask.T
        OneminusY_T = (1 - Y).T * mask.T
    
    # Beta prior matrices for H
    A = np.ones_like(H) * (alpha - 1)
    B = np.ones_like(H) * (beta - 1)
    
    # ======================== H UPDATE (continuous in [0,1]) ========================
    WH = W.T @ H  # Shape (m, n)
    
    # Numerator and denominator for H update
    numerator = H * (W @ (Y_masked / (WH + eps))) + A
    denominator = (1 - H) * (W @ ((1 - Y_masked) / (1 - WH + eps))) + B
    
    # Update H - stays continuous in (0, 1)
    H_new = numerator / (numerator + denominator + eps)
    H_new = np.clip(H_new, eps, 1 - eps)
    
    # ======================== W UPDATE (columns sum to 1) ========================
    HW_T = H_new.T @ W  # Shape (n, m)
    
    # Update W with normalization
    W_new = W * (H_new @ (Y_T / (HW_T + eps)) + (1 - H_new) @ (OneminusY_T / (1 - HW_T + eps)))
    W_new = W_new / n  # Critical: divide by n to maintain simplex
    
    # Ensure columns sum to 1 (should be maintained by /n, but ensure numerical stability)
    W_new = W_new / W_new.sum(axis=0, keepdims=True)
    
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
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, List[float], float, int]:
    """
    NBMF-MM solver supporting both orientations.
    
    Parameters
    ----------
    Y : array-like, shape (m, n)
        Binary data matrix
    n_components : int
        Number of components
    orientation : {"beta-dir", "dir-beta"}
        - "beta-dir": W rows sum to 1, H has Beta prior
        - "dir-beta": W has Beta prior, H columns sum to 1
    
    Returns
    -------
    W : array-like, shape (m, k)
        First factor matrix
    H : array-like, shape (k, n)
        Second factor matrix
    losses : list
        Loss values per iteration
    time_elapsed : float
        Total time
    n_iter : int
        Number of iterations
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert sparse mask to dense if needed
    if mask is not None and hasattr(mask, 'toarray'):
        mask = mask.toarray()
    
    m, n = Y.shape
    k = n_components
    
    # Handle orientation by transposing if needed
    if orientation == "dir-beta":
        # Dir-Beta is equivalent to Beta-Dir on Y^T
        Y = Y.T
        m, n = n, m
        if mask is not None:
            if hasattr(mask, 'toarray'):  # sparse matrix
                mask = mask.toarray()
            mask = mask.T
        # Swap init matrices
        if W_init is not None and H_init is not None:
            W_init, H_init = H_init.T, W_init.T
    
    # Initialize
    if W_init is None:
        W_init = np.random.uniform(0.1, 0.9, (m, k))
    if H_init is None:
        H_init = np.random.uniform(0.1, 0.9, (k, n))
    
    # Convert to internal notation (W is k×m, H is k×n)
    W = W_init.T  # Now (k, m)
    H = H_init    # H_init is already (k, n)
    
    # Normalize W columns to sum to 1
    W = W / W.sum(axis=0, keepdims=True)
    
    # Track losses
    losses = []
    loss_prev = np.inf
    
    # Main loop
    for iteration in range(max_iter):
        # MM update
        W, H = nbmf_mm_update_beta_dir(Y, W, H, mask, alpha, beta, eps)
        
        # Compute loss
        WH = W.T @ H
        if mask is None:
            log_lik = Y * np.log(WH + eps) + (1 - Y) * np.log(1 - WH + eps)
            n_obs = Y.size
        else:
            Y_masked = Y * mask
            log_lik = Y_masked * np.log(WH + eps) + (1 - Y_masked) * np.log(1 - WH + eps)
            n_obs = np.count_nonzero(mask)
        
        # Prior term
        A = (alpha - 1) * np.sum(np.log(H + eps))
        B = (beta - 1) * np.sum(np.log(1 - H + eps))
        
        # Total loss
        loss = -(np.sum(log_lik) + A + B) / n_obs
        losses.append(loss)
        
        if verbose > 0 and iteration % 10 == 0:
            print(f"Iter {iteration:4d}: Loss = {loss:.6f}")
        
        # Check convergence
        if iteration > 0:
            rel_change = abs(loss_prev - loss) / abs(loss_prev)
            if rel_change < tol:
                if verbose > 0:
                    print(f"Converged at iteration {iteration}")
                break
        loss_prev = loss
    
    # Convert back to external notation
    W_final = W.T  # Shape (m, k)
    H_final = H   # Shape (k, n)

    # Handle orientation output
    if orientation == "dir-beta":
        # Transpose back for dir-beta
        W_final, H_final = H_final.T, W_final.T

    # ------------------------------------------------------------------
    # Enforce simplex on the appropriate factor within small tolerance.
    # - beta-dir: rows of W should sum to 1
    # - dir-beta: columns of H should sum to 1
    # We renormalize only if deviation exceeds a tiny tolerance.
    # ------------------------------------------------------------------
    eps = 1e-12
    tol = 1e-9

    if orientation == "beta-dir":
        # Ensure W rows sum to 1
        if W_final.size:
            row_sums = W_final.sum(axis=1, keepdims=True)
            dev = np.max(np.abs(row_sums - 1.0)) if row_sums.size else 0.0
            if np.isfinite(dev) and (dev > tol):
                safe = (row_sums > eps)
                # Avoid divide-by-zero; leave degenerate rows untouched
                if np.any(safe):
                    W_final[safe.ravel(), :] = W_final[safe.ravel(), :] / row_sums[safe]
    else:  # dir-beta
        # Ensure H columns sum to 1
        if H_final.size:
            col_sums = H_final.sum(axis=0, keepdims=True)
            dev = np.max(np.abs(col_sums - 1.0)) if col_sums.size else 0.0
            if np.isfinite(dev) and (dev > tol):
                safe = (col_sums > eps)
                if np.any(safe):
                    H_final[:, safe.ravel()] = H_final[:, safe.ravel()] / col_sums[:, safe.ravel()]

    n_iter = iteration + 1
    return W_final, H_final, losses, 0.0, n_iter
