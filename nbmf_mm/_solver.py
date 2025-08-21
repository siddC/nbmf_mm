import numpy as np
import time
from typing import Tuple, Optional, List

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
    eps: float = 1e-8,
    orientation: str = "beta-dir"
) -> Tuple[np.ndarray, np.ndarray, List[float], float, int]:
    """
    NBMF-MM solver implementation following Magron & Févotte (2022).
    
    CRITICAL: One factor is continuous in [0,1], the other has simplex constraints!
    
    Parameters
    ----------
    Y : array-like, shape (m, n)
        Binary data matrix {0, 1}
    n_components : int
        Number of latent components k
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance on relative loss change
    alpha, beta : float
        Beta prior parameters for the continuous factor
    W_init, H_init : array-like, optional
        Initial matrices
    mask : array-like, optional
        Binary mask for observed entries
    random_state : int, optional
        Random seed
    verbose : int
        Verbosity level
    eps : float
        Small constant for numerical stability
    orientation : str, default="beta-dir"
        - "beta-dir": H continuous in [0,1], W rows sum to 1 (simplex)
        - "dir-beta": W continuous in [0,1], H columns sum to 1 (simplex)
                     (implemented via transpose symmetry)
        
    Returns
    -------
    W : array-like, shape (m, k)
        Factor matrix - continuous or simplex depending on orientation
    H : array-like, shape (k, n)
        Factor matrix - continuous or simplex depending on orientation
    losses : list
        Loss values per iteration
    time_elapsed : float
        Total computation time
    n_iter : int
        Number of iterations run
    """
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
        
    # Get dimensions
    m, n = Y.shape
    k = n_components
    
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones_like(Y)
        
    # Initialize factors
    if W_init is None:
        W_init = np.random.uniform(0.1, 0.9, (m, k))
    if H_init is None:
        H_init = np.random.uniform(0.1, 0.9, (k, n))
    
    # Initialize tracking
    start_time = time.time()
    
    if orientation.lower() == "dir-beta":
        # =====================================================================
        # DIR-BETA via TRANSPOSE SYMMETRY
        # Transpose input, call beta-dir solver, then transpose results back
        # =====================================================================
        
        # Transpose the input data and mask
        Y_transposed = Y.T  # Now (n, m)
        mask_transposed = mask.T if mask is not None else None
        
        # Transpose and swap the initial matrices for the transposed problem  
        # For dir-beta: we want W continuous, H simplex columns
        # After transpose: we want H_transposed continuous, W_transposed simplex rows
        W_init_transposed = H_init.T if H_init is not None else None  # (n, k)
        H_init_transposed = W_init.T if W_init is not None else None  # (k, m)
        
        # Call beta-dir solver on transposed problem
        W_transposed, H_transposed, losses, time_elapsed_inner, n_iter = nbmf_mm_solver(
            Y=Y_transposed,
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            beta=beta,
            W_init=W_init_transposed,
            H_init=H_init_transposed,
            mask=mask_transposed,
            random_state=random_state,
            verbose=verbose,
            eps=eps,
            orientation="beta-dir"  # Always use beta-dir for the actual computation
        )
        
        # Transpose results back to get the correct dir-beta solution
        # W_transposed is (n, k) with rows summing to 1
        # H_transposed is (k, m) with continuous values
        # We want: W (m, k) continuous, H (k, n) with columns summing to 1
        W_final = H_transposed.T  # (m, k) - will be continuous
        H_final = W_transposed.T  # (k, n) - will have columns summing to 1
        
        time_elapsed = time_elapsed_inner
        
    else:
        # =====================================================================
        # BETA-DIR ORIENTATION: H continuous [0,1], W rows sum to 1 (simplex)
        # =====================================================================
        
        # CRITICAL: Transpose to match Magron's notation
        # In Magron's code: W is (k, m) and H is (k, n)
        W = W_init.T  # Now (k, m)
        H = H_init    # H_init is already (k, n)
        
        # Normalize W columns to sum to 1 (simplex constraint)
        W = W / W.sum(axis=0, keepdims=True)
        
        # Precompute masked versions for efficiency
        Y_masked = Y * mask
        Y_T = Y.T * mask.T  # Transposed masked Y
        OneminusY_T = (1 - Y.T) * mask.T  # Transposed masked (1-Y)
        
        # Beta prior matrices for H
        A = np.ones_like(H) * (alpha - 1)
        B = np.ones_like(H) * (beta - 1)
        
        # Initialize tracking
        losses = []
        loss_prev = np.inf
        
        # Main optimization loop
        for iteration in range(max_iter):
            # ======================== H UPDATE ========================
            # CRITICAL: H is NOT forced to binary! H stays continuous in (0, 1)
            
            # Compute W^T @ H
            WH = W.T @ H  # Shape (m, n)
            
            # Numerator: H * W @ (Y / (WH + eps)) + A
            numerator = H * (W @ (Y_masked / (WH + eps))) + A
            
            # Denominator: (1 - H) * W @ (((1 - Y) / (1 - WH + eps))) + B
            denominator = (1 - H) * (W @ ((1 - Y_masked) / (1 - WH + eps))) + B
            
            # Update H (stays in (0, 1) naturally)
            H = numerator / (numerator + denominator + eps)
            
            # Clip to avoid numerical issues at boundaries
            H = np.clip(H, eps, 1 - eps)
            
            # ======================== W UPDATE ========================
            # W columns must stay on simplex
            
            # Compute H @ W^T (for the transpose computation)
            HW_T = H.T @ W  # Shape (n, m)
            
            # Update W with normalization by n (maintains simplex)
            W_numerator = W * (H @ (Y_T / (HW_T + eps)) + (1 - H) @ (OneminusY_T / (1 - HW_T + eps)))
            W = W_numerator / n
            
            # Ensure W stays normalized (columns sum to 1)
            W = W / W.sum(axis=0, keepdims=True)
            
            # ======================== COMPUTE LOSS ========================
            WH = W.T @ H  # Recompute after updates
            
            # Log-likelihood term
            log_lik = Y_masked * np.log(WH + eps) + (1 - Y_masked) * np.log(1 - WH + eps)
            
            # Prior term for H (Beta prior)
            prior = A * np.log(H + eps) + B * np.log(1 - H + eps)
            
            # Total loss (negative log posterior)
            loss = -(np.sum(log_lik) + np.sum(prior)) / np.count_nonzero(mask)
            losses.append(loss)
            
            if verbose > 0 and iteration % 10 == 0:
                print(f"Iteration {iteration:4d}, Loss: {loss:.6f}")
                
            # Check convergence
            if iteration > 0:
                rel_change = abs(loss_prev - loss) / abs(loss_prev)
                if rel_change < tol:
                    if verbose > 0:
                        print(f"Converged at iteration {iteration} (rel_change: {rel_change:.2e})")
                    break
            loss_prev = loss
        
        # Transpose back to our convention
        W_final = W.T  # Shape (m, k)
        H_final = H    # Shape (k, n)
        
        time_elapsed = time.time() - start_time
        n_iter = iteration + 1
    
    return W_final, H_final, losses, time_elapsed, n_iter