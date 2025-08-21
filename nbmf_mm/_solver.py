import numpy as np
try:
    import scipy.sparse as sp
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False

def _is_sparse(X):
    """Check if X is a sparse matrix."""
    return HAS_SPARSE and sp.issparse(X)

def _to_dense(X):
    """Convert sparse matrix to dense."""
    if _is_sparse(X):
        return X.toarray()
    return X

def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_nll(V, W, H, alpha=1.0, beta=1.0):
    """
    Compute negative log-likelihood for NBMF-MM.
    
    Parameters
    ----------
    V : array-like, shape (m, n)
        Binary data matrix
    W : array-like, shape (m, k)
        Non-negative factor matrix
    H : array-like, shape (k, n)
        Binary factor matrix
    alpha, beta : float
        Beta prior parameters for W
    
    Returns
    -------
    nll : float
        Negative log-likelihood
    """
    V_dense = _to_dense(V)  # Convert to dense for operations
    m, n = V_dense.shape
    k = W.shape[1]
    
    # Likelihood term
    WH = W @ H
    P = sigmoid(WH)
    eps = 1e-10
    likelihood = -np.sum(V_dense * np.log(P + eps) + (1 - V_dense) * np.log(1 - P + eps))
    
    # Prior term (Beta prior on W)
    prior = -np.sum((alpha - 1) * np.log(W + eps) + (beta - 1) * np.log(1 - W + eps))
    
    return likelihood + prior

def mm_update(V, W, H, alpha=1.0, beta=1.0, orientation="beta-dir"):
    """
    One iteration of MM algorithm for NBMF-MM.
    
    This implements equations (10) and (11) from Magron & Févotte (2022).
    
    Parameters
    ----------
    V : array-like, shape (m, n)
        Binary data matrix
    W : array-like, shape (m, k)
        Non-negative factor matrix
    H : array-like, shape (k, n)
        Binary factor matrix
    alpha, beta : float
        Beta prior parameters for W
    orientation : str
        Either "beta-dir" (H has Beta prior/binary, W has Dirichlet prior/simplex) or 
        "dir-beta" (H has Dirichlet prior/simplex, W has Beta prior/binary)
    
    Returns
    -------
    W_new : array-like, shape (m, k)
        Updated W
    H_new : array-like, shape (k, n)
        Updated H
    """
    V_dense = _to_dense(V)  # Convert to dense for operations
    m, n = V_dense.shape
    k = W.shape[1]
    eps = 1e-10
    
    # Step 1: Update H - depends on orientation
    if orientation == "beta-dir":
        # H has Beta prior (binary constraint from original paper)
        gradient_H = W.T @ (2 * V_dense - 1)  # shape (k, n)
        H_new = (gradient_H > 0).astype(float)
    else:  # dir-beta
        # H has Dirichlet prior (simplex constraint - columns sum to 1)
        # Use MM multiplicative update for simplex-constrained factors
        numerator = W.T @ V_dense  # shape (k, n)
        denominator = W.T @ sigmoid(W @ H)  # shape (k, n)
        H_new = H * (numerator / np.maximum(denominator, eps))
        
        # Project columns to simplex (normalize to sum to 1)
        for j in range(n):
            col_sum = H_new[:, j].sum()
            if col_sum <= eps:
                # If all zeros, initialize uniformly
                H_new[:, j] = 1.0 / k
            else:
                # Normalize to sum to 1
                H_new[:, j] = H_new[:, j] / col_sum
    
    # Step 2: Update W - depends on orientation
    if orientation == "beta-dir":
        # W has Dirichlet prior (simplex constraint - rows sum to 1)
        numerator = V_dense @ H_new.T + (alpha - 1)
        denominator = sigmoid(W @ H_new) @ H_new.T + (beta - 1)
        W_new = W * (numerator / np.maximum(denominator, eps))
        
        # Project rows to simplex (sum to 1) 
        for i in range(m):
            row_sum = W_new[i, :].sum()
            if row_sum <= eps:
                # If all zeros, initialize uniformly
                W_new[i, :] = 1.0 / k
            else:
                # Normalize to sum to 1
                W_new[i, :] = W_new[i, :] / row_sum
        
        W_new = np.clip(W_new, eps, 1 - eps)
    else:  # dir-beta  
        # W has Beta prior (binary constraint for symmetry)
        # Use same approach as H binary update but transposed
        gradient_W = (2 * V_dense - 1) @ H_new.T  # shape (m, k)
        W_new = (gradient_W > 0).astype(float)
        # Ensure some small non-zero values for numerical stability  
        W_new = np.clip(W_new, eps, 1 - eps)
    
    return W_new, H_new