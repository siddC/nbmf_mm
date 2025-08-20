import numpy as np

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
    m, n = V.shape
    k = W.shape[1]
    
    # Likelihood term
    WH = W @ H
    P = sigmoid(WH)
    eps = 1e-10
    likelihood = -np.sum(V * np.log(P + eps) + (1 - V) * np.log(1 - P + eps))
    
    # Prior term (Beta prior on W)
    prior = -np.sum((alpha - 1) * np.log(W + eps) + (beta - 1) * np.log(1 - W + eps))
    
    return likelihood + prior

def mm_update(V, W, H, alpha=1.0, beta=1.0):
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
    
    Returns
    -------
    W_new : array-like, shape (m, k)
        Updated W
    H_new : array-like, shape (k, n)
        Updated H
    """
    m, n = V.shape
    k = W.shape[1]
    eps = 1e-10
    
    # Step 1: Update H (binary) - Equation (10) from paper
    # For each H[i,j], set to 1 if gradient > 0, else 0
    # Gradient: sum_l W[l,i] * (2*V[l,j] - 1)
    gradient_H = W.T @ (2 * V - 1)  # shape (k, n)
    H_new = (gradient_H > 0).astype(float)
    
    # Step 2: Update W (non-negative) - Equation (11) from paper
    # This is the correct MM multiplicative update from the paper
    # The problem might be the beta value causing issues
    
    # Compute auxiliary variables - this is EXACTLY from the specification
    numerator = V @ H_new.T + (alpha - 1)
    
    # Fixed denominator calculation - this was the issue!
    # The original had: np.ones((m, 1)) @ np.sum(H_new, axis=1, keepdims=True).T + beta
    # But this gives wrong dimensions. It should be:
    denominator = sigmoid(W @ H_new) @ H_new.T + (beta - 1)
    
    # Multiplicative update
    W_new = W * (numerator / np.maximum(denominator, eps))
    
    # Ensure W stays in valid range [eps, 1-eps] for Beta prior
    W_new = np.clip(W_new, eps, 1 - eps)
    
    return W_new, H_new