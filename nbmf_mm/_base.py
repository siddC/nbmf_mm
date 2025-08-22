import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from ._solver import nbmf_mm_solver
from ._utils import check_is_fitted

class NBMFMM(BaseEstimator, TransformerMixin):
    """
    Non-negative Binary Matrix Factorization via Majorization-Minimization.
    
    Implements the NBMF-MM algorithm from:
    P. Magron and C. Févotte, "A majorization-minimization algorithm for
    nonnegative binary matrix factorization," IEEE Signal Processing Letters, 2022.
    
    IMPORTANT: Despite the name "binary", the factor H is continuous in [0,1]
    during optimization. The "binary" refers to the input data Y.
    
    Parameters
    ----------
    n_components : int, default=10
        Number of components (latent dimension k)
        
    alpha : float, default=1.2
        Beta prior parameter for H.
        
    beta : float, default=1.2
        Beta prior parameter for H.
        
    max_iter : int, default=500
        Maximum number of iterations
        
    tol : float, default=1e-5
        Tolerance for convergence based on relative change in loss
        
    W_init : array-like, shape (n_samples, n_components), optional
        Initial W matrix
        
    H_init : array-like, shape (n_components, n_features), optional
        Initial H matrix
        
    random_state : int, RandomState instance or None, default=None
        Random state for initialization
        
    verbose : int, default=0
        Verbosity level
        
    orientation : str, default="beta-dir"
        - "beta-dir": H continuous in [0,1], W rows sum to 1 (simplex)
        - "dir-beta": W continuous in [0,1], H columns sum to 1 (simplex)
        
    Attributes
    ----------
    W_ : array-like, shape (n_samples, n_components)
        Factor matrix - continuous or simplex depending on orientation
        
    components_ : array-like, shape (n_components, n_features)  
        Factor matrix - continuous or simplex depending on orientation
        
    n_iter_ : int
        Actual number of iterations
        
    loss_curve_ : list
        Loss values per iteration
    """
    
    def __init__(self, n_components=10, alpha=1.2, beta=1.2,
                 max_iter=500, tol=1e-5, 
                 W_init=None, H_init=None, init=None, random_state=None, verbose=0,
                 orientation="beta-dir"):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.W_init = W_init
        self.H_init = H_init
        self.init = init  # For compatibility - currently unused
        self.random_state = random_state
        self.verbose = verbose
        # Orientation parameter kept for backward compatibility but ignored
        # Our implementation always follows the paper's beta-dir approach
        self.orientation = orientation
    
    
    def fit(self, X, y=None, mask=None):
        """Fit NBMF model to binary data X."""
        # Validate input
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        
        # Handle sparse matrices first
        if hasattr(X, 'toarray'):  # sparse matrix
            X = X.toarray()
            
        # Check if data is binary or in [0,1]
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError("X must be binary")
        
        # Normalize orientation parameter  
        orientation_normalized = self._normalize_orientation(self.orientation)
        self.orientation = orientation_normalized  # Store normalized form
            
        # Call the solver with paper-correct implementation
        W, H, losses, time_elapsed, n_iter = nbmf_mm_solver(
            Y=X,
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            alpha=self.alpha,
            beta=self.beta,
            W_init=self.W_init,
            H_init=self.H_init,
            mask=mask,
            random_state=self.random_state,
            verbose=self.verbose,
            orientation=orientation_normalized
        )
        
        # Store results
        self.W_ = W  # Shape (n_samples, n_components), format depends on orientation
        self.components_ = H  # Shape (n_components, n_features), format depends on orientation
        self.loss_curve_ = losses
        self.objective_history_ = losses  # Backward compatibility
        self.loss_ = losses[-1] if losses else np.inf  # Current loss
        self.n_iter_ = n_iter
        self.reconstruction_err_ = losses[-1] if losses else np.inf
        
        return self
    
    def _normalize_orientation(self, orientation):
        """Normalize orientation parameter to standard form."""
        # Handle case-insensitive aliases
        orientation_map = {
            "beta-dir": "beta-dir",
            "dir-beta": "dir-beta", 
            "Beta-Dir": "beta-dir",
            "Dir-Beta": "dir-beta",
            "Dir Beta": "dir-beta",
            "binary ICA": "beta-dir",
            "Binary ICA": "beta-dir", 
            "bICA": "beta-dir",
            "Aspect Bernoulli": "dir-beta"
        }
        
        if orientation in orientation_map:
            return orientation_map[orientation]
        else:
            raise ValueError(f"Unknown orientation: {orientation}. "
                           f"Must be one of {list(orientation_map.keys())}")
    
    def fit_transform(self, X, y=None):
        """
        Learn model and return transformed data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
            
        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.W_
    
    def transform(self, X, mask=None):
        """Transform X by finding W given fixed H."""
        check_is_fitted(self, ['components_'])
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        
        if hasattr(X, 'toarray'):  # sparse matrix
            X = X.toarray()
            
        m = X.shape[0]
        k = self.n_components
        H = self.components_
        
        # Initialize W randomly
        W = np.random.uniform(0.1, 0.9, (m, k))
        
        # Run a few iterations to find W given fixed H
        for _ in range(50):
            # Same W update as in fit, but with fixed H
            W_T = W.T
            HW_T = H.T @ W_T
            
            if mask is None:
                Y_T = X.T
                OneminusY_T = (1 - X).T
            else:
                Y_T = X.T * mask.T
                OneminusY_T = (1 - X).T * mask.T
                
            W_T = W_T * (H @ (Y_T / (HW_T + 1e-8)) + (1 - H) @ (OneminusY_T / (1 - HW_T + 1e-8)))
            W_T = W_T / X.shape[1]
            W_T = W_T / W_T.sum(axis=0, keepdims=True)
            W = W_T.T
            
        # Ensure W stays in bounds
        W = np.clip(W, 1e-8, 1.0)
        # Re-normalize rows to sum to 1 after clipping
        W = W / W.sum(axis=1, keepdims=True)
        return W
    
    def inverse_transform(self, W):
        """Transform W back to data space."""
        check_is_fitted(self, ['components_'])
        W = check_array(W, dtype=np.float64)
        
        # Compute reconstruction
        # Note: W has rows summing to 1, H is in (0, 1)
        reconstruction = W @ self.components_
        # Ensure reconstruction stays in [0, 1]
        return np.clip(reconstruction, 0.0, 1.0)
    
    def score(self, X, mask=None):
        """
        Compute the average log-likelihood per observed entry.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
        mask : array-like, optional
            Binary mask for observed entries
            
        Returns
        -------
        score : float
            Average log-likelihood per observed entry
        """
        check_is_fitted(self, ['components_'])
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        
        if hasattr(X, 'toarray'):  # sparse matrix
            X = X.toarray()
            
        # Get reconstruction
        X_recon = self.inverse_transform(self.transform(X))
        
        # Compute log-likelihood
        eps = 1e-8
        if mask is None:
            log_lik = X * np.log(X_recon + eps) + (1 - X) * np.log(1 - X_recon + eps)
            n_obs = X.size
        else:
            X_masked = X * mask
            log_lik = X_masked * np.log(X_recon + eps) + (1 - X_masked) * np.log(1 - X_recon + eps)
            n_obs = np.count_nonzero(mask)
            
        return np.sum(log_lik) / n_obs
    
    def perplexity(self, X, mask=None):
        """
        Compute perplexity of the model on X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
        mask : array-like, optional
            Binary mask for observed entries
            
        Returns
        -------
        perplexity : float
            Perplexity value
        """
        return np.exp(-self.score(X, mask))


# Alias for backwards compatibility
NBMF = NBMFMM