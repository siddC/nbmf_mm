import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from ._solver import mm_update, compute_nll, sigmoid
from ._utils import check_is_fitted

class NBMFMM(BaseEstimator, TransformerMixin):
    """
    Non-negative Binary Matrix Factorization via Majorization-Minimization.
    
    Implements the NBMF-MM algorithm from:
    P. Magron and C. Févotte, "A majorization-minimization algorithm for
    nonnegative binary matrix factorization," IEEE Signal Processing Letters, 2022.
    
    Parameters
    ----------
    n_components : int, default=10
        Number of components (latent dimension k)
        
    alpha : float, default=1.2
        First parameter of Beta prior for W.
        Values > 1 encourage values away from 0.
        
    beta : float, default=1.2
        Second parameter of Beta prior for W.
        Values > 1 encourage values away from 1.
        
    max_iter : int, default=1000
        Maximum number of iterations
        
    tol : float, default=1e-5
        Tolerance for convergence based on relative change in NLL
        
    init : str, default='random'
        Initialization method:
        - 'random': Random initialization
        - 'custom': Use custom W_init and H_init
        
    W_init : array-like, shape (n_samples, n_components), optional
        Initial W matrix (only used if init='custom')
        
    H_init : array-like, shape (n_components, n_features), optional
        Initial H matrix (only used if init='custom')
        
    random_state : int, RandomState instance or None, default=None
        Random state for initialization
        
    verbose : int, default=0
        Verbosity level
        
    orientation : str, default='beta-dir'
        Orientation of the factorization:
        - 'beta-dir': H binary, W simplex rows (matches paper)
        - 'dir-beta': H simplex columns, W binary
        
    Attributes
    ----------
    W_ : array-like, shape (n_samples, n_components)
        Non-negative factor matrix
        
    components_ : array-like, shape (n_components, n_features)
        Binary factor matrix H
        
    n_iter_ : int
        Actual number of iterations
        
    loss_ : float
        Final negative log-likelihood
        
    loss_curve_ : list
        NLL value at each iteration
    """
    
    def __init__(self, n_components=10, alpha=1.2, beta=1.2,
                 max_iter=1000, tol=1e-5, init='random',
                 W_init=None, H_init=None, random_state=None, verbose=0,
                 orientation="beta-dir"):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.W_init = W_init
        self.H_init = H_init
        self.random_state = random_state
        self.verbose = verbose
        
        # Normalize orientation parameter (case-insensitive aliases)
        if isinstance(orientation, str):
            orientation_lower = orientation.lower().replace(" ", "-")
            orientation_map = {
                "beta-dir": "beta-dir",
                "binary-ica": "beta-dir", 
                "bica": "beta-dir",
                "dir-beta": "dir-beta",
                "aspect-bernoulli": "dir-beta"
            }
            if orientation_lower in orientation_map:
                self.orientation = orientation_map[orientation_lower]
            else:
                raise ValueError(f"Unknown orientation: {orientation}")
        else:
            self.orientation = orientation
    
    def _initialize(self, X):
        """Initialize W and H matrices."""
        m, n = X.shape
        k = self.n_components
        rng = check_random_state(self.random_state)
        
        if self.init == 'random':
            # Initialize W uniformly in (0, 1)
            W = rng.uniform(0.1, 0.9, size=(m, k))
            
            # Initialize H as binary with probability matching data sparsity
            sparsity = X.mean()
            H = (rng.random((k, n)) < sparsity).astype(float)
            
        elif self.init == 'custom':
            if self.W_init is None or self.H_init is None:
                raise ValueError("W_init and H_init must be provided when init='custom'")
            W = np.array(self.W_init, dtype=float)
            H = np.array(self.H_init, dtype=float)
            
            # Validate shapes
            if W.shape != (m, k):
                raise ValueError(f"W_init shape {W.shape} != expected {(m, k)}")
            if H.shape != (k, n):
                raise ValueError(f"H_init shape {H.shape} != expected {(k, n)}")
                
        else:
            raise ValueError(f"Invalid init parameter: {self.init}")
            
        return W, H
    
    def fit(self, X, y=None, mask=None):
        """
        Learn NBMF-MM model for the data X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix (binary or in [0,1])
        y : Ignored
            Not used, present for API consistency
        mask : array-like, shape (n_samples, n_features), optional
            Mask for missing data (1 for observed, 0 for missing)
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=float)
        
        # Handle mask parameter (convert sparse mask to dense if needed)
        if mask is not None:
            mask = check_array(mask, accept_sparse=['csr', 'csc'], dtype=float)
            if hasattr(mask, 'toarray'):  # sparse matrix
                mask = mask.toarray()
        
        # Convert sparse X to dense for validation and algorithm
        if hasattr(X, 'toarray'):  # sparse matrix
            X_dense = X.toarray()
        else:
            X_dense = X
            
        # Validate that X is binary or in [0,1]
        if not (np.all((X_dense == 0) | (X_dense == 1)) or np.all((X_dense >= 0) & (X_dense <= 1))):
            raise ValueError("X must be binary {0,1} or probabilities in [0,1]")
        
        m, n = X_dense.shape
        X = X_dense  # Use dense version for algorithm
        
        # Initialize
        W, H = self._initialize(X)
        
        # Store loss values
        self.loss_curve_ = []
        self.objective_history_ = []
        
        # MM iterations
        for iteration in range(self.max_iter):
            # Compute current loss
            nll = compute_nll(X, W, H, self.alpha, self.beta)
            self.loss_curve_.append(nll)
            self.objective_history_.append(nll)
            
            if self.verbose > 0 and iteration % 10 == 0:
                print(f"Iteration {iteration:4d}, NLL: {nll:.6f}")
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(self.loss_curve_[-1] - self.loss_curve_[-2]) / abs(self.loss_curve_[-2])
                if rel_change < self.tol:
                    if self.verbose > 0:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # MM update
            W, H = mm_update(X, W, H, self.alpha, self.beta, self.orientation)
        
        # Store final results
        self.W_ = W
        self.components_ = H
        self.n_iter_ = iteration + 1
        self.loss_ = self.loss_curve_[-1]
        self.reconstruction_err_ = self.loss_curve_[-1]
        
        return self
    
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
    
    def transform(self, X):
        """
        Transform data X according to fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
            
        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Transformed data
        """
        check_is_fitted(self, ['W_', 'components_'])
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=float)
        
        m, n = X.shape
        k = self.n_components
        
        # Initialize W randomly
        rng = check_random_state(self.random_state)
        W = rng.uniform(0.1, 0.9, size=(m, k))
        
        # Fix H and optimize W
        H = self.components_
        for _ in range(100):  # Mini optimization for W given fixed H
            numerator = X @ H.T + (self.alpha - 1)
            denominator = np.ones((m, 1)) @ np.sum(H, axis=1, keepdims=True).T + self.beta
            W = W * (numerator / (denominator + 1e-10))
            W = np.clip(W, 1e-10, 1 - 1e-10)
        
        return W
    
    def inverse_transform(self, W):
        """
        Transform W back to data space.
        
        Parameters
        ----------
        W : array-like, shape (n_samples, n_components)
            Transformed data
            
        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Data matrix in original space (probabilities)
        """
        check_is_fitted(self, 'components_')
        W = check_array(W, accept_sparse=False, dtype=float)
        
        # Return probabilities
        return sigmoid(W @ self.components_)
    
    def score(self, X, y=None, mask=None):
        """
        Return the average negative log-likelihood on the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
            
        Returns
        -------
        score : float
            Average negative log-likelihood per element
        """
        check_is_fitted(self, ['W_', 'components_'])
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=float)
        
        nll = compute_nll(X, self.W_, self.components_, self.alpha, self.beta)
        return -nll / X.size  # Return negative for sklearn convention (higher is better)
    
    def perplexity(self, X, mask=None):
        """Calculate perplexity of the model on data X."""
        check_is_fitted(self, ['W_', 'components_'])
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=float)
        
        # Simple perplexity calculation
        P = sigmoid(self.W_ @ self.components_)
        if mask is not None:
            mask = check_array(mask, accept_sparse=False, dtype=float)
            ll = np.sum(mask * (X * np.log(P + 1e-10) + (1-X) * np.log(1-P + 1e-10)))
            n_obs = np.sum(mask)
        else:
            ll = np.sum(X * np.log(P + 1e-10) + (1-X) * np.log(1-P + 1e-10))
            n_obs = X.size
            
        return np.exp(-ll / n_obs)


# Alias for backwards compatibility
NBMF = NBMFMM