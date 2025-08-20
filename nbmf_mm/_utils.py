import numpy as np

def check_is_fitted(estimator, attributes):
    """Check if estimator is fitted by verifying attributes exist."""
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet.")

def generate_synthetic_binary_data(n_samples=100, n_features=50, n_components=5,
                                 sparsity=0.3, random_state=None):
    """
    Generate synthetic binary data with known factorization.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_components : int
        Number of latent components
    sparsity : float
        Sparsity level (probability of 1)
    random_state : int or RandomState
        Random seed
        
    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Binary data matrix
    W_true : array-like, shape (n_samples, n_components)
        True W matrix
    H_true : array-like, shape (n_components, n_features)
        True H matrix
    """
    rng = np.random.RandomState(random_state)
    
    # Generate true factors
    W_true = rng.uniform(0.1, 0.9, size=(n_samples, n_components))
    H_true = (rng.random((n_components, n_features)) < sparsity).astype(float)
    
    # Generate observations
    P = 1 / (1 + np.exp(-W_true @ H_true))
    X = (rng.random((n_samples, n_features)) < P).astype(float)
    
    return X, W_true, H_true