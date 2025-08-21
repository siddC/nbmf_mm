"""
NBMF-MM: Non-negative Binary Matrix Factorization via Majorization-Minimization

Implementation of the algorithm from:
P. Magron and C. Févotte, "A majorization-minimization algorithm for
nonnegative binary matrix factorization," IEEE Signal Processing Letters, 2022.
"""

import numpy as np
from ._base import NBMFMM, NBMF
from ._solver import sigmoid, compute_nll, mm_update

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Legacy compatibility functions for tests
def bernoulli_nll(V, Theta, **kwargs):
    """Legacy wrapper for bernoulli negative log-likelihood."""
    # Theta is W @ H in old interface
    V_dense = V.toarray() if hasattr(V, 'toarray') else V
    m, n = V_dense.shape
    P = sigmoid(Theta)
    eps = 1e-10
    return -np.sum(V_dense * np.log(P + eps) + (1 - V_dense) * np.log(1 - P + eps))

def objective(V, W, H, mask=None, orientation="beta-dir", alpha=1.0, beta=1.0):
    """Legacy wrapper for MAP objective (NLL + Beta prior)."""
    return compute_nll(V, W, H, alpha, beta)

def mm_step_beta_dir(V, W, H, alpha=1.0, beta=1.0, mask=None):
    """Legacy wrapper for MM step in beta-dir orientation."""
    return mm_update(V, W, H, alpha, beta, "beta-dir")

def mm_step_dir_beta(V, W, H, alpha=1.0, beta=1.0, mask=None):
    """Legacy wrapper for MM step in dir-beta orientation."""
    return mm_update(V, W, H, alpha, beta, "dir-beta")

__all__ = ['NBMFMM', 'NBMF', 'bernoulli_nll', 'objective', 'mm_step_beta_dir', 'mm_step_dir_beta']