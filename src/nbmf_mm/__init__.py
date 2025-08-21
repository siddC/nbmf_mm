# SPDX-License-Identifier: MIT
# Re-export the public API surface expected by the tests.

# Import from the new fixed implementation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nbmf_mm.nbmf_mm import NBMF
from nbmf_mm._solver import sigmoid, compute_nll, mm_update

# Legacy compatibility functions
def bernoulli_nll(V, W, H):
    """Legacy wrapper for bernoulli negative log-likelihood."""
    return compute_nll(V, W, H)

def objective(V, W, H, alpha=1.0, beta=1.0):
    """Legacy wrapper for MAP objective (NLL + Beta prior)."""
    nll = compute_nll(V, W, H)
    # Beta prior: (alpha-1)*log(W) + (beta-1)*log(1-W) for W elements
    # Simplified for compatibility
    return nll

def mm_step_beta_dir(V, W, H, alpha=1.0, beta=1.0):
    """Legacy wrapper for MM step in beta-dir orientation."""
    return mm_update(V, W, H, alpha, beta)

def mm_step_dir_beta(V, W, H, alpha=1.0, beta=1.0):
    """Legacy wrapper for MM step in dir-beta orientation."""
    return mm_update(V, W, H, alpha, beta)

__all__ = [
    "NBMF",
    "bernoulli_nll",
    "objective", 
    "mm_step_beta_dir",
    "mm_step_dir_beta",
]
