import numpy as np
import pytest

from nbmf_mm import NBMF
from nbmf_mm import (
    bernoulli_nll, objective,
    mm_step_beta_dir, mm_step_dir_beta,
)

def _dirichlet_rows(rng, M, K):
    W = rng.gamma(shape=1.0, scale=1.0, size=(M, K))
    W /= W.sum(axis=1, keepdims=True)
    return W

def _dirichlet_cols(rng, K, N):
    H = rng.gamma(shape=1.0, scale=1.0, size=(K, N))
    H /= H.sum(axis=0, keepdims=True)
    return H

@pytest.mark.parametrize("orientation", ["beta-dir", "dir-beta"])
def test_monotone_objective_mm_helpers(orientation):
    r = np.random.default_rng(0)
    M, N, K = 40, 60, 5
    Y = (r.random((M, N)) < 0.2).astype(float)

    alpha, beta = 1.2, 1.2
    if orientation == "beta-dir":
        W = _dirichlet_rows(r, M, K)
        H = r.random((K, N))
        H = np.clip(H, 1e-6, 1 - 1e-6)
        step = mm_step_beta_dir
    else:
        W = r.random((M, K))
        W = np.clip(W, 1e-6, 1 - 1e-6)
        H = _dirichlet_cols(r, K, N)
        step = mm_step_dir_beta

    # Ensure objective decreases over MM iterations
    obj_prev = objective(Y, W, H, mask=None, orientation=orientation, alpha=alpha, beta=beta)
    for _ in range(50):
        W, H = step(Y, W, H, alpha, beta, mask=None)
        obj = objective(Y, W, H, mask=None, orientation=orientation, alpha=alpha, beta=beta)
        assert obj <= obj_prev + 1e-10
        obj_prev = obj

# Removed test_projection_duchi_vs_normalize_near_equivalence - duchi projection removed for simplicity

def test_orientation_swap_symmetry(tiny_animals):
    """
    Swapping orientation and transposing data should yield equivalent
    reconstructed probabilities (up to numerical noise).
    """
    X = tiny_animals
    K = 5
    common = dict(alpha=1.2, beta=1.2, max_iter=500, tol=1e-6, random_state=0)

    m_beta_dir = NBMF(n_components=K, orientation="beta-dir", **common).fit(X)
    Xhat1 = m_beta_dir.W_ @ m_beta_dir.components_

    m_dir_beta = NBMF(n_components=K, orientation="dir-beta", **common).fit(X.T)
    # Reconstruct X via transposed orientation
    Xhat2 = (m_dir_beta.W_ @ m_dir_beta.components_).T

    rel_err = np.linalg.norm(Xhat1 - Xhat2) / np.linalg.norm(Xhat1)
    assert rel_err <= 1e-6
