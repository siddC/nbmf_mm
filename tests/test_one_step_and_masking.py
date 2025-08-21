import numpy as np
import pytest

from nbmf_mm import mm_step_beta_dir, mm_step_dir_beta, objective

def _dirichlet_rows(rng, M, K):
    W = rng.gamma(shape=1.0, scale=1.0, size=(M, K))
    W /= W.sum(axis=1, keepdims=True)
    return W

def _dirichlet_cols(rng, K, N):
    H = rng.gamma(shape=1.0, scale=1.0, size=(K, N))
    H /= H.sum(axis=0, keepdims=True)
    return H

@pytest.mark.parametrize("orientation", ["beta-dir", "dir-beta"])
def test_simplex_preservation_one_step(orientation):
    r = np.random.default_rng(123)
    M, N, K = 30, 40, 6
    Y = (r.random((M, N)) < 0.3).astype(float)

    alpha, beta = 1.2, 1.2
    if orientation == "beta-dir":
        W = _dirichlet_rows(r, M, K)
        H = np.clip(r.random((K, N)), 1e-6, 1 - 1e-6)
        W2, H2 = mm_step_beta_dir(Y, W, H, alpha, beta, mask=None)
        # Rows of W must still sum to 1
        assert np.allclose(W2.sum(axis=1), 1.0, atol=1e-12)
        assert np.all((H2 >= 0.0) & (H2 <= 1.0))
    else:
        W = np.clip(r.random((M, K)), 1e-6, 1 - 1e-6)
        H = _dirichlet_cols(r, K, N)
        W2, H2 = mm_step_dir_beta(Y, W, H, alpha, beta, mask=None)
        # Columns of H must still sum to 1
        assert np.allclose(H2.sum(axis=0), 1.0, atol=1e-12)
        assert np.all((W2 >= 0.0) & (W2 <= 1.0))

def test_masked_training_paths():
    r = np.random.default_rng(7)
    M, N, K = 50, 70, 8
    Y = (r.random((M, N)) < 0.25).astype(float)
    # Observe only 80% entries
    mask = (r.random((M, N)) < 0.8).astype(float)

    # Run a few iterations with helpers to ensure stability & monotonicity
    alpha, beta = 1.1, 1.3
    W = _dirichlet_rows(r, M, K)
    H = np.clip(r.random((K, N)), 1e-6, 1 - 1e-6)
    obj_prev = objective(Y, W, H, mask, "beta-dir", alpha, beta)
    for _ in range(30):
        W, H = mm_step_beta_dir(Y, W, H, alpha, beta, mask=mask)
        obj = objective(Y, W, H, mask, "beta-dir", alpha, beta)
        assert obj <= obj_prev + 1e-10
        obj_prev = obj
