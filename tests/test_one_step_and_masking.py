import numpy as np
import pytest

from nbmf_mm import NBMF

# Helper functions no longer needed as we test the full solver

@pytest.mark.parametrize("orientation", ["beta-dir", "dir-beta"])
def test_simplex_preservation_full_solver(orientation):
    """Test that full solver preserves constraints after fitting."""
    r = np.random.default_rng(123)
    M, N, K = 30, 40, 6
    Y = (r.random((M, N)) < 0.3).astype(float)

    alpha, beta = 1.2, 1.2
    model = NBMF(n_components=K, orientation=orientation, alpha=alpha, beta=beta, 
                 max_iter=10, random_state=123)
    model.fit(Y)
    
    W = model.W_
    H = model.components_
    
    if orientation == "beta-dir":
        # Rows of W must sum to 1
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)
        assert np.all((H >= 0.0) & (H <= 1.0))
    else:
        # Columns of H must sum to 1
        assert np.allclose(H.sum(axis=0), 1.0, atol=1e-10)
        assert np.all((W >= 0.0) & (W <= 1.0))

def test_masked_training_paths():
    """Test training with missing data using mask parameter."""
    r = np.random.default_rng(7)
    M, N, K = 50, 70, 8
    Y = (r.random((M, N)) < 0.25).astype(float)
    # Observe only 80% entries
    mask = (r.random((M, N)) < 0.8).astype(float)

    # Test that masked training works and converges monotonically
    alpha, beta = 1.1, 1.3
    model = NBMF(n_components=K, orientation="beta-dir", alpha=alpha, beta=beta, 
                 max_iter=30, random_state=7, tol=1e-8)
    model.fit(Y, mask=mask)
    
    # Check monotonic convergence
    losses = model.loss_curve_
    violations = []
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-12:
            violations.append(i)
    
    assert len(violations) == 0, f"Found {len(violations)} monotonicity violations with masking!"
    assert len(losses) > 1, "Should have multiple loss values"
