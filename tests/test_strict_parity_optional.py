import numpy as np
import pytest

from nbmf_mm import NBMF

# Helper function no longer needed

def test_initialization_and_convergence():
    """Test that NBMF works with custom initialization and converges properly."""
    r = np.random.default_rng(123)
    M, N, K = 20, 25, 4
    Y = (r.random((M, N)) < 0.3).astype(float)

    alpha, beta = 1.2, 1.2
    
    # Create custom initialization
    W0 = r.gamma(shape=1.0, scale=1.0, size=(M, K))
    W0 /= W0.sum(axis=1, keepdims=True)
    H0 = np.clip(r.random((K, N)), 1e-6, 1 - 1e-6)

    # Test that solver works with custom initialization  
    model = NBMF(
        n_components=K,
        orientation="beta-dir",
        alpha=alpha, beta=beta,
        random_state=123,
        max_iter=50, tol=1e-8,
        W_init=W0, H_init=H0,
    ).fit(Y)

    # Verify constraints are satisfied
    W1 = model.W_
    H1 = model.components_
    
    # W rows should sum to 1 (simplex constraint)
    assert np.allclose(W1.sum(axis=1), 1.0, atol=1e-10)
    # H should be in [0,1]
    assert np.all((H1 >= 0.0) & (H1 <= 1.0))
    
    # Should achieve monotonic convergence
    losses = model.loss_curve_
    violations = []
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-12:
            violations.append(i)
    
    assert len(violations) == 0, f"Found {len(violations)} monotonicity violations!"
