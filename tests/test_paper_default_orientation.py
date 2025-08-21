"""
Verify default orientation matches Magron & Févotte 2022.
"""
import numpy as np
from nbmf_mm import NBMF


def test_paper_default_orientation():
    """Verify default orientation matches Magron & Févotte 2022."""
    X = np.random.rand(50, 30)
    model = NBMF(n_components=5)  # Using defaults
    model.fit(X)
    
    # With paper default (beta-dir):
    # - H should be binary
    # - W should be non-negative with simplex rows
    H = model.components_
    W = model.W_
    
    # Check H is binary (allowing for small numerical errors)
    H_rounded = np.round(H)
    assert np.allclose(H, H_rounded, atol=1e-8), "H must be binary with default orientation"
    assert np.all((H_rounded == 0) | (H_rounded == 1)), "H must be binary with default orientation"
    
    # Check W rows sum to 1 (simplex constraint)
    row_sums = W.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    print("✓ Default orientation matches paper")


if __name__ == "__main__":
    test_paper_default_orientation()