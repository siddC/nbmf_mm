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
    # - H should be continuous in [0,1] with Beta prior
    # - W should have rows that sum to 1 (simplex constraint)
    H = model.components_
    W = model.W_
    
    # Check H is continuous in [0,1]
    assert np.all((H >= 0) & (H <= 1)), "H must be in [0,1] with default orientation"
    # H should have many unique values (continuous, not binary)
    h_unique = len(np.unique(H))
    assert h_unique > 10, f"H should be continuous, got only {h_unique} unique values"
    
    # Check W rows sum to 1 (simplex constraint)
    row_sums = W.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    print("PASS: Default orientation matches paper (H continuous, W rows simplex)")


if __name__ == "__main__":
    test_paper_default_orientation()