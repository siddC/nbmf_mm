"""
Test only the public API, no internal imports.
"""
import numpy as np
import pytest
from nbmf_mm import NBMF, NBMFMM  # Only public imports!


class TestPublicAPI:
    """Test suite using only public API."""
    
    def test_basic_fit(self):
        """Test basic fitting works."""
        X = np.random.rand(100, 50)
        model = NBMF(n_components=10)
        model.fit(X)
        
        assert hasattr(model, 'W_')
        assert hasattr(model, 'components_')
        assert model.W_.shape == (100, 10)
        assert model.components_.shape == (10, 50)
    
    def test_transform(self):
        """Test transform method."""
        X_train = np.random.rand(100, 50)
        X_test = np.random.rand(20, 50)
        
        model = NBMF(n_components=10)
        model.fit(X_train)
        W_test = model.transform(X_test)
        
        assert W_test.shape == (20, 10)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.random.rand(100, 50)
        model = NBMF(n_components=10)
        W = model.fit_transform(X)
        
        assert W.shape == (100, 10)
        np.testing.assert_allclose(W, model.W_)
    
    def test_inverse_transform(self):
        """Test inverse_transform method."""
        X = np.random.rand(100, 50)
        model = NBMF(n_components=10)
        model.fit(X)
        
        X_reconstructed = model.inverse_transform(model.W_)
        assert X_reconstructed.shape == X.shape
        assert np.all((X_reconstructed >= 0) & (X_reconstructed <= 1))
    
    def test_score(self):
        """Test score method."""
        X = np.random.rand(100, 50)
        model = NBMF(n_components=10)
        model.fit(X)
        
        score = model.score(X)
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_perplexity(self):
        """Test perplexity method."""
        X = np.random.rand(100, 50)
        model = NBMF(n_components=10)
        model.fit(X)
        
        perp = model.perplexity(X)
        assert isinstance(perp, float)
        assert perp > 0
    
    def test_nbmfmm_alias(self):
        """Test NBMFMM alias works."""
        X = np.random.rand(100, 50)
        model = NBMFMM(n_components=10)  # Using alias
        model.fit(X)
        
        assert hasattr(model, 'W_')
        assert hasattr(model, 'components_')
    
    def test_orientations(self):
        """Test both orientations work."""
        X = np.random.rand(100, 50)
        
        # Test beta-dir (paper default)
        model1 = NBMF(n_components=10, orientation="beta-dir")
        model1.fit(X)
        H1 = model1.components_
        W1 = model1.W_
        
        # H should be continuous in [0,1] with Beta prior
        assert np.all((H1 >= 0) & (H1 <= 1)), "H must be in [0,1] for beta-dir"
        h1_unique = len(np.unique(H1))
        assert h1_unique > 10, f"H should be continuous for beta-dir, got {h1_unique} unique values"
        # W rows should sum to 1
        np.testing.assert_allclose(W1.sum(axis=1), 1.0, rtol=1e-5)
        
        # Test dir-beta (alternative)
        model2 = NBMF(n_components=10, orientation="dir-beta")
        model2.fit(X)
        H2 = model2.components_
        W2 = model2.W_
        
        # H columns should sum to 1
        np.testing.assert_allclose(H2.sum(axis=0), 1.0, rtol=1e-5)
        # W should be continuous in [0,1] with Beta prior
        assert np.all((W2 >= 0) & (W2 <= 1)), "W must be in [0,1] for dir-beta"
        w2_unique = len(np.unique(W2))
        assert w2_unique > 10, f"W should be continuous for dir-beta, got {w2_unique} unique values"
    
    def test_sparse_input(self):
        """Test sparse matrix input."""
        from scipy import sparse
        
        X_dense = np.random.rand(100, 50)
        X_sparse = sparse.csr_matrix(X_dense)
        
        model = NBMF(n_components=10)
        model.fit(X_sparse)
        
        assert hasattr(model, 'W_')
        assert hasattr(model, 'components_')
    
    def test_masked_training(self):
        """Test masked training."""
        X = np.random.rand(100, 50)
        mask = np.random.rand(100, 50) > 0.1  # 90% observed
        
        model = NBMF(n_components=10)
        model.fit(X, mask=mask)
        
        score = model.score(X, mask=mask)
        assert isinstance(score, float)
    
    def test_reproducibility(self):
        """Test random_state gives reproducible results."""
        X = np.random.rand(100, 50)
        
        model1 = NBMF(n_components=10, random_state=42)
        model1.fit(X)
        
        model2 = NBMF(n_components=10, random_state=42)
        model2.fit(X)
        
        np.testing.assert_allclose(model1.W_, model2.W_)
        np.testing.assert_array_equal(model1.components_, model2.components_)
    
    def test_paper_default_orientation(self):
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
    pytest.main([__file__, "-v"])