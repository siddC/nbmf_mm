import numpy as np
import pytest
from nbmf_mm import NBMF

def test_h_continuous_not_binary():
    """Verify H stays continuous, NOT binary during optimization."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=50)
    model.fit(X)
    
    H = model.components_
    
    # H should be continuous in (0, 1), NOT binary
    unique_values = np.unique(H)
    assert len(unique_values) > 2, "H should be continuous, not binary!"
    assert np.all((H >= 0) & (H <= 1)), "H should be in [0, 1]"
    
    # Check that H has many distinct values (continuous)
    assert len(unique_values) > 100, f"H has only {len(unique_values)} unique values, should be continuous"
    
    print(f"[PASS] H is continuous with {len(unique_values)} unique values")

def test_w_simplex_constraint():
    """Verify W rows sum to 1 (simplex constraint)."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=50)
    model.fit(X)
    
    W = model.W_
    row_sums = W.sum(axis=1)
    
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5,
                              err_msg="W rows must sum to 1 (simplex constraint)")
    
    print(f"[PASS] W rows sum to 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}")

def test_monotonic_convergence():
    """Test that loss decreases monotonically (MM property)."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=100, tol=1e-8)
    model.fit(X)
    
    losses = model.loss_curve_
    
    # Check strict monotonicity
    violations = []
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-12:
            violations.append(i)
            print(f" Violation at iter {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}")
    
    assert len(violations) == 0, f"Found {len(violations)} monotonicity violations!"
    
    print(f"[PASS] Perfect monotonic convergence over {len(losses)} iterations")

def test_reconstruction_probabilities():
    """Test that reconstruction gives valid probabilities."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10)
    model.fit(X)
    
    X_reconstructed = model.inverse_transform(model.W_)
    
    # Should be probabilities in (0, 1)
    assert np.all((X_reconstructed >= 0) & (X_reconstructed <= 1)), \
        "Reconstructed values should be probabilities in [0, 1]"
    
    # Should NOT be binary
    unique_recon = np.unique(X_reconstructed)
    assert len(unique_recon) > 100, \
        f"Reconstruction should be continuous probabilities, got {len(unique_recon)} unique values"
        
    print(f"[PASS] Reconstruction gives continuous probabilities")

def test_beta_prior_effect():
    """Test that Beta prior parameters affect the solution."""
    np.random.seed(42)
    X = (np.random.rand(50, 30) < 0.3).astype(float)
    
    # Model with symmetric prior (no preference)
    model1 = NBMF(n_components=5, alpha=1.0, beta=1.0, max_iter=100)
    model1.fit(X)
    H1 = model1.components_
    
    # Model with prior favoring values near 0
    model2 = NBMF(n_components=5, alpha=0.5, beta=2.0, max_iter=100, random_state=42)
    model2.fit(X)
    H2 = model2.components_
    
    # Model with prior favoring values near 1
    model3 = NBMF(n_components=5, alpha=2.0, beta=0.5, max_iter=100, random_state=42)
    model3.fit(X)
    H3 = model3.components_
    
    # Check that priors have expected effect on H
    assert H2.mean() < H1.mean(), "Beta(0.5, 2) should push H toward 0"
    assert H3.mean() > H1.mean(), "Beta(2, 0.5) should push H toward 1"
    
    print(f"[PASS] Beta prior affects solution: H means = {H1.mean():.3f}, {H2.mean():.3f}, {H3.mean():.3f}")

def test_dir_beta_w_continuous_not_binary():
    """Verify W stays continuous in dir-beta orientation, NOT binary during optimization."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=50, orientation="dir-beta")
    model.fit(X)
    
    W = model.W_
    
    # W should be continuous in (0, 1), NOT binary
    unique_values = np.unique(W)
    assert len(unique_values) > 2, "W should be continuous, not binary!"
    assert np.all((W >= 0) & (W <= 1)), "W should be in [0, 1]"
    
    # Check that W has many distinct values (continuous)
    assert len(unique_values) > 100, f"W has only {len(unique_values)} unique values, should be continuous"
    
    print(f"[PASS] W is continuous with {len(unique_values)} unique values (dir-beta)")

def test_dir_beta_h_simplex_constraint():
    """Verify H columns sum to 1 (simplex constraint) in dir-beta orientation."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=50, orientation="dir-beta")
    model.fit(X)
    
    H = model.components_
    col_sums = H.sum(axis=0)
    
    np.testing.assert_allclose(col_sums, 1.0, rtol=1e-5,
                              err_msg="H columns must sum to 1 (simplex constraint)")
    
    print(f"[PASS] H columns sum to 1: min={col_sums.min():.6f}, max={col_sums.max():.6f} (dir-beta)")

def test_dir_beta_monotonic_convergence():
    """Test that loss decreases monotonically in dir-beta orientation."""
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    model = NBMF(n_components=10, max_iter=100, tol=1e-8, orientation="dir-beta")
    model.fit(X)
    
    losses = model.loss_curve_
    
    # Check strict monotonicity
    violations = []
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-12:
            violations.append(i)
            print(f" Violation at iter {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}")
    
    assert len(violations) == 0, f"Found {len(violations)} monotonicity violations!"
    
    print(f"[PASS] Perfect monotonic convergence over {len(losses)} iterations (dir-beta)")

def test_orientation_symmetry():
    """Test that both orientations work and have symmetric behavior."""
    np.random.seed(42)
    X = (np.random.rand(50, 30) < 0.3).astype(float)
    
    # Test beta-dir
    model_bd = NBMF(n_components=5, max_iter=100, orientation="beta-dir", random_state=42)
    model_bd.fit(X)
    
    # Test dir-beta  
    model_db = NBMF(n_components=5, max_iter=100, orientation="dir-beta", random_state=42)
    model_db.fit(X)
    
    # Check constraints for beta-dir
    W_bd = model_bd.W_
    H_bd = model_bd.components_
    row_sums_bd = W_bd.sum(axis=1)
    assert np.allclose(row_sums_bd, 1.0, rtol=1e-5), "beta-dir: W rows should sum to 1"
    assert len(np.unique(H_bd)) > 50, "beta-dir: H should be continuous"
    
    # Check constraints for dir-beta
    W_db = model_db.W_
    H_db = model_db.components_
    col_sums_db = H_db.sum(axis=0)
    assert np.allclose(col_sums_db, 1.0, rtol=1e-5), "dir-beta: H columns should sum to 1"
    assert len(np.unique(W_db)) > 50, "dir-beta: W should be continuous"
    
    print(f"[PASS] Both orientations work correctly with proper constraints")