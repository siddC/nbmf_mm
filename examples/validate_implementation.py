#!/usr/bin/env python3
"""
Validate that our implementation matches the paper's mathematical specification.
"""
import numpy as np
from nbmf_mm import NBMF
import matplotlib.pyplot as plt


def validate_monotonicity():
    """Ensure MM algorithm has monotonic convergence."""
    print("Testing Monotonic Convergence...")
    
    # Generate test data
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    # Fit with paper settings
    model = NBMF(
        n_components=10,
        orientation="beta-dir",  # Paper setting
        alpha=1.2,
        beta=1.2,
        max_iter=200,
        verbose=1
    )
    model.fit(X)
    
    # Check monotonicity
    losses = model.loss_curve_
    violations = 0
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-10:
            violations += 1
            print(f"  Violation at iteration {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}")
    
    if violations == 0:
        print("PASS: Perfect monotonic convergence!!")
    else:
        print(f"FAIL: {violations} monotonicity violations found!!")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood')
    plt.title(f'Monotonic Convergence Test (Violations: {violations})')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/monotonicity_test.png')
    plt.show()
    
    return violations == 0


def validate_constraints():
    """Ensure constraints are satisfied."""
    print("\nTesting Constraints...")
    
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    
    # Test beta-dir (paper setting)
    print("\n1. Testing beta-dir (paper setting):")
    model = NBMF(n_components=10, orientation="beta-dir")
    model.fit(X)
    
    H = model.components_
    W = model.W_
    
    # Check H is continuous in [0,1] 
    h_in_bounds = np.all((H >= 0) & (H <= 1))
    h_unique = len(np.unique(H))
    is_continuous = h_unique > 10  # Should have many unique values
    print(f"  H continuous in [0,1]: {h_in_bounds}")
    print(f"  H unique values: {h_unique} (should be > 10)")
    
    # Check W rows sum to 1
    row_sums = W.sum(axis=1)
    simplex_ok = np.allclose(row_sums, 1.0, rtol=1e-5)
    print(f"  W rows on simplex: {simplex_ok}")
    print(f"  W row sums range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    
    # Test dir-beta (alternative)
    print("\n2. Testing dir-beta (alternative):")
    model2 = NBMF(n_components=10, orientation="dir-beta")
    model2.fit(X)
    H2 = model2.components_
    W2 = model2.W_
    
    # Check H columns sum to 1
    col_sums = H2.sum(axis=0)
    simplex_ok2 = np.allclose(col_sums, 1.0, rtol=1e-5)
    print(f"  H columns on simplex: {simplex_ok2}")
    print(f"  H column sums range: [{col_sums.min():.6f}, {col_sums.max():.6f}]")
    
    # Check W is continuous in [0,1]
    w_in_bounds = np.all((W2 >= 0) & (W2 <= 1))
    w_unique = len(np.unique(W2))
    is_continuous2 = w_unique > 10  # Should have many unique values
    print(f"  W continuous in [0,1]: {w_in_bounds}")
    print(f"  W unique values: {w_unique} (should be > 10)")
    
    return h_in_bounds and is_continuous and simplex_ok and simplex_ok2 and w_in_bounds and is_continuous2


def main():
    print("="*60)
    print("NBMF-MM IMPLEMENTATION VALIDATION")
    print("="*60)
    
    # Run all validations
    mono_ok = validate_monotonicity()
    const_ok = validate_constraints()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Monotonic Convergence: {'PASS' if mono_ok else 'FAIL'}")
    print(f"Constraint Satisfaction: {'PASS' if const_ok else 'FAIL'}")
    
    if mono_ok and const_ok:
        print("\nSUCCESS: All validations passed! Implementation is correct.")
    else:
        print("\nWARNING: Some validations failed. Check implementation.")


if __name__ == "__main__":
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    main()