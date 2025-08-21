#!/usr/bin/env python3
"""Validate our implementation matches Magron's behavior."""

import numpy as np
from nbmf_mm import NBMF
import matplotlib.pyplot as plt

def validate_orientation(X, orientation="beta-dir"):
    """Validate a specific orientation."""
    print(f"\n{'='*60}")
    print(f"VALIDATING {orientation.upper()} ORIENTATION")
    print(f"{'='*60}")
    
    # Fit model
    model = NBMF(
        n_components=10,
        alpha=1.2,
        beta=1.2,
        max_iter=200,
        tol=1e-6,
        verbose=0,  # Reduced verbosity for cleaner output
        orientation=orientation
    )
    model.fit(X)
    
    # Get results
    W = model.W_
    H = model.components_
    losses = model.loss_curve_
    
    print(f"\nRESULTS FOR {orientation.upper()}:")
    print("-" * 40)
    
    if orientation.lower() == "beta-dir":
        # 1. Check H is continuous
        H_unique = np.unique(H)
        print(f"1. H Continuity:")
        print(f"   Unique values in H: {len(H_unique)}")
        print(f"   H range: [{H.min():.4f}, {H.max():.4f}]")
        print(f"   H mean: {H.mean():.4f}")
        is_continuous = len(H_unique) > 100
        print(f"   [PASS] H is continuous" if is_continuous else "   [FAIL] H is not continuous!!")
        
        # 2. Check W simplex constraint (rows sum to 1)
        print(f"\n2. W Simplex Constraint (rows sum to 1):")
        row_sums = W.sum(axis=1)
        print(f"   W row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
        simplex_ok = np.allclose(row_sums, 1.0, rtol=1e-5)
        print(f"   [PASS] W rows sum to 1" if simplex_ok else "   [FAIL] W simplex constraint violated!!")
        
    else:  # dir-beta
        # 1. Check W is continuous
        W_unique = np.unique(W)
        print(f"1. W Continuity:")
        print(f"   Unique values in W: {len(W_unique)}")
        print(f"   W range: [{W.min():.4f}, {W.max():.4f}]")
        print(f"   W mean: {W.mean():.4f}")
        is_continuous = len(W_unique) > 100
        print(f"   [PASS] W is continuous" if is_continuous else "   [FAIL] W is not continuous!!")
        
        # 2. Check H simplex constraint (columns sum to 1)
        print(f"\n2. H Simplex Constraint (columns sum to 1):")
        col_sums = H.sum(axis=0)
        print(f"   H column sums: min={col_sums.min():.6f}, max={col_sums.max():.6f}")
        simplex_ok = np.allclose(col_sums, 1.0, rtol=1e-5)
        print(f"   [PASS] H columns sum to 1" if simplex_ok else "   [FAIL] H simplex constraint violated!!")
    
    # 3. Check monotonic convergence
    print(f"\n3. Monotonic Convergence:")
    violations = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i-1] + 1e-12)
    print(f"   Iterations: {len(losses)}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Monotonicity violations: {violations}")
    print(f"   [PASS] Perfect monotonic convergence" if violations == 0 else f"   [FAIL] {violations} violations!!")
    
    return is_continuous, simplex_ok, violations == 0, losses

def main():
    print("="*60)
    print("VALIDATING NBMF-MM IMPLEMENTATION")
    print("BOTH BETA-DIR AND DIR-BETA ORIENTATIONS")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    X = (np.random.rand(100, 50) < 0.3).astype(float)
    print(f"\nData shape: {X.shape}")
    print(f"Data sparsity: {X.mean():.3f}")
    
    # Test both orientations
    results = {}
    all_losses = {}
    
    for orientation in ["beta-dir", "dir-beta"]:
        is_continuous, simplex_ok, monotonic, losses = validate_orientation(X, orientation)
        results[orientation] = (is_continuous, simplex_ok, monotonic)
        all_losses[orientation] = losses
    
    # Overall validation
    print("\n" + "="*60)
    print("OVERALL VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for orientation in ["beta-dir", "dir-beta"]:
        is_continuous, simplex_ok, monotonic = results[orientation]
        orientation_passed = is_continuous and simplex_ok and monotonic
        all_passed = all_passed and orientation_passed
        
        status = "[PASS]" if orientation_passed else "[FAIL]"
        print(f"{status} {orientation.upper()}: Continuous={is_continuous}, Simplex={simplex_ok}, Monotonic={monotonic}")
    
    # Plot convergence for both orientations
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    losses_bd = all_losses["beta-dir"]
    violations_bd = sum(1 for i in range(1, len(losses_bd)) if losses_bd[i] > losses_bd[i-1] + 1e-12)
    plt.semilogy(losses_bd, 'b-', linewidth=2, label='beta-dir')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Beta-Dir Convergence (Violations: {violations_bd})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    losses_db = all_losses["dir-beta"]
    violations_db = sum(1 for i in range(1, len(losses_db)) if losses_db[i] > losses_db[i-1] + 1e-12)
    plt.semilogy(losses_db, 'r-', linewidth=2, label='dir-beta')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Dir-Beta Convergence (Violations: {violations_db})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nbmf_convergence_validation_both_orientations.png')
    plt.show()
    
    # Final result
    print(f"\n{'='*60}")
    if all_passed:
        print("[SUCCESS] ALL VALIDATIONS PASSED FOR BOTH ORIENTATIONS!")
        print("Implementation correctly follows Magron & Fevotte (2022)")
        print("- BETA-DIR: H continuous [0,1], W rows sum to 1")
        print("- DIR-BETA: W continuous [0,1], H columns sum to 1")
        print("- Both orientations exhibit perfect monotonic convergence")
    else:
        print("[WARNING] SOME VALIDATIONS FAILED")
        print("Check implementation against reference")
    print("="*60)

if __name__ == "__main__":
    main()