import numpy as np
import matplotlib.pyplot as plt
from nbmf_mm import NBMFMM
from nbmf_mm._utils import generate_synthetic_binary_data

def test_convergence_plot():
    """Generate convergence plot for visual inspection."""
    X, _, _ = generate_synthetic_binary_data(100, 80, 10, random_state=42)
    
    model = NBMFMM(n_components=10, max_iter=500, tol=1e-8,
                  random_state=42, verbose=1)
    model.fit(X)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(model.loss_curve_, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('NBMF-MM Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    
    print(f"Converged in {model.n_iter_} iterations")
    print(f"Final NLL: {model.loss_}")
    
    # Check strict monotonicity
    losses = model.loss_curve_
    violations = []
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] + 1e-12:
            violations.append((i, losses[i-1], losses[i]))
    
    if violations:
        print(f"WARNING: {len(violations)} monotonicity violations found:")
        for i, prev, curr in violations[:5]:  # Show first 5
            print(f"  Iteration {i}: {prev:.10f} -> {curr:.10f} (diff: {curr-prev:.2e})")
    else:
        print("✓ Perfect monotonic convergence achieved!")

if __name__ == "__main__":
    test_convergence_plot()