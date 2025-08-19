#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Magron & Févotte (2022) NBMF-MM experiments.

This script reproduces the experiments from the original NBMF-MM paper using our
scikit-learn compatible implementation. It performs hyperparameter tuning and
evaluation on the same datasets and train/val/test splits as the original work.

Datasets
--------
- animals: Binary presence/absence matrix of animals across locations
- paleo: Binary fossil occurrence matrix across geological periods
- lastfm: Binary music listening matrix (users × artists)

Protocol
--------
1. Load datasets from data/magron2022/*.rda files
2. Use pre-computed train/val/test splits from data/magron2022/
3. Perform grid search over (K, α, β) hyperparameters
4. Evaluate best models on test set with multiple random initializations
5. Save results to outputs/chauhan2025/

Outputs
-------
For each dataset, saves:
- nbmf-mm_val.npz: Validation cross-entropy for all hyperparameter combinations
- nbmf-mm_model.npz: Best model parameters (W, H) from validation
- nbmf-mm_test_init.npz: Test cross-entropy across multiple random initializations

References
----------
.. [1] P. Magron & C. Févotte (2022). A majorization-minimization algorithm for
       nonnegative binary matrix factorization. IEEE Signal Processing Letters.
       https://doi.org/10.1109/LSP.2022.3185610

.. [2] Original implementation: https://github.com/magronp/NMF-binary
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pyreadr


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Find the repository root directory.
    
    Walks up the directory tree looking for a directory containing both
    'data' and 'outputs' folders.
    
    Parameters
    ----------
    start : Path, optional
        Starting directory for the search. If None, uses the directory
        containing this script.
        
    Returns
    -------
    Path
        Path to the repository root directory.
        
    Raises
    ------
    FileNotFoundError
        If repository root cannot be found.
    """
    here = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Could not find repository root (no directory with 'data' and 'outputs')")


# Setup paths and imports
REPO_ROOT = find_repo_root()
for add in (REPO_ROOT, REPO_ROOT / "src"):
    if str(add) not in sys.path:
        sys.path.insert(0, str(add))

from nbmf_mm import NBMF  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "magron2022"
OUT_ROOT = REPO_ROOT / "outputs" / "chauhan2025"


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def bernoulli_cross_entropy(
    X: np.ndarray, 
    P: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    eps: float = 1e-9
) -> float:
    """
    Compute mean negative log-likelihood (cross-entropy) for Bernoulli model.
    
    This matches the evaluation metric used in the original paper.
    
    Parameters
    ----------
    X : np.ndarray
        Binary observations (0 or 1)
    P : np.ndarray
        Predicted probabilities (0 < p < 1)
    mask : np.ndarray, optional
        Binary mask indicating observed entries. If None, all entries are observed.
    eps : float, default=1e-9
        Small constant to prevent log(0)
        
    Returns
    -------
    float
        Mean cross-entropy over observed entries
    """
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    P = np.clip(P, eps, 1.0 - eps)
    nll = -(mask * (X * np.log(P) + (1 - X) * np.log(1 - P))).sum()
    denom = float(mask.sum()) if float(mask.sum()) > 0 else 1.0
    return float(nll / denom)


def fit_nbmf_mm(
    X: np.ndarray,
    train_mask: np.ndarray,
    n_components: int,
    alpha: float,
    beta: float,
    *,
    orientation: str = "beta-dir",
    max_iter: int = 2000,
    tol: float = 1e-8,
    seed: int = 0,
    use_numexpr: bool = True,
    projection_method: str = "duchi",
    projection_backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Fit NBMF-MM model with given hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Binary data matrix
    train_mask : np.ndarray
        Binary mask for training entries
    n_components : int
        Number of latent components (rank K)
    alpha : float
        Beta prior parameter α
    beta : float
        Beta prior parameter β
    orientation : str, default="beta-dir"
        Factorization orientation ("beta-dir" matches the paper)
    max_iter : int, default=2000
        Maximum iterations for optimization
    tol : float, default=1e-8
        Convergence tolerance
    seed : int, default=0
        Random seed for reproducibility
    use_numexpr : bool, default=True
        Whether to use NumExpr acceleration
    projection_method : str, default="duchi"
        Simplex projection method
    projection_backend : str, default="auto"
        Backend for simplex projection
        
    Returns
    -------
    W : np.ndarray
        Sample weights matrix (n_samples × n_components)
    H : np.ndarray
        Feature weights matrix (n_features × n_components)
    Xhat : np.ndarray
        Reconstructed probabilities
    total_time : float
        Total fitting time in seconds
    n_iter : int
        Number of iterations until convergence
    """
    t0 = time.perf_counter()
    model = NBMF(
        n_components=n_components,
        orientation=orientation,
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        use_numexpr=use_numexpr,
        projection_method=projection_method,
        projection_backend=projection_backend,
    ).fit(X, mask=train_mask)
    
    total_time = time.perf_counter() - t0
    W = model.W_
    H = model.components_.T  # Transpose to match original format
    Xhat = model.inverse_transform(W)
    
    return W, H, Xhat, total_time, model.n_iter_


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset and pre-computed train/val/test splits.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset ("animals", "paleo", or "lastfm")
        
    Returns
    -------
    X : np.ndarray
        Binary data matrix
    train_mask : np.ndarray
        Training mask
    val_mask : np.ndarray
        Validation mask
    test_mask : np.ndarray
        Test mask
    """
    # Load data
    data_file = DATA_DIR / f"{dataset_name}.rda"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    result = pyreadr.read_r(str(data_file))
    X = np.asarray(list(result.values())[0], dtype=float)
    
    # Load splits
    split_file = SPLIT_DIR / f"{dataset_name}_split.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    splits = np.load(split_file)
    train_mask = splits["train_mask"]
    val_mask = splits["val_mask"]
    test_mask = splits["test_mask"]
    
    return X, train_mask, val_mask, test_mask


def run_hyperparameter_grid(
    dataset_name: str,
    K_grid: list[int],
    alpha_grid: list[float],
    beta_grid: list[float],
    n_init: int = 1,
    seed: int = 0,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Run hyperparameter grid search for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    K_grid : list[int]
        Grid of rank values to try
    alpha_grid : list[float]
        Grid of α values to try
    beta_grid : list[float]
        Grid of β values to try
    n_init : int, default=1
        Number of random initializations per hyperparameter combination
    seed : int, default=0
        Random seed for reproducibility
        
    Returns
    -------
    val_scores : np.ndarray
        Validation cross-entropy scores for all combinations
    best_params : dict
        Best hyperparameters found
    best_model : dict
        Best model parameters (W, H)
    """
    print(f"\nRunning hyperparameter grid for {dataset_name}...")
    
    X, train_mask, val_mask, test_mask = load_dataset(dataset_name)
    
    # Initialize results arrays
    val_scores = np.full((len(K_grid), len(alpha_grid), len(beta_grid)), np.nan)
    best_val_score = np.inf
    best_params = {}
    best_model = {}
    
    # Grid search
    for i, K in enumerate(K_grid):
        for j, alpha in enumerate(alpha_grid):
            for k, beta in enumerate(beta_grid):
                print(f"  K={K}, α={alpha:.1f}, β={beta:.1f}", end=" ")
                
                # Try multiple initializations
                scores = []
                for init in range(n_init):
                    try:
                        W, H, Xhat, _, _ = fit_nbmf_mm(
                            X, train_mask, K, alpha, beta, seed=seed + init
                        )
                        score = bernoulli_cross_entropy(X, Xhat, val_mask)
                        scores.append(score)
                    except Exception as e:
                        print(f"Error: {e}")
                        scores.append(np.nan)
                
                # Use best score from initializations
                if scores and not all(np.isnan(scores)):
                    val_score = np.nanmin(scores)
                    val_scores[i, j, k] = val_score
                    print(f"→ {val_score:.6f}")
                    
                    # Update best if better
                    if val_score < best_val_score:
                        best_val_score = val_score
                        best_params = {"K": K, "alpha": alpha, "beta": beta}
                        # Save best model
                        W, H, Xhat, _, _ = fit_nbmf_mm(
                            X, train_mask, K, alpha, beta, seed=seed
                        )
                        best_model = {"W": W, "H": H, "Xhat": Xhat}
                else:
                    print("→ failed")
    
    return val_scores, best_params, best_model


def evaluate_test_performance(
    dataset_name: str,
    best_params: dict,
    n_init: int = 10,
    seed: int = 0,
) -> Tuple[list[float], list[float], list[int]]:
    """
    Evaluate best model on test set with multiple initializations.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    best_params : dict
        Best hyperparameters from validation
    n_init : int, default=10
        Number of random initializations
    seed : int, default=0
        Random seed for reproducibility
        
    Returns
    -------
    test_scores : list[float]
        Test cross-entropy scores
    test_times : list[float]
        Training times
    test_iters : list[int]
        Number of iterations until convergence
    """
    print(f"\nEvaluating test performance for {dataset_name}...")
    
    X, train_mask, val_mask, test_mask = load_dataset(dataset_name)
    
    test_scores = []
    test_times = []
    test_iters = []
    
    for init in range(n_init):
        print(f"  Initialization {init + 1}/{n_init}", end=" ")
        try:
            W, H, Xhat, total_time, n_iter = fit_nbmf_mm(
                X, train_mask,
                best_params["K"],
                best_params["alpha"],
                best_params["beta"],
                seed=seed + init
            )
            score = bernoulli_cross_entropy(X, Xhat, test_mask)
            test_scores.append(score)
            test_times.append(total_time)
            test_iters.append(n_iter)
            print(f"→ {score:.6f} ({total_time:.2f}s)")
        except Exception as e:
            print(f"Error: {e}")
            test_scores.append(np.nan)
            test_times.append(np.nan)
            test_iters.append(np.nan)
    
    return test_scores, test_times, test_iters


def main():
    """Run the complete reproduction experiment."""
    print("NBMF-MM Reproduction Experiment")
    print("=" * 40)
    
    # Ensure output directory exists
    ensure_dir(OUT_ROOT)
    
    # Hyperparameter grids (matching the paper)
    K_grids = {
        "animals": [3, 4, 5, 6, 8],
        "paleo": [5, 10, 15, 20, 25],
        "lastfm": [10, 15, 20, 25, 30],
    }
    alpha_beta_grid = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    
    datasets = ["animals", "paleo", "lastfm"]
    
    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} {'='*20}")
        
        # Create dataset output directory
        dataset_out = OUT_ROOT / dataset
        ensure_dir(dataset_out)
        
        # Run hyperparameter grid search
        val_scores, best_params, best_model = run_hyperparameter_grid(
            dataset,
            K_grids[dataset],
            alpha_beta_grid,
            alpha_beta_grid,
            n_init=1,
            seed=0,
        )
        
        # Save validation results
        np.savez_compressed(
            dataset_out / "nbmf-mm_val.npz",
            val_pplx=val_scores,
            list_hyper=[K_grids[dataset], alpha_beta_grid, alpha_beta_grid],
        )
        
        # Save best model
        np.savez_compressed(
            dataset_out / "nbmf-mm_model.npz",
            W=best_model["W"],
            H=best_model["H"],
            Xhat=best_model["Xhat"],
            best_params=best_params,
        )
        
        # Evaluate on test set
        test_scores, test_times, test_iters = evaluate_test_performance(
            dataset, best_params, n_init=10, seed=0
        )
        
        # Save test results
        np.savez_compressed(
            dataset_out / "nbmf-mm_test_init.npz",
            test_pplx=test_scores,
            test_time=test_times,
            test_iter=test_iters,
            best_params=best_params,
        )
        
        print(f"\n{dataset} completed!")
        print(f"Best parameters: K={best_params['K']}, α={best_params['alpha']:.1f}, β={best_params['beta']:.1f}")
        print(f"Test cross-entropy: {np.nanmean(test_scores):.6f} ± {np.nanstd(test_scores):.6f}")
    
    print(f"\n{'='*50}")
    print("All experiments completed!")
    print(f"Results saved to: {OUT_ROOT}")
    print("\nNext steps:")
    print("1. Run examples/display_reproduced_results.py to generate figures")
    print("2. Compare with outputs/magron2022/figures/ for validation")


if __name__ == "__main__":
    main()
