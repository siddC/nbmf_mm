#!/usr/bin/env python3
"""
Reproduce experiments from Magron & Févotte (2022) using nbmf_mm package.
Generates data for Figures 1, 2, and 3 comparison.
"""
import os
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from scipy.io import loadmat
import pyreadr
from nbmf_mm import NBMF

# Setup paths
DATA_DIR = Path("data")
SPLIT_DIR = Path("data/magron2022")
OUTPUT_DIR = Path("outputs/chauhan2025")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(12345)

def load_dataset_and_splits(dataset_name):
    """Load dataset and pre-computed splits."""
    # Load the raw data (.rda file)
    data_path = DATA_DIR / f"{dataset_name}.rda"
    data = pyreadr.read_r(str(data_path))[dataset_name].to_numpy()
    
    # Load the splits
    split_path = SPLIT_DIR / f"{dataset_name}_split.npz"
    splits = np.load(split_path)
    train_mask = splits['train_mask']
    val_mask = splits['val_mask']
    test_mask = splits['test_mask']
    
    return data, train_mask, val_mask, test_mask

def compute_perplexity(Y, Y_hat, mask=None, eps=1e-8):
    """Compute perplexity for binary data."""
    if mask is None:
        mask = np.ones_like(Y)
    
    log_lik = Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps)
    perplexity = -np.sum(mask * log_lik) / np.count_nonzero(mask)
    return np.exp(perplexity)

def train_nbmf_mm(Y, train_mask, n_components, alpha, beta, max_iter=500, tol=1e-5):
    """Train NBMF-MM model using our implementation."""
    # Use beta-dir orientation to match paper
    model = NBMF(
        n_components=n_components,
        orientation="beta-dir",  # W rows sum to 1, H continuous
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        random_state=12345,
        verbose=0
    )
    
    # Fit on training data
    start_time = time.time()
    model.fit(Y, mask=train_mask)
    train_time = time.time() - start_time
    
    # Get factors
    W = model.W_
    H = model.components_
    n_iter = model.n_iter_
    
    return W, H, train_time, n_iter

def run_figure1_experiments():
    """
    Figure 1: Validation perplexity vs hyperparameters.
    Tests different values of alpha and beta on validation set.
    """
    print("\n" + "="*60)
    print("FIGURE 1: Hyperparameter Validation")
    print("="*60)
    
    datasets = ['animals', 'lastfm', 'paleo']
    
    # Hyperparameter grids (from paper)
    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    beta_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Fixed n_components for validation (from paper)
    n_components_dict = {
        'animals': 4,
        'lastfm': 8,
        'paleo': 4
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        
        # Load data and splits
        Y, train_mask, val_mask, test_mask = load_dataset_and_splits(dataset)
        print(f"  Shape: {Y.shape}")
        print(f"  Sparsity: {Y.mean():.4f}")
        
        k = n_components_dict[dataset]
        dataset_results = []
        
        for alpha in alpha_values:
            for beta in beta_values:
                print(f"  Testing a={alpha:.1f}, b={beta:.1f}...", end=" ")
                
                # Train model
                W, H, train_time, n_iter = train_nbmf_mm(
                    Y, train_mask, k, alpha, beta
                )
                
                # Compute perplexities
                Y_hat = W @ H
                train_perp = compute_perplexity(Y, Y_hat, train_mask)
                val_perp = compute_perplexity(Y, Y_hat, val_mask)
                
                dataset_results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'train_perplexity': train_perp,
                    'val_perplexity': val_perp,
                    'n_iter': n_iter,
                    'time': train_time
                })
                
                print(f"Val perp: {val_perp:.3f}")
        
        results[dataset] = pd.DataFrame(dataset_results)
        
        # Save results
        output_path = OUTPUT_DIR / f"figure1_{dataset}_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(dataset_results, f)
        
        # Also save as CSV for easy viewing
        csv_path = OUTPUT_DIR / f"figure1_{dataset}_results.csv"
        results[dataset].to_csv(csv_path, index=False)
        
        # Find best hyperparameters
        best_idx = results[dataset]['val_perplexity'].idxmin()
        best_params = results[dataset].iloc[best_idx]
        print(f"  Best: a={best_params['alpha']:.1f}, b={best_params['beta']:.1f}, "
              f"Val perp={best_params['val_perplexity']:.3f}")
    
    return results

def run_figure2_experiments():
    """
    Figure 2: Test perplexity comparison.
    Compare our NBMF-MM with Magron's results.
    """
    print("\n" + "="*60)
    print("FIGURE 2: Test Performance Comparison")
    print("="*60)
    
    datasets = ['animals', 'lastfm', 'paleo']
    
    # Best hyperparameters (from Figure 1 or paper)
    best_params = {
        'animals': {'alpha': 2.0, 'beta': 2.0, 'k': 4},
        'lastfm': {'alpha': 1.0, 'beta': 1.0, 'k': 8},
        'paleo': {'alpha': 2.0, 'beta': 2.0, 'k': 4}
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        
        # Load data and splits
        Y, train_mask, val_mask, test_mask = load_dataset_and_splits(dataset)
        params = best_params[dataset]
        
        # Train with best parameters
        W, H, train_time, n_iter = train_nbmf_mm(
            Y, train_mask,
            params['k'],
            params['alpha'],
            params['beta'],
            max_iter=1000
        )
        
        # Compute test perplexity
        Y_hat = W @ H
        test_perp = compute_perplexity(Y, Y_hat, test_mask)
        
        # Load Magron's results for comparison
        magron_path = Path(f"outputs/magron2022/{dataset}/NBMF-MM_val.npz")
        if magron_path.exists():
            magron_data = np.load(magron_path, allow_pickle=True)
            val_pplx = magron_data['val_pplx']  # shape: (n_k, n_alpha, n_beta)
            list_hyper = magron_data['list_hyper']
            k_values = list_hyper[0]
            
            # Find k index that matches our k
            k_idx = None
            for i, k_val in enumerate(k_values):
                if k_val == params['k']:
                    k_idx = i
                    break
            
            if k_idx is not None:
                # Find best alpha, beta for this k
                k_results = val_pplx[k_idx]  # shape: (n_alpha, n_beta)
                min_idx = np.unravel_index(np.argmin(k_results), k_results.shape)
                magron_perp = k_results[min_idx]
            else:
                magron_perp = np.nan
        else:
            magron_perp = np.nan
        
        results[dataset] = {
            'chauhan2025_perplexity': test_perp,
            'magron2022_perplexity': magron_perp,
            'n_iter': n_iter,
            'time': train_time,
            'W': W,
            'H': H
        }
        
        print(f"  Our perplexity: {test_perp:.4f}")
        print(f"  Magron perplexity: {magron_perp:.4f}")
        print(f"  Iterations: {n_iter}")
        print(f"  Time: {train_time:.2f}s")
        
        # Save results
        output_path = OUTPUT_DIR / f"figure2_{dataset}_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results[dataset], f)
    
    return results

def run_figure3_experiments():
    """
    Figure 3: Varying n_components comparison.
    Compare our NBMF-MM with Magron's across different k values.
    """
    print("\n" + "="*60)
    print("FIGURE 3: Components Analysis")
    print("="*60)
    
    datasets = ['animals', 'lastfm', 'paleo']
    
    # Range of k values to test (matching Magron's experiment)
    k_ranges = {
        'animals': [2, 4, 8, 16],
        'lastfm': [2, 4, 8, 16],
        'paleo': [2, 4, 8, 16]
    }
    
    # Use optimal hyperparameters from Figure 1
    best_params = {
        'animals': {'alpha': 2.0, 'beta': 2.0},
        'lastfm': {'alpha': 1.0, 'beta': 1.0},
        'paleo': {'alpha': 2.0, 'beta': 2.0}
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        
        # Load data and splits
        Y, train_mask, val_mask, test_mask = load_dataset_and_splits(dataset)
        params = best_params[dataset]
        k_values = k_ranges[dataset]
        
        dataset_results = []
        
        for k in k_values:
            print(f"  k={k}...", end=" ")
            
            # Train model
            W, H, train_time, n_iter = train_nbmf_mm(
                Y, train_mask, k,
                params['alpha'],
                params['beta']
            )
            
            # Compute perplexities
            Y_hat = W @ H
            train_perp = compute_perplexity(Y, Y_hat, train_mask)
            val_perp = compute_perplexity(Y, Y_hat, val_mask)
            test_perp = compute_perplexity(Y, Y_hat, test_mask)
            
            # Load Magron's results if available
            magron_path = Path(f"outputs/magron2022/{dataset}/NBMF-MM_val.npz")
            if magron_path.exists():
                magron_data = np.load(magron_path, allow_pickle=True)
                val_pplx = magron_data['val_pplx']  # shape: (n_k, n_alpha, n_beta)
                list_hyper = magron_data['list_hyper']
                k_values = list_hyper[0]
                
                # Find k index that matches our k
                k_idx = None
                for i, k_val in enumerate(k_values):
                    if k_val == k:
                        k_idx = i
                        break
                
                if k_idx is not None:
                    # Find best alpha, beta for this k
                    k_results = val_pplx[k_idx]  # shape: (n_alpha, n_beta)
                    min_idx = np.unravel_index(np.argmin(k_results), k_results.shape)
                    magron_test_perp = k_results[min_idx]
                else:
                    magron_test_perp = np.nan
            else:
                magron_test_perp = np.nan
            
            dataset_results.append({
                'k': k,
                'train_perplexity': train_perp,
                'val_perplexity': val_perp,
                'test_perplexity': test_perp,
                'magron_test_perplexity': magron_test_perp,
                'n_iter': n_iter,
                'time': train_time
            })
            
            print(f"Test perp: {test_perp:.3f} (Magron: {magron_test_perp:.3f})")
        
        results[dataset] = pd.DataFrame(dataset_results)
        
        # Save results
        output_path = OUTPUT_DIR / f"figure3_{dataset}_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(dataset_results, f)
        
        # Also save as CSV
        csv_path = OUTPUT_DIR / f"figure3_{dataset}_results.csv"
        results[dataset].to_csv(csv_path, index=False)
    
    return results

def main():
    """Run all experiments."""
    print("="*60)
    print("REPRODUCING MAGRON & FÉVOTTE (2022) EXPERIMENTS")
    print("Using nbmf_mm implementation (Chauhan 2025)")
    print("="*60)
    
    # Figure 1: Hyperparameter validation
    fig1_results = run_figure1_experiments()
    
    # Figure 2: Test performance comparison
    fig2_results = run_figure2_experiments()
    
    # Figure 3: Varying components
    fig3_results = run_figure3_experiments()
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)
    
    # Save summary
    summary = {
        'figure1': fig1_results,
        'figure2': fig2_results,
        'figure3': fig3_results
    }
    
    summary_path = OUTPUT_DIR / "all_experiments_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

if __name__ == "__main__":
    main()