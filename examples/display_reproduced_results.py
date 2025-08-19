#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for NBMF-MM reproduction experiments.

This script creates publication-quality figures comparing our NBMF-MM implementation
with the original Magron & Févotte (2022) results. It generates three main figures:

Figure 1: Validation heatmaps
    Shows cross-entropy scores across hyperparameter grids (α, β) for each dataset.
    Each panel corresponds to the optimal rank K found during validation.

Figure 2: Test performance comparison
    Boxplots comparing test cross-entropy between original (magron2022) and our
    reproduction (chauhan2025) across multiple random initializations.

Figure 3: Component analysis (LastFM)
    Heatmap comparison of learned components H between original and reproduction
    for the LastFM dataset, showing qualitative agreement.

Inputs
------
- outputs/chauhan2025/: Results from our reproduction experiments
- outputs/magron2022/: Original results from Magron & Févotte (2022)

Outputs
-------
- outputs/chauhan2025/figures/: Generated figures in PNG format

References
----------
.. [1] P. Magron & C. Févotte (2022). A majorization-minimization algorithm for
       nonnegative binary matrix factorization. IEEE Signal Processing Letters.
       https://doi.org/10.1109/LSP.2022.3185610
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib

# Prefer interactive backend if available; otherwise fall back to Agg
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


# Setup paths
REPO_ROOT = find_repo_root()
for add in (REPO_ROOT, REPO_ROOT / "src"):
    if str(add) not in sys.path:
        sys.path.insert(0, str(add))

DATA_DIR = REPO_ROOT / "data"
OUT_MG = REPO_ROOT / "outputs" / "magron2022"
OUT_CH = REPO_ROOT / "outputs" / "chauhan2025"
FIG_DIR = OUT_CH / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
datasets = ["animals", "paleo", "lastfm"]
n_datasets = len(datasets)


def create_validation_heatmaps():
    """
    Create Figure 1: Validation cross-entropy heatmaps.
    
    Shows α-β heatmaps for the optimal rank K for each dataset.
    """
    print("Creating Figure 1: Validation heatmaps...")
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 3.6))
    if n_datasets == 1:
        axes = [axes]
    
    for j, dataset in enumerate(datasets):
        # Load validation results
        val_file = OUT_CH / dataset / "nbmf-mm_val.npz"
        if not val_file.exists():
            print(f"Warning: {val_file} not found, skipping {dataset}")
            continue
            
        val_data = np.load(val_file, allow_pickle=True)
        val_pplx = val_data["val_pplx"]  # Cross-entropy values
        list_hyper = val_data["list_hyper"]
        list_nfactors, list_alpha, list_beta = list_hyper
        
        # Find optimal K (minimum cross-entropy across all hyperparameters)
        ind_k_opt, _, _ = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
        
        # Create heatmap
        ax = axes[j]
        im = ax.imshow(val_pplx[ind_k_opt, :, :], aspect="auto", cmap="gray")
        ax.invert_yaxis()
        
        # Set tick labels
        xpositions = np.arange(len(list_beta))[::2]
        ax.set_xticks(xpositions)
        ax.set_xticklabels([f"{float(b):.1f}" for b in list_beta][::2])
        ax.set_xlabel(r"$\beta$", fontsize=14)
        
        if j == 0:
            ypositions = np.arange(len(list_alpha))[::2]
            ax.set_yticks(ypositions)
            ax.set_yticklabels([f"{float(a):.1f}" for a in list_alpha][::2])
            ax.set_ylabel(r"$\alpha$", fontsize=14)
        else:
            ax.set_yticks([])
        
        ax.set_title(f"{dataset} (K={list_nfactors[ind_k_opt]})", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_val_heatmaps.png", dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def create_test_comparison():
    """
    Create Figure 2: Test performance comparison.
    
    Boxplots comparing cross-entropy between original and reproduction results.
    """
    print("Creating Figure 2: Test performance comparison...")
    
    models = ["magron2022", "chauhan2025"]
    n_models = len(models)
    n_init = 10
    
    # Initialize results arrays
    test_pplx_all = np.full((n_init, n_models, n_datasets), np.nan)
    test_time_all = np.full((n_init, n_models, n_datasets), np.nan)
    test_iter_all = np.full((n_init, n_models, n_datasets), np.nan)
    
    # Load results for each dataset
    for d, dataset in enumerate(datasets):
        # Original results (magron2022)
        mg_file = OUT_MG / dataset / "NBMF-MM_test_init.npz"
        if mg_file.exists():
            mg_data = np.load(mg_file, allow_pickle=True)
            test_pplx_all[:, 0, d] = mg_data["test_pplx"]
            test_time_all[:, 0, d] = mg_data["test_time"]
            test_iter_all[:, 0, d] = mg_data["test_iter"]
        else:
            print(f"Warning: {mg_file} not found")
        
        # Our reproduction results (chauhan2025)
        ch_file = OUT_CH / dataset / "nbmf-mm_test_init.npz"
        if ch_file.exists():
            ch_data = np.load(ch_file, allow_pickle=True)
            test_pplx_all[:, 1, d] = ch_data["test_pplx"]
            test_time_all[:, 1, d] = ch_data["test_time"]
            test_iter_all[:, 1, d] = ch_data["test_iter"]
        else:
            print(f"Warning: {ch_file} not found")
    
    # Create boxplots
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 3.6))
    if n_datasets == 1:
        axes = [axes]
    
    xpos = [1, 2]
    for d, dataset in enumerate(datasets):
        ax = axes[d]
        
        # Filter out NaN values
        data1 = test_pplx_all[:, 0, d]
        data2 = test_pplx_all[:, 1, d]
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if len(data1) > 0 and len(data2) > 0:
            ax.boxplot([data1, data2], positions=xpos)
            ax.set_xticks(xpos)
            ax.set_xticklabels(["Original", "Ours"])
            ax.set_ylabel("Test Cross-entropy", fontsize=12)
            ax.set_title(dataset, fontsize=14)
            
            # Add mean values as text
            mean1, mean2 = np.mean(data1), np.mean(data2)
            ax.text(0.02, 0.98, f"Original: {mean1:.4f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(0.02, 0.90, f"Ours: {mean2:.4f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{dataset} (no data)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_test_comparison.png", dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def create_component_comparison():
    """
    Create Figure 3: Component comparison for LastFM dataset.
    
    Shows heatmaps of learned components H from both original and reproduction.
    """
    print("Creating Figure 3: Component comparison (LastFM)...")
    
    # Load best models for LastFM
    mg_model_file = OUT_MG / "lastfm" / "NBMF-MM_model.npz"
    ch_model_file = OUT_CH / "lastfm" / "nbmf-mm_model.npz"
    
    if not mg_model_file.exists() or not ch_model_file.exists():
        print("Warning: Model files not found, skipping component comparison")
        return
    
    # Load components
    mg_data = np.load(mg_model_file, allow_pickle=True)
    ch_data = np.load(ch_model_file, allow_pickle=True)
    
    H_mg = mg_data["H"]  # Original components
    H_ch = ch_data["H"]  # Our components
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original components
    im1 = ax1.imshow(H_mg, aspect='auto', cmap='viridis')
    ax1.set_title("Original (Magron & Févotte 2022)", fontsize=14)
    ax1.set_xlabel("Components", fontsize=12)
    ax1.set_ylabel("Features", fontsize=12)
    plt.colorbar(im1, ax=ax1)
    
    # Our components
    im2 = ax2.imshow(H_ch, aspect='auto', cmap='viridis')
    ax2.set_title("Our Reproduction", fontsize=14)
    ax2.set_xlabel("Components", fontsize=12)
    ax2.set_ylabel("Features", fontsize=12)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_component_comparison.png", dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def main():
    """Generate all figures for the reproduction experiment."""
    print("NBMF-MM Figure Generation")
    print("=" * 30)
    
    # Set matplotlib style for publication-quality figures
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })
    
    # Generate figures
    create_validation_heatmaps()
    create_test_comparison()
    create_component_comparison()
    
    print(f"\nFigures saved to: {FIG_DIR}")
    print("\nGenerated figures:")
    print("1. figure1_val_heatmaps.png - Validation cross-entropy heatmaps")
    print("2. figure2_test_comparison.png - Test performance comparison")
    print("3. figure3_component_comparison.png - Component analysis (LastFM)")
    
    # Compare with original figures if available
    original_fig_dir = OUT_MG / "figures"
    if original_fig_dir.exists():
        print(f"\nOriginal figures available at: {original_fig_dir}")
        print("Compare the generated figures with the original ones for validation.")


if __name__ == "__main__":
    main()
