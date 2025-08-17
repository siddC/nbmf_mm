#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the figures analogous to Magron & Févotte (2022) Figs. 1-3 from the
artifacts produced by reproduct_magron2022.py.

Outputs
-------
outputs/magron2022/fig_val_grids.png   # α–β validation perplexity, best K per dataset
outputs/magron2022/fig_test_boxplots.png
outputs/magron2022/fig_lastfm_H.png

Notes
-----
- Figure 1: Validation perplexity maps over (α, β) for the selected rank K*.
- Figure 2: Test perplexity boxplots over 10 seeds (NBMF-EM, NBMF-MM, logPCA).
- Figure 3: Heatmap of H (components) on lastfm with best (K*, α*, β*).

Paper and repo for reference:
  - https://arxiv.org/abs/2204.09741
  - https://github.com/magronp/NMF-binary
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def _load_val_grid(ds_dir: Path):
    z = np.load(ds_dir / "val_grid.npz", allow_pickle=False)
    scores = z["scores"]  # shape: (len(K_grid), len(alpha_grid)*len(beta_grid), 4)
    K_grid = z["K_grid"]
    alpha_grid = z["alpha_grid"]
    beta_grid = z["beta_grid"]
    bestK = int(z["bestK"])
    bestA = float(z["bestA"])
    bestB = float(z["bestB"])
    # Build a dict: per K, a matrix [len(alpha), len(beta)] of perplexities
    mats = {}
    A, B = len(alpha_grid), len(beta_grid)
    for i, K in enumerate(K_grid):
        mat = np.zeros((A, B), dtype=float)
        for j in range(A * B):
            k, a, b, v = scores[i, j]
            ia = np.where(np.isclose(alpha_grid, a))[0][0]
            ib = np.where(np.isclose(beta_grid, b))[0][0]
            mat[ia, ib] = v
        mats[int(K)] = mat
    return mats, K_grid, alpha_grid, beta_grid, bestK, bestA, bestB


def _load_test_csv(ds_dir: Path):
    path = ds_dir / "test_results.csv"
    tbl = {}
    with open(path, "r") as f:
        header = next(f)
        for line in f:
            dataset, method, K, alpha, beta, seed, test_perplexity, elapsed_s = line.strip().split(",")
            tbl.setdefault(method, []).append(float(test_perplexity))
    return tbl


def plot_val_grids(outdir: Path, root: Path, datasets=("animals", "paleo", "lastfm")):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    for ax, ds in zip(axs, datasets):
        ds_dir = root / ds
        mats, K_grid, alpha_grid, beta_grid, bestK, bestA, bestB = _load_val_grid(ds_dir)
        M = mats[bestK]
        im = ax.imshow(M, origin="lower", aspect="auto",
                       extent=[beta_grid[0], beta_grid[-1], alpha_grid[0], alpha_grid[-1]])
        ax.set_title(ds)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha$")
        # mark best (α*, β*)
        ax.plot([bestB], [bestA], marker="o", markersize=4, linestyle="None")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Validation perplexity (lower is better)")
    fig.suptitle("Validation perplexity over (α, β) at selected rank $K^*$", y=1.02, fontsize=12)
    fig.savefig(outdir / "fig_val_grids.png", dpi=200)
    plt.close(fig)


def plot_test_boxplots(outdir: Path, root: Path, datasets=("animals", "paleo", "lastfm")):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    for ax, ds in zip(axs, datasets):
        ds_dir = root / ds
        tbl = _load_test_csv(ds_dir)
        labels = []
        data = []
        for method in ("NBMF-EM", "NBMF-MM", "logPCA"):
            if method in tbl:
                labels.append(method)
                data.append(tbl[method])
        bp = ax.boxplot(data, labels=labels, showmeans=False)
        ax.set_title(ds)
        ax.set_ylabel("Test perplexity")
    fig.suptitle("Test perplexity (10 random initializations)", y=1.02, fontsize=12)
    fig.savefig(outdir / "fig_test_boxplots.png", dpi=200)
    plt.close(fig)


def plot_lastfm_H(outdir: Path, root: Path):
    ds_dir = root / "lastfm"
    H_path = ds_dir / "H_lastfm_best.npy"
    if not H_path.exists():
        print("[warn] H_lastfm_best.npy not found — skip Fig. 3.")
        return
    H = np.load(H_path)
    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    im = ax.imshow(H, aspect="auto", origin="lower")
    ax.set_xlabel("features (N)")
    ax.set_ylabel("components (K)")
    ax.set_title("NBMF-MM components (H) on lastfm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("membership weight")
    fig.savefig(outdir / "fig_lastfm_H.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render figures from reproduced NBMF-MM experiments.")
    ap.add_argument("--root", type=Path, default=Path("outputs/magron2022"),
                    help="Root directory produced by reproduct_magron2022.py")
    args = ap.parse_args()
    outdir = Path(args.root)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_val_grids(outdir, outdir)
    plot_test_boxplots(outdir, outdir)
    plot_lastfm_H(outdir, outdir)
    print(f"Figures saved under {outdir}")


if __name__ == "__main__":
    main()
