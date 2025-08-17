# display_reproduced_results.py
# -*- coding: utf-8 -*-
"""
Display validation and test results in the style of Magron & Févotte (2022).

- Plot 1: validation perplexity heatmaps (NBMF-MM) at optimal K per dataset.
- Plot 2: test set perplexity boxplots across models per dataset.
- Plot 3: (optional) H visualization for lastfm comparing NBMF-MM and logPCA
          (shown only if both models are available).
"""

import os
import numpy as np
import pyreadr
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

data_dir = "data/"
out_dir  = "outputs/"
datasets = ["animals", "paleo", "lastfm"]
models   = ["NBMF-EM", "NBMF-MM", "logPCA"]
n_datasets = len(datasets)

# ---- Plot 1: α×β heatmaps at best K ----
plt.figure(figsize=(4 * n_datasets, 3.2))
for j, ds in enumerate(datasets):
    path = os.path.join(out_dir, ds, "NBMF-MM_val.npz")
    if not os.path.exists(path):
        continue
    z = np.load(path, allow_pickle=True)
    val_pplx = z["val_pplx"]; list_nfactors, list_alpha, list_beta = z["list_hyper"]
    ind_k_opt, ind_a_opt, ind_b_opt = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
    ax = plt.subplot(1, n_datasets, j+1)
    im = ax.imshow(val_pplx[ind_k_opt, :, :], aspect='auto', cmap='gray')
    ax.invert_yaxis()
    xticks = np.arange(len(list_beta))[::2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{float(b):.1f}" for b in list_beta][::2])
    if j == 0:
        yticks = np.arange(len(list_alpha))[::2]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{float(a):.1f}" for a in list_alpha][::2])
        ax.set_ylabel(r'$\alpha$')
    else:
        ax.set_yticks([])
    ax.set_xlabel(r'$\beta$')
    ax.set_title(ds, fontsize=14)
plt.tight_layout()
plt.show()

# ---- Plot 2: test perplexity boxplots ----
plt.figure(figsize=(4 * n_datasets, 3.2))
labels = ["NBMF-EM", "NBMF-MM", "logPCA"]
for j, ds in enumerate(datasets):
    ax = plt.subplot(1, n_datasets, j+1)
    bp_data = []
    xt = []
    for m in models:
        path = os.path.join(out_dir, ds, f"{m}_test_init.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            bp_data.append(d["test_pplx"])
            xt.append(m)
    if bp_data:
        ax.boxplot(bp_data, showfliers=False, positions=range(1, len(bp_data)+1), widths=0.7)
        ax.set_xticks(range(1, len(bp_data)+1))
        ax.set_xticklabels(xt, rotation=45, ha='right')
    ax.set_title(ds, fontsize=14)
    if j == 0:
        ax.set_ylabel("Perplexity")
plt.tight_layout()
plt.show()

# ---- Plot 3: H visualization for lastfm (optional) ----
# Only if both NBMF‑MM_model.npz and logPCA_model.npz exist
ds = "lastfm"
mm_path  = os.path.join(out_dir, ds, "NBMF-MM_model.npz")
lp_path  = os.path.join(out_dir, ds, "logPCA_model.npz")
if os.path.exists(mm_path) and os.path.exists(lp_path):
    H_mm = np.load(mm_path, allow_pickle=True)["H"]
    H_lp = np.load(lp_path, allow_pickle=True)["H"]
    # swap example components as in authors' script (optional)
    # H_mm[:, [2,3]] = H_mm[:, [3,2]]
    # H_lp[:, [2,3]] = H_lp[:, [3,2]]

    data = pyreadr.read_r(os.path.join(data_dir, f"{ds}.rda"))[ds]
    labels_plot = np.array(data.columns)

    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(H_mm, aspect='auto', cmap='binary')
    ax1.set_title("NBMF‑MM"); ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(H_lp, aspect='auto', cmap='binary')
    ax2.set_title("logPCA"); ax2.set_xticks([]); ax2.set_yticks([])
    plt.tight_layout(); plt.show()
