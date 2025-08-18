# display_reproduced_results.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pyreadr
import matplotlib

# Prefer an interactive backend if available; otherwise fall back to Agg
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Paths & constants
data_dir = Path("data")
data_ch_dir = data_dir / "chauhan2025"
data_mg_dir = data_dir / "magron2022"

out_root_mg = Path("outputs") / "magron2022"
out_root_ch = Path("outputs") / "chauhan2025"
fig_dir = out_root_ch / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

datasets = ["animals", "paleo", "lastfm"]
n_datasets = len(datasets)


# ---------------------------
# Figure 1: Validation heatmaps (our NBMF-MM)
# ---------------------------
plt.figure(figsize=(4 * n_datasets, 3.6))
for j, ds in enumerate(datasets, start=1):
    val_loader = np.load(out_root_ch / ds / "nbmf-mm_val.npz", allow_pickle=True)
    val_pplx = val_loader["val_pplx"]
    list_nfactors, list_alpha, list_beta = val_loader["list_hyper"]

    # Find the (K, alpha, beta) of minimum perplexity and show the alpha/beta heatmap for that K
    ind_k_opt, ind_alpha_opt, ind_beta_opt = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
    ax = plt.subplot(1, n_datasets, j)
    im = ax.imshow(val_pplx[ind_k_opt, :, :], aspect="auto", cmap="gray")
    ax.invert_yaxis()

    xpositions = np.arange(len(list_beta))[::2]
    ax.set_xticks(xpositions)
    ax.set_xticklabels([f"{float(b):.1f}" for b in list_beta][::2])
    ax.set_xlabel(r"$\beta$", fontsize=14)

    if j == 1:
        ypositions = np.arange(len(list_alpha))[::2]
        ax.set_yticks(ypositions)
        ax.set_yticklabels([f"{float(a):.1f}" for a in list_alpha][::2])
        ax.set_ylabel(r"$\alpha$", fontsize=14)
    else:
        ax.set_yticks([])

    ax.set_title(ds, fontsize=14)

plt.tight_layout()
plt.savefig(fig_dir / "figure1_val_heatmaps.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass


# ---------------------------
# Figure 2: Perplexity comparison (magron2022 vs chauhan2025)
# ---------------------------
models = ["magron2022", "chauhan2025"]
n_models = len(models)
n_init = 10

test_pplx_all = np.full((n_init, n_models, n_datasets), np.nan)
test_time_all = np.full((n_init, n_models, n_datasets), np.nan)
test_iter_all = np.full((n_init, n_models, n_datasets), np.nan)

for d, ds in enumerate(datasets):
    # old (magron2022)
    mg = np.load(out_root_mg / ds / "NBMF-MM_test_init.npz", allow_pickle=True)
    test_pplx_all[:, 0, d] = mg["test_pplx"]
    test_time_all[:, 0, d] = mg["test_time"]
    test_iter_all[:, 0, d] = mg["test_iter"]

    # new (chauhan2025)
    ch = np.load(out_root_ch / ds / "nbmf-mm_test_init.npz", allow_pickle=True)
    test_pplx_all[:, 1, d] = ch["test_pplx"]
    test_time_all[:, 1, d] = ch["test_time"]
    test_iter_all[:, 1, d] = ch["test_iter"]

# Boxplots
plt.figure(figsize=(4 * n_datasets, 3.6))
xpos = [1, 2]
for d, ds in enumerate(datasets, start=1):
    ax = plt.subplot(1, n_datasets, d)
    ax.boxplot(
        [test_pplx_all[:, 0, d-1], test_pplx_all[:, 1, d-1]],
        showfliers=False, positions=xpos, widths=[0.75, 0.75]
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["magron2022", "chauhan2025"], rotation=25, fontsize=11)
    if d == 1:
        ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(ds, fontsize=14)

plt.tight_layout()
plt.savefig(fig_dir / "figure2_perplexity_comparison.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass

# Print means table (optional console output)
for d, ds in enumerate(datasets):
    print(f"---- {ds}")
    print("magron2022: pplx={:.3f}, time={:.4f}, iters={:.1f}".format(
        np.nanmean(test_pplx_all[:, 0, d]),
        np.nanmean(test_time_all[:, 0, d]),
        np.nanmean(test_iter_all[:, 0, d]),
    ))
    print("chauhan2025: pplx={:.3f}, time={:.4f}, iters={:.1f}".format(
        np.nanmean(test_pplx_all[:, 1, d]),
        np.nanmean(test_time_all[:, 1, d]),
        np.nanmean(test_iter_all[:, 1, d]),
    ))


# ---------------------------
# Figure 3: H comparison on lastfm (magron2022 vs chauhan2025)
# ---------------------------
dataset = "lastfm"
# Load lastfm labels (band names) from the .rda
data_lastfm = pyreadr.read_r(data_dir / f"{dataset}.rda")[dataset]

H_mg = np.load(out_root_mg / dataset / "NBMF-MM_model.npz", allow_pickle=True)["H"]
H_ch = np.load(out_root_ch / dataset / "nbmf-mm_model.npz", allow_pickle=True)["H"]

# Optional swap for visualization parity (same trick as 2022 script). :contentReference[oaicite:12]{index=12}
if H_mg.shape[1] >= 4:
    H_mg[:, [2, 3]] = H_mg[:, [3, 2]]
if H_ch.shape[1] >= 4:
    H_ch[:, [2, 3]] = H_ch[:, [3, 2]]

# Curated subset & ordering (as in 2022 script). :contentReference[oaicite:13]{index=13}
plot_range = np.concatenate((np.arange(120, 130), np.arange(184, 199)), axis=0)
plot_range = plot_range[[4, 5, 1, 0, 3, 14, 15, 2, 6, 7, 8, 9, 11, 12, 17, 18, 19, 10, 13, 16, 20, 21, 22, 23, 24]]

labels_plot = np.array(data_lastfm.columns)[plot_range]
H_mg_plot = H_mg[plot_range, :]
H_ch_plot = H_ch[plot_range, :]
ypos = np.arange(len(labels_plot))

plt.figure(figsize=(10, 8))
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(H_mg_plot, aspect="auto", cmap="binary")
ax1.set_yticks(ypos)
ax1.set_yticklabels(labels_plot, fontsize=11)
ax1.set_xticks([])
ax1.set_title("magron2022 (NBMF-MM)", fontsize=14)

ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(H_ch_plot, aspect="auto", cmap="binary")
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_title("chauhan2025 (nbmf-mm)", fontsize=14)

plt.tight_layout()
plt.savefig(fig_dir / "figure3_lastfm_H_compare.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass

print(f"\nFigures saved under: {fig_dir.resolve()}")
