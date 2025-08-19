#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for apples-to-apples NBMF-MM reproduction (our solver vs magron2022).

Figure 1: Validation α–β heatmaps at the best K (our solver).
Figure 2: Test-set cross-entropy (mean NLL) boxplots — magron2022 vs chauhan2025.
Figure 3: LastFM H comparison — align our components to magron2022 (Hungarian),
          same curated artist subset and the 2↔3 column swap used by 2022.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pyreadr
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_repo_root(start: Path | None = None) -> Path:
    here = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Could not find repository root (need 'data' and 'outputs').")

REPO_ROOT = find_repo_root()
for add in (REPO_ROOT, REPO_ROOT / "src"):
    if str(add) not in sys.path:
        sys.path.insert(0, str(add))

DATA_DIR = REPO_ROOT / "data"
OUT_MG = REPO_ROOT / "outputs" / "magron2022"
OUT_CH = REPO_ROOT / "outputs" / "chauhan2025"
FIG_DIR = OUT_CH / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["animals", "paleo", "lastfm"]

# ---------------- Figure 1: α–β heatmaps ----------------
plt.figure(figsize=(4 * len(datasets), 3.6))
for j, ds in enumerate(datasets, start=1):
    val_loader = np.load(OUT_CH / ds / "nbmf-mm_val.npz", allow_pickle=True)
    val_pplx = val_loader["val_pplx"]
    list_nfactors, list_alpha, list_beta = val_loader["list_hyper"]

    ind_k_opt, _, _ = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
    ax = plt.subplot(1, len(datasets), j)
    ax.imshow(val_pplx[ind_k_opt, :, :], aspect="auto", cmap="gray")
    ax.invert_yaxis()
    x2 = np.arange(len(list_beta))[::2]
    ax.set_xticks(x2); ax.set_xticklabels([f"{float(b):.1f}" for b in list_beta][::2])
    if j == 1:
        y2 = np.arange(len(list_alpha))[::2]
        ax.set_yticks(y2); ax.set_yticklabels([f"{float(a):.1f}" for a in list_alpha][::2])
        ax.set_ylabel(r"$\alpha$", fontsize=14)
    else:
        ax.set_yticks([])
    ax.set_xlabel(r"$\beta$", fontsize=14)
    ax.set_title(f"{ds} (K={list_nfactors[ind_k_opt]})", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure1_val_heatmaps.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass
plt.close()

# ---------------- Figure 2: test-set comparison ----------------
n_init = 10
test_pplx_all = np.full((n_init, 2, len(datasets)), np.nan)
test_time_all = np.full((n_init, 2, len(datasets)), np.nan)
test_iter_all = np.full((n_init, 2, len(datasets)), np.nan)
for d, ds in enumerate(datasets):
    mg = np.load(OUT_MG / ds / "NBMF-MM_test_init.npz", allow_pickle=True)
    ch = np.load(OUT_CH / ds / "nbmf-mm_test_init.npz", allow_pickle=True)
    test_pplx_all[:, 0, d] = mg["test_pplx"]; test_time_all[:, 0, d] = mg["test_time"]; test_iter_all[:, 0, d] = mg["test_iter"]
    test_pplx_all[:, 1, d] = ch["test_pplx"]; test_time_all[:, 1, d] = ch["test_time"]; test_iter_all[:, 1, d] = ch["test_iter"]

plt.figure(figsize=(4 * len(datasets), 3.6))
xpos = [1, 2]; labels = ["magron2022", "chauhan2025"]
for d, ds in enumerate(datasets, start=1):
    ax = plt.subplot(1, len(datasets), d)
    ax.boxplot([test_pplx_all[:, 0, d-1], test_pplx_all[:, 1, d-1]], showfliers=False, positions=xpos, widths=[0.75, 0.75])
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=25, fontsize=11)
    if d == 1: ax.set_ylabel("Cross-entropy (mean NLL)", fontsize=12)
    ax.set_title(ds, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure2_ce_comparison.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass
plt.close()

# ---------------- Figure 3: LastFM H comparison (aligned) ----------------
dataset = "lastfm"
mg_model = np.load(OUT_MG / dataset / "NBMF-MM_model.npz", allow_pickle=True)
ch_model = np.load(OUT_CH / dataset / "nbmf-mm_model.npz", allow_pickle=True)
H_mg = mg_model["H"]; H_ch = ch_model["H"]

# Align our columns to magron2022 by Hungarian on |corr|
def align_columns(H_ref: np.ndarray, H: np.ndarray) -> np.ndarray:
    C = np.corrcoef(H_ref.T, H.T)[:H_ref.shape[1], H_ref.shape[1]:]
    C = -np.abs(C)
    try:
        from scipy.optimize import linear_sum_assignment
        _, c = linear_sum_assignment(C); return H[:, c]
    except Exception:
        # greedy fallback
        k = H_ref.shape[1]; chosen, order = set(), []
        for j in range(k):
            corr = np.abs(np.corrcoef(H_ref[:, j], H, rowvar=False)[0, 1:])
            corr[list(chosen)] = -np.inf; idx = int(np.argmax(corr))
            order.append(idx); chosen.add(idx)
        return H[:, order]

H_ch = align_columns(H_mg, H_ch)
# 2022 display swaps components 2 and 3 for visualization
if H_mg.shape[1] >= 4: H_mg[:, [2, 3]] = H_mg[:, [3, 2]]
if H_ch.shape[1] >= 4: H_ch[:, [2, 3]] = H_ch[:, [3, 2]]

# Curated subset of artists as in 2022 display_results.py
plot_range = np.concatenate((np.arange(120, 130), np.arange(184, 199)), axis=0)
plot_range = plot_range[[4, 5, 1, 0, 3, 14, 15, 2, 6, 7, 8, 9, 11, 12, 17, 18, 19, 10, 13, 16, 20, 21, 22, 23, 24]]
labels_plot = np.array(pyreadr.read_r(DATA_DIR / f"{dataset}.rda")[dataset].columns)[plot_range]
H_mg_plot = H_mg[plot_range, :]; H_ch_plot = H_ch[plot_range, :]

ypos = np.arange(len(labels_plot))
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(1, 2, 1); ax1.imshow(H_mg_plot, aspect="auto", cmap="binary"); ax1.set_yticks(ypos); ax1.set_yticklabels(labels_plot, fontsize=11); ax1.set_xticks([]); ax1.set_title("magron2022 (NBMF-MM)", fontsize=14)
ax2 = plt.subplot(1, 2, 2); ax2.imshow(H_ch_plot, aspect="auto", cmap="binary"); ax2.set_yticks([]); ax2.set_xticks([]); ax2.set_title("chauhan2025 (nbmf-mm)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure3_lastfm_H_compare.png", dpi=150, bbox_inches="tight")
try: plt.show()
except Exception: pass
plt.close()

print(f"\nFigures saved under: {FIG_DIR.resolve()}")
