#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Magron & Févotte (2022) experiments using our nbmf-mm implementation,
*with the exact same train/val/test splits* stored under data/magron2022/.

- Loads animals/paleo/lastfm from data/*.rda (as in the 2022 repo).
- Loads splits from data/magron2022/<dataset>_split.npz (no new splits are created).
- Runs NBMF-MM grid search over (K, alpha, beta) using our class (orientation='dir-beta').
- Saves outputs under outputs/chauhan2025/<dataset>/:
    - nbmf-mm_val.npz          (validation cube)
    - nbmf-mm_model.npz        (best model on validation; contains W, H, Y_hat, hyper_params, time, iters)
    - nbmf-mm_test_init.npz    (test perplexities over multiple random inits at best hyper-params)

This script mirrors the flow of the original NMF-binary/main_script.py, but replaces its solver
with nbmf_mm.NBMF. (Original script cited.)  :contentReference[oaicite:2]{index=2}
Paper & original code:  :contentReference[oaicite:3]{index=3}
"""
from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import pyreadr

from nbmf_mm import NBMF  # our implementation


# ---------------------------
# Path helpers
# ---------------------------
def find_repo_root(start: Path | None = None) -> Path:
    """
    Resolve the repository root so this script can live in nbmf_mm/examples/.
    We look upwards for a directory containing 'data' and 'outputs'.
    """
    here = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    # Fallback: two levels up from examples/ -> repo root
    return here.parents[2]


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "magron2022"         # <-- use 2022 splits, do NOT write data/chauhan2025
OUT_ROOT = REPO_ROOT / "outputs" / "chauhan2025"


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def bernoulli_perplexity(X: np.ndarray, P: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-9) -> float:
    """
    Perplexity = exp(NLL / (#observed)) for the mean-parameterized Bernoulli model.
    """
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    P = np.clip(P, eps, 1.0 - eps)
    nll = -(mask * (X * np.log(P) + (1 - X) * np.log(1 - P))).sum()
    denom = float(mask.sum()) if float(mask.sum()) > 0 else 1.0
    return float(np.exp(nll / denom))


def fit_nbmf_mm(
    X: np.ndarray,
    train_mask: np.ndarray,
    n_components: int,
    alpha: float,
    beta: float,
    *,
    orientation: str = "dir-beta",
    max_iter: int = 2000,
    tol: float = 1e-8,
    seed: int = 0,
    use_numexpr: bool = True,
    projection_method: str = "duchi",
    projection_backend: str = "auto",
):
    """
    Fit NBMF-MM with given hyperparameters. Returns (W, H_featxcomp, Xhat, tot_time, iters).
    H is returned as (n_features x n_components) to match the 2022 npz files.
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
    tot_time = time.perf_counter() - t0

    W = model.W_                                # (n_samples, n_components)
    H = model.components_.T                      # (n_features, n_components)
    Xhat = model.inverse_transform(W)            # reconstruction probs (n_samples, n_features)
    iters = getattr(model, "n_iter_", None)
    if iters is None:
        hist = getattr(model, "objective_history_", None)
        iters = len(hist) if hist is not None else np.nan
    return W, H, Xhat, tot_time, iters


def training_with_validation(
    X: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    list_nfactors: list[int],
    list_alpha: list[float],
    list_beta: list[float],
    out_dir_dataset: str | Path,
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
    base_seed: int = 12345,
):
    """
    Grid over K x alpha x beta; save best model (by val perplexity) and the full validation cube.
    Mirrors the structure of the original main_script's NBMF-MM branch.  :contentReference[oaicite:4]{index=4}
    """
    nk, na, nb = len(list_nfactors), len(list_alpha), len(list_beta)
    val_pplx = np.zeros((nk, na, nb))
    opt_pplx = np.inf

    counter, total = 1, nk * na * nb
    for ik, K in enumerate(list_nfactors):
        for ia, a in enumerate(list_alpha):
            for ib, b in enumerate(list_beta):
                print(f"--- Hyperparameters {counter}/{total}: K={K}, alpha={a:.3g}, beta={b:.3g}")

                seed = (base_seed * 10_007 + 97 * K + 3 * ia + 5 * ib) % (2**32 - 1)
                W, H, Xhat, tot_time, iters = fit_nbmf_mm(
                    X, train_mask, K, a, b, max_iter=max_iter, tol=tol, seed=seed
                )
                perplx = bernoulli_perplexity(X, Xhat, mask=val_mask)
                print(f"Val perplexity: {perplx:.4f}")
                val_pplx[ik, ia, ib] = perplx

                if perplx < opt_pplx:
                    np.savez(
                        Path(out_dir_dataset) / "nbmf-mm_model.npz",
                        W=W,
                        H=H,
                        Y_hat=Xhat,
                        hyper_params=(int(K), float(a), float(b)),
                        time=tot_time,
                        loss=None,
                        iters=iters,
                    )
                    opt_pplx = perplx
                counter += 1

    np.savez(
        Path(out_dir_dataset) / "nbmf-mm_val.npz",
        val_pplx=np.array(val_pplx),
        list_hyper=np.array([list_nfactors, list_alpha, list_beta], dtype=object),
    )


def train_test_init(
    X: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    hyper_params,
    out_dir_dataset: str | Path,
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
    n_init: int = 10,
    base_seed: int = 12345,
):
    """
    Fix hyper-params and evaluate test-time perplexity over many random restarts,
    as in the 2022 workflow.  :contentReference[oaicite:5]{index=5}
    """
    K, a, b = int(hyper_params[0]), float(hyper_params[1]), float(hyper_params[2])
    test_pplx = np.zeros((n_init,), dtype=float)
    test_time = np.zeros((n_init,), dtype=float)
    test_iter = np.zeros((n_init,), dtype=float)

    for i in range(n_init):
        seed = (base_seed + i) % (2**32 - 1)
        t0 = time.perf_counter()
        W, H, Xhat, _, iters = fit_nbmf_mm(
            X, train_mask, K, a, b, max_iter=max_iter, tol=tol, seed=seed
        )
        dt = time.perf_counter() - t0
        test_pplx[i] = bernoulli_perplexity(X, Xhat, mask=test_mask)
        test_time[i] = dt
        test_iter[i] = iters if iters is not None else np.nan

    np.savez(
        Path(out_dir_dataset) / "nbmf-mm_test_init.npz",
        test_pplx=test_pplx,
        test_time=test_time,
        test_iter=test_iter,
    )


# ---------------------------
# Main driver
# ---------------------------
if __name__ == "__main__":
    np.random.seed(12345)

    datasets = ["animals", "paleo", "lastfm"]
    max_iter = 2000
    tol = 1e-8
    n_init = 10
    list_nfactors = [2, 4, 8, 16]
    list_alpha = list_beta = list(np.linspace(1.0, 3.0, 11))

    for ds in datasets:
        print(f"\n==== Dataset: {ds} ====")

        # Load data and the *fixed* 2022 splits
        df = pyreadr.read_r(DATA_DIR / f"{ds}.rda")[ds]
        X = df.to_numpy(dtype=float)

        split_file = SPLIT_DIR / f"{ds}_split.npz"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Expected split at {split_file}. Please run the original 2022 pipeline or copy the file."
            )
        with np.load(split_file) as sp:
            train_mask = sp["train_mask"].astype(float)
            val_mask = sp["val_mask"].astype(float)
            test_mask = sp["test_mask"].astype(float)

        # Output dir for this dataset
        out_dir_ds = OUT_ROOT / ds
        ensure_dir(out_dir_ds)

        # Validation grid + save best model & val cube
        training_with_validation(
            X, train_mask, val_mask, list_nfactors, list_alpha, list_beta,
            out_dir_ds, max_iter=max_iter, tol=tol, base_seed=12345
        )

        # Test-time multi-init at the chosen hyper-params (load back from the saved model file)
        best = np.load(out_dir_ds / "nbmf-mm_model.npz", allow_pickle=True)["hyper_params"]
        train_test_init(
            X, train_mask, test_mask, best, out_dir_ds,
            max_iter=max_iter, tol=tol, n_init=n_init, base_seed=12345
        )

    print("\nAll done. Outputs are under outputs/chauhan2025 and figures will be created by display_reproduced_results.py.")
