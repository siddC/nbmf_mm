#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Magron & Févotte (2022) NBMF-MM experiments using our nbmf-mm solver,
STRICTLY matching the original protocol and splits.

- Loads animals/paleo/lastfm from data/*.rda
- Loads *fixed* magron2022 splits from data/magron2022/<dataset>_split.npz
- Grid-search: K in {2,4,8,16}, alpha,beta in linspace(1.0,3.0,11)  (paper/2022 repo)
- Orientation: Beta prior on H, simplex (Dirichlet) on W  -> orientation='beta-dir'
- Metric: Bernoulli mean negative log-likelihood (cross-entropy), identical to 2022
- Outputs under outputs/chauhan2025/<dataset>/ with 2022-compatible keys

Paper & original code:
  - https://arxiv.org/abs/2204.09741  (Magron & Févotte, 2022)  # apples-to-apples target
  - https://github.com/magronp/NMF-binary                     # original scripts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pyreadr


# ---------------------------
# Paths & imports
# ---------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    here = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Could not find repository root (no 'data' and 'outputs').")


REPO_ROOT = find_repo_root()
for add in (REPO_ROOT, REPO_ROOT / "src"):
    if str(add) not in sys.path:
        sys.path.insert(0, str(add))

from nbmf_mm import NBMF  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "magron2022"          # use the *original* splits
OUT_ROOT = REPO_ROOT / "outputs" / "chauhan2025"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------
# Metrics & fitting
# ---------------------------
def bernoulli_cross_entropy(
    X: np.ndarray,
    P: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-9,
) -> float:
    """Mean Bernoulli negative log-likelihood over observed entries."""
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
    orientation: str = "beta-dir",   # paper’s orientation: Beta on H, simplex on W
    max_iter: int = 2000,
    tol: float = 1e-8,
    seed: int = 0,
    use_numexpr: bool = True,
    projection_method: str = "duchi",
    projection_backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Fit NBMF-MM and return (W, H_featxcomp, Xhat, time, n_iter)."""
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
    tot = time.perf_counter() - t0

    W = model.W_
    H = model.components_.T           # store as (n_features x n_components), like 2022 files
    Xhat = model.inverse_transform(W)

    # robust n_iter
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None:
        hist = getattr(model, "objective_history_", None)
        n_iter = len(hist) if hist is not None else np.nan

    return W, H, Xhat, float(tot), int(n_iter) if n_iter == n_iter else np.nan  # keep np.nan for missing


# ---------------------------
# Data I/O
# ---------------------------
def load_dataset(dataset_name: str):
    """Load binary matrix X and magron2022 splits."""
    data_file = DATA_DIR / f"{dataset_name}.rda"
    split_file = SPLIT_DIR / f"{dataset_name}_split.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset: {data_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file} (from magron2022)")

    X = pyreadr.read_r(str(data_file))[dataset_name].to_numpy(dtype=float)
    splits = np.load(split_file)
    return X, splits["train_mask"].astype(float), splits["val_mask"].astype(float), splits["test_mask"].astype(float)


# ---------------------------
# Training loops
# ---------------------------
def run_validation_grid(
    dataset_name: str,
    K_grid: list[int],
    alpha_grid: list[float],
    beta_grid: list[float],
    *,
    seed: int = 12345,
):
    """Single‑init validation grid (as in 2022 repo)."""
    X, train_mask, val_mask, _ = load_dataset(dataset_name)

    val = np.full((len(K_grid), len(alpha_grid), len(beta_grid)), np.inf)
    best = {"score": np.inf, "K": None, "alpha": None, "beta": None}
    best_model = {}

    for i, K in enumerate(K_grid):
        for j, a in enumerate(alpha_grid):
            for k, b in enumerate(beta_grid):
                print(f"  {dataset_name}: K={K}, α={a:.1f}, β={b:.1f}")
                W, H, Xhat, tot_time, n_iter = fit_nbmf_mm(
                    X, train_mask, K, a, b, seed=((seed * 10007) + 97*K + 3*j + 5*k) % (2**32-1)
                )
                ce = bernoulli_cross_entropy(X, Xhat, mask=val_mask)
                val[i, j, k] = ce

                if ce < best["score"]:
                    best.update({"score": ce, "K": K, "alpha": a, "beta": b})
                    best_model = {"W": W, "H": H, "Xhat": Xhat, "time": tot_time, "iters": n_iter}

    return val, best, best_model


def evaluate_test_multi_init(
    dataset_name: str,
    best_params: dict,
    *,
    n_init: int = 10,
    seed: int = 12345,
):
    """Evaluate at chosen hyper‑params with many random inits (as in 2022)."""
    X, train_mask, _, test_mask = load_dataset(dataset_name)

    test_scores, test_times, test_iters = [], [], []
    for i in range(n_init):
        W, H, Xhat, tot_time, n_iter = fit_nbmf_mm(
            X, train_mask,
            best_params["K"], best_params["alpha"], best_params["beta"],
            seed=(seed + i) % (2**32-1)
        )
        ce = bernoulli_cross_entropy(X, Xhat, mask=test_mask)
        test_scores.append(ce)
        test_times.append(tot_time)
        test_iters.append(n_iter)

    return np.array(test_scores), np.array(test_times), np.array(test_iters)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    np.random.seed(12345)
    ensure_dir(OUT_ROOT)

    datasets = ["animals", "paleo", "lastfm"]

    # Hyperparameter grids EXACTLY as in 2022 main_script.py
    K_grid = [2, 4, 8, 16]
    alpha_beta = np.linspace(1.0, 3.0, 11).tolist()

    for ds in datasets:
        print(f"\n==== {ds.upper()} ====")
        out_dir = OUT_ROOT / ds
        ensure_dir(out_dir)

        # Validation grid
        val_scores, best, best_model = run_validation_grid(
            ds, K_grid, alpha_beta, alpha_beta, seed=12345
        )

        # Save validation cube (same field names as 2022)
        np.savez(
            out_dir / "nbmf-mm_val.npz",
            val_pplx=val_scores,
            list_hyper=np.array([K_grid, alpha_beta, alpha_beta], dtype=object),
        )

        # Save best model with 2022‑compatible keys
        np.savez(
            out_dir / "nbmf-mm_model.npz",
            W=best_model["W"],
            H=best_model["H"],
            Y_hat=best_model["Xhat"],
            hyper_params=(int(best["K"]), float(best["alpha"]), float(best["beta"])),
            time=float(best_model["time"]),
            loss=None,
            iters=best_model["iters"],
            best_params=np.array(best, dtype=object),
        )
        print(f"Best (K, α, β) = ({best['K']}, {best['alpha']:.1f}, {best['beta']:.1f}); "
              f"val CE = {best['score']:.6f}")

        # Test‑set multi‑init evaluation
        test_scores, test_times, test_iters = evaluate_test_multi_init(ds, best, n_init=10, seed=12345)
        np.savez(
            out_dir / "nbmf-mm_test_init.npz",
            test_pplx=test_scores,   # 2022 scripts name it 'perplexity' but it's mean NLL
            test_time=test_times,
            test_iter=test_iters,
            best_params=np.array(best, dtype=object),
        )

    print("\nDone. Results in outputs/chauhan2025/<dataset>/")
