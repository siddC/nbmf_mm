#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the experiments from:
  Paul Magron & Cédric Févotte (2022), "A majorization-minimization algorithm
  for nonnegative binary matrix factorization", IEEE Signal Processing Letters.

Protocol (as in §4 of the paper):
  - Datasets: animals, paleo, lastfm. (Binary matrices.)
  - Split entries into train/val/test = 70%/15%/15%.
  - Tune hyperparameters (rank K and Beta prior α, β) using validation perplexity.
  - Report test perplexity for 10 random initializations.
  - Stopping: relative objective change < 1e-5 or max_iter = 2000.
  - Compare NBMF-MM (ours), NBMF-EM proxy (α=β=1), and optional logisticPCA baseline.

References:
  - Paper: https://arxiv.org/abs/2204.09741
  - Code template & datasets: https://github.com/magronp/NMF-binary
  - logisticPCA (R): https://github.com/andland/logisticPCA

This script targets the scikit-learn-style estimator shipped in this repo:
    from nbmf_mm import NBMF
and uses orientation="beta-dir" (Beta prior on H; row-simplex on W) to match the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Your estimator (scikit-learn style)
try:
    from nbmf_mm import NBMF
except Exception as e:
    raise RuntimeError(
        "Could not import 'NBMF' from nbmf_mm. "
        "Install the package in editable mode first: `pip install -e .`"
    ) from e


# ---------------------------- I/O utilities --------------------------------- #

def _try_import_pyreadr():
    try:
        import pyreadr  # type: ignore
        return pyreadr
    except Exception:
        return None


def _try_import_rpy2():
    try:
        import rpy2.robjects as ro  # type: ignore
        from rpy2.robjects import numpy2ri  # type: ignore
        from rpy2.robjects.packages import importr  # type: ignore
        numpy2ri.activate()
        return ro, importr
    except Exception:
        return None, None


def load_binary_matrix(path: Path) -> np.ndarray:
    """
    Load a binary matrix from .rda/.RData (prefers pyreadr, falls back to rpy2),
    or from .npz/.npy (numpy).

    Returns
    -------
    X : np.ndarray (float64), values in {0,1} or [0,1]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext in {".npz", ".npy"}:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # choose the only array or 'arr_0'
            keys = list(arr.keys())
            key = "arr_0" if "arr_0" in arr else keys[0]
            X = np.asarray(arr[key], dtype=float)
        else:
            X = np.asarray(arr, dtype=float)
        return X

    if ext in {".rda", ".rdata"}:
        # Try pyreadr first
        pyreadr = _try_import_pyreadr()
        if pyreadr is not None:
            result = pyreadr.read_r(path.as_posix())
            # Pick the first matrix/data.frame-like object
            for key, obj in result.items():
                arr = np.asarray(obj)
                if arr.ndim == 2:
                    X = arr.astype(float)
                    return X
            raise ValueError(f"No 2D arrays found in {path}")
        # Fall back to rpy2
        ro, importr = _try_import_rpy2()
        if ro is not None:
            base = importr("base")
            env = ro.Environment()
            base.load(path.as_posix(), env)
            for key in env.keys():
                arr = np.array(env.find(key))
                if arr.ndim == 2:
                    return arr.astype(float)
            raise ValueError(f"No 2D arrays found in {path}")
        raise RuntimeError(
            f"Install either 'pyreadr' or 'rpy2' to read RData: {path}"
        )

    raise ValueError(f"Unsupported file type: {path.suffix}")


# ------------------------- metrics / splitting ------------------------------- #

def make_random_masks(shape: Tuple[int, int], seed: int,
                      train_frac: float = 0.70,
                      val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random entry-wise split into train/val/test masks.
    """
    M, N = shape
    rng = np.random.default_rng(seed)
    r = rng.random((M, N))
    train_thr = train_frac
    val_thr = train_frac + val_frac
    train = (r < train_thr).astype(float)
    val = ((r >= train_thr) & (r < val_thr)).astype(float)
    test = (r >= val_thr).astype(float)
    return train, val, test


def bernoulli_mean_nll(y: np.ndarray, p: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Average negative log-likelihood for Bernoulli(y | p) over masked entries.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    if mask is None:
        mask = np.ones_like(y, dtype=float)
    nobs = float(mask.sum())
    if nobs == 0.0:
        return np.nan
    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float((nll * mask).sum() / nobs)


def perplexity(y: np.ndarray, p: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Perplexity as used in the paper: exp(mean NLL over the eval set).
    """
    return float(np.exp(bernoulli_mean_nll(y, p, mask)))


# -------------------------- models / baselines ------------------------------- #

def fit_nbmf_mm(
    X: np.ndarray,
    mask_train: np.ndarray,
    K: int,
    alpha: float,
    beta: float,
    random_state: int,
    max_iter: int = 2000,
    tol: float = 1e-5,
    projection_method: str = "duchi",
    projection_backend: str = "auto",
    use_numexpr: bool = True,
) -> Tuple[NBMF, np.ndarray]:
    """
    Train NBMF-MM (orientation='beta-dir') on the training subset.
    Returns the fitted model and the full-matrix probabilities Xhat.
    """
    model = NBMF(
        n_components=K,
        orientation="beta-dir",   # Beta prior on H; W row-simplex, as in the paper
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        projection_method=projection_method,
        projection_backend=projection_backend,
        use_numexpr=use_numexpr,
        n_init=1,
    ).fit(X, mask=mask_train)

    # Full reconstruction (probabilities)
    Xhat = model.inverse_transform(model.W_)  # shape like X
    return model, Xhat


def fit_logistic_pca_predict(
    X: np.ndarray,
    K: int,
    max_iters: int = 2000,
    tol: float = 1e-5,
    m: Optional[int] = 4,
) -> Optional[np.ndarray]:
    """
    Train logistic PCA (R package 'logisticPCA') and return fitted probabilities.
    If rpy2/logisticPCA are unavailable, returns None.
    NOTE: logisticPCA does not support masked fitting. For simplicity we fit on
    the full matrix and *evaluate* on masked entries (as the paper does).
    """
    ro, importr = _try_import_rpy2()
    if ro is None:
        return None
    try:
        logpca = importr("logisticPCA")
    except Exception:
        return None

    rmat = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    # Fit LPCA
    if m is None:
        # Let the package solve for m using its internal validation trick
        fit = logpca.logisticPCA(
            rmat, k=int(K), m=0, quiet=True,
            max_iters=int(max_iters), conv_criteria=tol, main_effects=True
        )
    else:
        fit = logpca.logisticPCA(
            rmat, k=int(K), m=int(m), quiet=True,
            max_iters=int(max_iters), conv_criteria=tol, main_effects=True
        )
    # Fitted probabilities on the TRAIN matrix (we evaluate with masks)
    fitted = logpca.fitted(fit, type="response")
    return np.array(fitted, dtype=float)


# ------------------------------ experiment ----------------------------------- #

@dataclass
class GridResult:
    dataset: str
    K: int
    alpha: float
    beta: float
    val_perplexity: float
    seed: int
    elapsed_s: float


@dataclass
class TestResult:
    dataset: str
    method: str
    K: int
    alpha: float
    beta: float
    seed: int
    test_perplexity: float
    elapsed_s: float


def run_validation_grid(
    X: np.ndarray,
    masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dataset: str,
    K_grid: Iterable[int],
    alpha_grid: Iterable[float],
    beta_grid: Iterable[float],
    seed: int,
    max_iter: int,
    tol: float,
    **fit_kwargs,
) -> Tuple[Dict[str, float], Tuple[int, float, float]]:
    """
    Run a grid over (K, alpha, beta). Return:
      – scores: mapping "K|alpha|beta" -> val perplexity
      – best triple (K*, alpha*, beta*)
    """
    mask_train, mask_val, _ = masks
    scores: Dict[str, float] = {}
    best = (None, None, None)
    best_val = np.inf

    for K in K_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                t0 = time.time()
                _, Xhat = fit_nbmf_mm(
                    X, mask_train, K, alpha, beta,
                    random_state=seed, max_iter=max_iter, tol=tol, **fit_kwargs
                )
                elapsed = time.time() - t0
                vpx = perplexity(X, Xhat, mask_val)
                key = f"{K}|{alpha}|{beta}"
                scores[key] = float(vpx)
                print(f"[val] {dataset}: K={K:>3d} α={alpha:.2f} β={beta:.2f} "
                      f"→ perplexity={vpx:.6f} ({elapsed:.1f}s)")
                if vpx < best_val:
                    best_val = vpx
                    best = (K, alpha, beta)

    assert all(x is not None for x in best)
    return scores, (int(best[0]), float(best[1]), float(best[2]))


def evaluate_test_runs(
    X: np.ndarray,
    masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dataset: str,
    method: str,
    K: int,
    alpha: float,
    beta: float,
    seeds: Iterable[int],
    max_iter: int,
    tol: float,
    **fit_kwargs,
) -> List[TestResult]:
    """
    Train (on train mask) and evaluate test perplexity over multiple seeds.
    """
    mask_train, _, mask_test = masks
    results: List[TestResult] = []
    for s in seeds:
        t0 = time.time()
        if method == "NBMF-EM":
            # EM proxy: α=β=1 (as noted in the paper)
            _, Xhat = fit_nbmf_mm(
                X, mask_train, K, 1.0, 1.0,
                random_state=s, max_iter=max_iter, tol=tol, **fit_kwargs
            )
            a, b = 1.0, 1.0
        elif method == "NBMF-MM":
            _, Xhat = fit_nbmf_mm(
                X, mask_train, K, alpha, beta,
                random_state=s, max_iter=max_iter, tol=tol, **fit_kwargs
            )
            a, b = alpha, beta
        elif method == "logPCA":
            Xhat = fit_logistic_pca_predict(
                X, K=K, max_iters=max_iter, tol=tol, m=4
            )
            if Xhat is None:
                print("[warn] rpy2/logisticPCA not available — skipping logPCA.")
                continue
            a, b = np.nan, np.nan
        else:
            raise ValueError(method)

        t = time.time() - t0
        tpx = perplexity(X, Xhat, mask_test)
        results.append(TestResult(dataset, method, K, float(a), float(b), int(s), float(tpx), float(t)))
        print(f"[test] {dataset}: {method:8s} K={K:>3d} "
              f"α={a if not np.isnan(a) else '—'} β={b if not np.isnan(b) else '—'} "
              f"→ perplexity={tpx:.6f} ({t:.1f}s)")
    return results


def dataset_default_grids(name: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Heuristic grids aligned with the paper’s figures (α,β ∈ {1.0,1.4,1.8,2.2,2.6,3.0}).
    K grids are sized to dataset scale and can be overridden from CLI.
    """
    ab = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    if name == "animals":
        K = [3, 4, 5, 6, 8]
    elif name == "paleo":
        K = [5, 10, 15, 20, 25]
    elif name == "lastfm":
        K = [10, 15, 20, 25, 30]
    else:
        K = [5, 10, 15, 20]
    return K, ab, ab


def locate_dataset_file(data_root: Path, dataset: str) -> Path:
    """
    Map dataset name to a file path under data_root.
    """
    candidates = [
        data_root / f"{dataset}.rda",
        data_root / f"{dataset}.RData",
        data_root / f"{dataset}.npz",
        data_root / f"{dataset}.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Dataset '{dataset}' not found under {data_root}")


# ---------------------------------- main ------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Reproduce Magron & Févotte (2022) NBMF-MM experiments."
    )
    p.add_argument("--data-root", type=Path, required=True,
                   help="Folder containing animals/paleo/lastfm (.rda/.npz/.npy).")
    p.add_argument("--datasets", type=str, default="animals,paleo,lastfm",
                   help="Comma-separated list of datasets to run.")
    p.add_argument("--outdir", type=Path, default=Path("outputs/magron2022"),
                   help="Where to save results.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the CV split & grid.")
    p.add_argument("--n-test-seeds", type=int, default=10, help="#seeds for test runs.")
    p.add_argument("--max-iter", type=int, default=2000, help="Max iterations.")
    p.add_argument("--tol", type=float, default=1e-5, help="Relative tolerance.")
    p.add_argument("--k-grid", type=str, default="", help="Optional K grid, e.g. '5,10,15'.")
    p.add_argument("--alpha-grid", type=str, default="", help="Optional alpha grid, e.g. '1,1.4,1.8'.")
    p.add_argument("--beta-grid", type=str, default="", help="Optional beta grid, e.g. '1,1.4,1.8'.")
    p.add_argument("--no-logpca", action="store_true", help="Disable logisticPCA baseline.")
    p.add_argument("--save-lastfm-H", action="store_true",
                   help="Save H heatmap-friendly array for lastfm with best params.")
    p.add_argument("--projection-method", type=str, default="duchi",
                   choices=["duchi", "normalize"])
    p.add_argument("--projection-backend", type=str, default="auto",
                   choices=["auto", "numba", "numpy"])
    p.add_argument("--no-numexpr", action="store_true", help="Disable NumExpr acceleration.")

    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    meta = {
        "paper": "Magron & Févotte (2022), A majorization–minimization algorithm for nonnegative binary matrix factorization",
        "split": "entry-wise 70/15/15 train/val/test",
        "stopping": {"tol": args.tol, "max_iter": args.max_iter},
        "estimator": "NBMF (orientation='beta-dir')",
        "logisticPCA": "optional via rpy2",
    }

    for ds in datasets:
        ds_dir = outdir / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        data_path = locate_dataset_file(Path(args.data_root), ds)
        X = load_binary_matrix(data_path)
        X = X.astype(float)
        print(f"Loaded {ds} from {data_path} with shape {X.shape}")

        mask_train, mask_val, mask_test = make_random_masks(X.shape, seed=args.seed)
        masks = (mask_train, mask_val, mask_test)

        # Grids
        def_gridK, def_gridA, def_gridB = dataset_default_grids(ds)
        K_grid = (
            [int(x) for x in args.k_grid.split(",")] if args.k_grid else def_gridK
        )
        alpha_grid = (
            [float(x) for x in args.alpha_grid.split(",")] if args.alpha_grid else def_gridA
        )
        beta_grid = (
            [float(x) for x in args.beta_grid.split(",")] if args.beta_grid else def_gridB
        )

        # Validation grid (single seed for selection as in paper)
        grid_scores, (bestK, bestA, bestB) = run_validation_grid(
            X, masks, ds, K_grid, alpha_grid, beta_grid, seed=args.seed,
            max_iter=args.max_iter, tol=args.tol,
            projection_method=args.projection_method,
            projection_backend=args.projection_backend,
            use_numexpr=not args.no_numexpr,
        )

        # Persist α–β heatmap data for the best K (used by display script)
        # Store all K slices too.
        np.savez_compressed(
            ds_dir / "val_grid.npz",
            scores=np.array(
                [[(k, a, b, grid_scores[f"{k}|{a}|{b}"])
                  for a in alpha_grid for b in beta_grid] for k in K_grid],
                dtype=float,
            ),
            K_grid=np.array(K_grid, dtype=int),
            alpha_grid=np.array(alpha_grid, dtype=float),
            beta_grid=np.array(beta_grid, dtype=float),
            bestK=bestK, bestA=bestA, bestB=bestB
        )
        with open(ds_dir / "val_best.json", "w") as f:
            json.dump({"bestK": bestK, "bestA": bestA, "bestB": bestB}, f, indent=2)

        # Test evaluation across seeds
        seeds = [int(s) for s in range(args.seed, args.seed + args.n_test_seeds)]
        rows: List[TestResult] = []
        rows += evaluate_test_runs(
            X, masks, ds, "NBMF-EM", bestK, bestA, bestB, seeds,
            max_iter=args.max_iter, tol=args.tol,
            projection_method=args.projection_method,
            projection_backend=args.projection_backend,
            use_numexpr=not args.no_numexpr,
        )
        rows += evaluate_test_runs(
            X, masks, ds, "NBMF-MM", bestK, bestA, bestB, seeds,
            max_iter=args.max_iter, tol=args.tol,
            projection_method=args.projection_method,
            projection_backend=args.projection_backend,
            use_numexpr=not args.no_numexpr,
        )
        if not args.no_logpca:
            rows += evaluate_test_runs(
                X, masks, ds, "logPCA", bestK, bestA, bestB, seeds,
                max_iter=args.max_iter, tol=args.tol,
            )

        # Save CSV
        hdr = "dataset,method,K,alpha,beta,seed,test_perplexity,elapsed_s\n"
        with open(ds_dir / "test_results.csv", "w") as f:
            f.write(hdr)
            for r in rows:
                f.write(
                    f"{r.dataset},{r.method},{r.K},{r.alpha},{r.beta},"
                    f"{r.seed},{r.test_perplexity:.12g},{r.elapsed_s:.6g}\n"
                )

        # Save H for lastfm visualization (paper's Fig. 3)
        if args.save_lastfm_H and ds.lower() == "lastfm":
            m, Xhat = fit_nbmf_mm(
                X, mask_train, bestK, bestA, bestB,
                random_state=args.seed, max_iter=args.max_iter, tol=args.tol,
                projection_method=args.projection_method,
                projection_backend=args.projection_backend,
                use_numexpr=not args.no_numexpr,
            )
            np.save(ds_dir / "H_lastfm_best.npy", m.components_)

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Use display_reproduced_results.py to render figures.")


if __name__ == "__main__":
    main()
