#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Magron & Févotte (2022) NBMF-MM experiments using our nbmf-mm solver,
STRICTLY matching the original protocol and splits.

- Loads animals/paleo/lastfm from data/*.rda
- Uses PRECOMPUTED splits from data/magron2022/*_split.npz
- Grid-search: K in {2,4,8,16}, alpha,beta in linspace(1.0,3.0,11)
- Orientation: Beta prior on H, simplex on W  -> orientation='beta-dir'
- Metric: Bernoulli mean NLL (cross-entropy) – identical to “perplexity” in 2022 code
- Projection: Euclidean simplex projection for W using CONDAT (finite-time), numpy backend,
  EPS=1e-8 everywhere, deterministic seeding
- Outputs under outputs/chauhan2025/<dataset>/ with 2022-compatible keys

Protocol mirrors NMF-binary/main_script.py and display_results.py.  # noqa
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pyreadr

def find_repo_root(start: Optional[Path] = None) -> Path:
    here = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Could not find repository root (need 'data' and 'outputs').")

REPO_ROOT = find_repo_root()
for add in (REPO_ROOT, REPO_ROOT / "src"):
    if str(add) not in sys.path:
        sys.path.insert(0, str(add))

from nbmf_mm import NBMF  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "magron2022"
OUT_ROOT = REPO_ROOT / "outputs" / "chauhan2025"

# === Compatibility knobs (tuned to match 2022) ===
EPS = 1e-8
PROJECTION_METHOD = "condat"       # exact, finite-time simplex projection (Condat 2016)
PROJECTION_BACKEND = "numpy"
USE_NUMEXPR = False                # avoid kernel fusion that can change rounding
ORIENTATION = "beta-dir"           # Beta prior on H, simplex on W (paper’s orientation)
MAX_ITER = 2000
TOL = 1e-8
SEED = 12345

def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def bernoulli_cross_entropy(X: np.ndarray, P: np.ndarray, mask: np.ndarray | None = None,
                            eps: float = EPS) -> float:
    """Mean Bernoulli NLL over observed entries (identical to 2022 'perplexity')."""
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    P = np.clip(P, eps, 1.0 - eps)
    nll = -(mask * (X * np.log(P) + (1 - X) * np.log(1 - P))).sum(dtype=np.float64)
    denom = float(mask.sum()) or 1.0
    return float(nll / denom)

def _check_simplex(W: np.ndarray, tol: float = 1e-7) -> None:
    if np.any(W < -1e-12):
        raise ValueError("W contains negative entries beyond tolerance.")
    row_sums = W.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"W rows do not sum to 1 within tol={tol} (min/max: "
                         f"{row_sums.min():.6g}, {row_sums.max():.6g}).")

def fit_nbmf_mm(X: np.ndarray, train_mask: np.ndarray, n_components: int,
                alpha: float, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Fit NBMF-MM and return (W, H_featxcomp, Xhat, time, n_iter)."""
    t0 = time.perf_counter()
    model = NBMF(
        n_components=n_components,
        orientation=ORIENTATION,
        alpha=alpha,
        beta=beta,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=seed,
        use_numexpr=USE_NUMEXPR,
        projection_method=PROJECTION_METHOD,
        projection_backend=PROJECTION_BACKEND,
        eps=EPS,                  # ensure internal clipping aligns with metric EPS
    ).fit(X, mask=train_mask.astype(float))
    tot = time.perf_counter() - t0

    W = model.W_.astype(np.float64, copy=False)
    H = model.components_.T.astype(np.float64, copy=False)     # features × comps (matches 2022)
    Xhat = model.inverse_transform(W).astype(np.float64, copy=False)

    # Integrity check – fail fast if projection drifted
    _check_simplex(W)

    # robust n_iter
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None:
        hist = getattr(model, "objective_history_", None)
        n_iter = len(hist) if hist is not None else np.nan

    return W, H, Xhat, float(tot), int(n_iter) if n_iter == n_iter else np.nan

def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_file = DATA_DIR / f"{name}.rda"
    split_file = SPLIT_DIR / f"{name}_split.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset: {data_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file} (magron2022)")

    df = pyreadr.read_r(str(data_file))[name]
    X = df.to_numpy(dtype=float)                  # dense (paper is dense)
    splits = np.load(split_file)
    train_mask = splits["train_mask"].astype(float)
    val_mask   = splits["val_mask"].astype(float)
    test_mask  = splits["test_mask"].astype(float)
    return X, train_mask, val_mask, test_mask

def run_validation_grid(dataset: str, K_grid: list[int],
                        alpha_grid: list[float], beta_grid: list[float],
                        seed: int) -> Tuple[np.ndarray, Dict, Dict]:
    X, train_mask, val_mask, _ = load_dataset(dataset)

    val = np.full((len(K_grid), len(alpha_grid), len(beta_grid)), np.inf, dtype=np.float64)
    best = {"score": np.inf, "K": None, "alpha": None, "beta": None}
    best_model: Dict[str, np.ndarray] = {}

    for i, K in enumerate(K_grid):
        for j, a in enumerate(alpha_grid):
            for k, b in enumerate(beta_grid):
                run_seed = (seed * 10007 + 97 * K + 3 * j + 5 * k) % (2**32 - 1)
                W, H, Xhat, tot, iters = fit_nbmf_mm(X, train_mask, K, a, b, seed=run_seed)
                ce = bernoulli_cross_entropy(X, Xhat, mask=val_mask)
                val[i, j, k] = ce
                if ce < best["score"]:
                    best.update({"score": ce, "K": K, "alpha": a, "beta": b})
                    best_model = {"W": W, "H": H, "Xhat": Xhat, "time": tot, "iters": iters}
    return val, best, best_model

def evaluate_test_multi_init(dataset: str, best_params: Dict, n_init: int, seed: int):
    X, train_mask, _, test_mask = load_dataset(dataset)
    test_scores, test_times, test_iters = [], [], []
    for i in range(n_init):
        W, H, Xhat, tot, iters = fit_nbmf_mm(
            X, train_mask,
            best_params["K"], best_params["alpha"], best_params["beta"],
            seed=(seed + i) % (2**32 - 1)
        )
        ce = bernoulli_cross_entropy(X, Xhat, mask=test_mask)
        test_scores.append(ce); test_times.append(tot); test_iters.append(iters)
    return np.array(test_scores), np.array(test_times), np.array(test_iters)

if __name__ == "__main__":
    np.random.seed(SEED)
    ensure_dir(OUT_ROOT)

    datasets = ["animals", "paleo", "lastfm"]
    K_grid = [2, 4, 8, 16]
    alpha_beta = np.linspace(1.0, 3.0, 11).tolist()

    for ds in datasets:
        print(f"\n==== {ds.upper()} ====")
        out_dir = OUT_ROOT / ds; ensure_dir(out_dir)

        val_scores, best, best_model = run_validation_grid(ds, K_grid, alpha_beta, alpha_beta, seed=SEED)

        # Save validation cube (identical key names to 2022)
        np.savez(out_dir / "nbmf-mm_val.npz",
                 val_pplx=val_scores,
                 list_hyper=np.array([K_grid, alpha_beta, alpha_beta], dtype=object))

        # Save best model with 2022-compatible keys
        np.savez(out_dir / "nbmf-mm_model.npz",
                 W=best_model["W"], H=best_model["H"], Y_hat=best_model["Xhat"],
                 hyper_params=(int(best["K"]), float(best["alpha"]), float(best["beta"])),
                 time=float(best_model["time"]), loss=None, iters=best_model["iters"])

        print(f"Best (K, α, β) = ({best['K']}, {best['alpha']:.1f}, {best['beta']:.1f}); "
              f"val CE = {best['score']:.6f}")

        # Test multi-init (n=10) at chosen hyper-params
        test_scores, test_times, test_iters = evaluate_test_multi_init(ds, best, n_init=10, seed=SEED)
        np.savez(out_dir / "nbmf-mm_test_init.npz",
                 test_pplx=test_scores, test_time=test_times, test_iter=test_iters,
                 best_params=np.array(best, dtype=object))

    print("\nDone. Results in outputs/chauhan2025/<dataset>/")
