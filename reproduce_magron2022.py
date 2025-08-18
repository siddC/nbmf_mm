# reproduce_magron2022.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
import numpy as np
import pyreadr

# Our implementation
from nbmf_mm import NBMF  # API documented in the repo README. :contentReference[oaicite:5]{index=5}


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def build_split_masks(shape, prop_train=0.70, prop_val=0.15, seed=12345):
    """
    Create disjoint train/val/test masks with given proportions over entries.
    Returns float masks in {0,1} with sum = 1 everywhere.
    """
    m, n = shape
    rng = np.random.default_rng(seed)
    u = rng.random((m, n))
    train_mask = (u < prop_train).astype(float)
    val_mask = ((u >= prop_train) & (u < prop_train + prop_val)).astype(float)
    test_mask = (1.0 - train_mask - val_mask).astype(float)
    return train_mask, val_mask, test_mask


def bernoulli_perplexity(X, P, mask=None, eps=1e-9) -> float:
    """
    Perplexity = exp( NLL / (# observed) ), Bernoulli mean-parametrized.
    X: binary {0,1}, P: probabilities in (0,1), mask in [0,1].
    """
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    P = np.clip(P, eps, 1.0 - eps)
    nll = -(mask * (X * np.log(P) + (1 - X) * np.log(1 - P))).sum()
    denom = float(mask.sum()) if float(mask.sum()) > 0 else 1.0
    return float(np.exp(nll / denom))


def fit_nbmf_mm(
    X, train_mask, n_components, alpha, beta, *,
    orientation="dir-beta", max_iter=2000, tol=1e-8, seed=0,
    use_numexpr=True, projection_method="duchi", projection_backend="auto"
):
    """
    Fit NBMF-MM with given hyperparameters. Returns (W, H_featxcomp, Xhat, tot_time, iters).
    H is saved as (n_features x n_components) to mirror the 2022 npz orientation.
    """
    t0 = time.perf_counter()
    model = NBMF(
        n_components=n_components,
        orientation=orientation,
        alpha=alpha, beta=beta,
        max_iter=max_iter, tol=tol,
        random_state=seed,
        use_numexpr=use_numexpr,
        projection_method=projection_method,
        projection_backend=projection_backend,
    ).fit(X, mask=train_mask)
    tot_time = time.perf_counter() - t0

    W = model.W_  # (n_samples, n_components)
    H_comp = model.components_            # (n_components, n_features)
    H = H_comp.T                          # (n_features, n_components) for parity with 2022 npz
    Xhat = model.inverse_transform(W)     # reconstructed probabilities, (n_samples, n_features)

    iters = getattr(model, "n_iter_", None)
    if iters is None:
        hist = getattr(model, "objective_history_", None)
        iters = len(hist) if hist is not None else np.nan

    return W, H, Xhat, tot_time, iters


def training_with_validation(
    X, train_mask, val_mask, list_nfactors, list_alpha, list_beta,
    out_dir_dataset, *, max_iter=2000, tol=1e-8, base_seed=12345
):
    """
    Grid over K x alpha x beta; save best model and the entire validation cube.
    """
    nk, na, nb = len(list_nfactors), len(list_alpha), len(list_beta)
    val_pplx = np.zeros((nk, na, nb))
    opt_pplx = np.inf
    best_triplet = None

    # iterate hyper-params (mirror 2022 main_script structure). :contentReference[oaicite:6]{index=6}
    counter, total = 1, nk * na * nb
    for ik, K in enumerate(list_nfactors):
        for ia, a in enumerate(list_alpha):
            for ib, b in enumerate(list_beta):
                print(f"--- Hyperparameters {counter}/{total}: K={K}, alpha={a:.3g}, beta={b:.3g}")
                # deterministic but distinct seeds per triple:
                seed = (base_seed * 10_007 + 97 * K + 3 * ia + 5 * ib) % (2**32 - 1)

                W, H, Xhat, tot_time, iters = fit_nbmf_mm(
                    X, train_mask, K, a, b,
                    max_iter=max_iter, tol=tol, seed=seed
                )

                # Perplexity on validation entries
                perplx = bernoulli_perplexity(X, Xhat, mask=val_mask)
                print(f"Val perplexity: {perplx:.4f}")
                val_pplx[ik, ia, ib] = perplx

                # Save best so far
                if perplx < opt_pplx:
                    np.savez(
                        os.path.join(out_dir_dataset, "nbmf-mm_model.npz"),
                        W=W, H=H, Y_hat=Xhat,
                        hyper_params=(int(K), float(a), float(b)),
                        time=tot_time, loss=None, iters=iters,
                    )
                    opt_pplx = perplx
                    best_triplet = (K, a, b)

                counter += 1

    # Store the full validation grid
    np.savez(
        os.path.join(out_dir_dataset, "nbmf-mm_val.npz"),
        val_pplx=np.array(val_pplx),
        list_hyper=np.array([list_nfactors, list_alpha, list_beta], dtype=object),
    )

    return best_triplet


def train_test_init(
    X, train_mask, test_mask, hyper_params, out_dir_dataset,
    *, max_iter=2000, tol=1e-8, n_init=10, base_seed=12345
):
    """
    Fix hyper-params and evaluate test perplexity over many random initializations
    (mirrors 2022 workflow and storage). :contentReference[oaicite:7]{index=7}
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
        os.path.join(out_dir_dataset, "nbmf-mm_test_init.npz"),
        test_pplx=test_pplx, test_time=test_time, test_iter=test_iter
    )


# ---------------------------
# Main driver (datasets & loops)
# ---------------------------
if __name__ == "__main__":
    # Reproduce 2022 protocol with our class. Defaults mirror the paper/code. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    rng_seed = 12345
    datasets = ["animals", "paleo", "lastfm"]
    data_dir = Path("data")
    out_root = Path("outputs") / "chauhan2025"
    split_root = Path("data") / "chauhan2025"

    # Hyperparameters
    max_iter = 2000
    tol = 1e-8
    n_init = 10
    list_nfactors = [2, 4, 8, 16]
    list_alpha = list_beta = list(np.linspace(1.0, 3.0, 11))

    np.random.seed(rng_seed)

    for ds in datasets:
        print(f"\n==== Dataset: {ds} ====")
        # Load binary DF from .rda (as in 2022 repo)
        df = pyreadr.read_r(str(data_dir / f"{ds}.rda"))[ds]
        X = df.to_numpy(dtype=float)

        # Build & save split masks
        train_mask, val_mask, test_mask = build_split_masks(
            X.shape, prop_train=0.70, prop_val=0.15, seed=rng_seed
        )
        ensure_dir(split_root)
        np.savez(split_root / f"{ds}_split.npz",
                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        # Output dir for this dataset
        out_dir_ds = out_root / ds
        ensure_dir(out_dir_ds)

        # Validation grid, save best model and full val cube
        best = training_with_validation(
            X, train_mask, val_mask, list_nfactors, list_alpha, list_beta,
            str(out_dir_ds), max_iter=max_iter, tol=tol, base_seed=rng_seed
        )
        print(f"Best (K, alpha, beta) on {ds}: {best}")

        # Test-time multi-init evaluation at the chosen hyper-params
        if best is None:  # fallback: load from model npz we just wrote
            best = np.load(out_dir_ds / "nbmf-mm_model.npz", allow_pickle=True)["hyper_params"]
        train_test_init(
            X, train_mask, test_mask, best, str(out_dir_ds),
            max_iter=max_iter, tol=tol, n_init=n_init, base_seed=rng_seed
        )

    print("\nAll done. Outputs are under outputs/chauhan2025 and splits under data/chauhan2025.")
