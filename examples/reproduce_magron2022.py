# reproduce_magron2022.py
# -*- coding: utf-8 -*-
"""
Reproduce Magron & Févotte (2022) experiments with our NBMF-MM implementation.

Outputs (per dataset under outputs/<dataset>/):
  - NBMF-MM_val.npz         : validation perplexity tensor and hyperparam grids
  - NBMF-MM_model.npz       : best W, H, Y_hat on val; best (K, alpha, beta)
  - NBMF-MM_test_init.npz   : test perplexity/time/iters across random initializations
  - NBMF-EM_* analogous (with alpha=beta=1)
  - logPCA_* analogous (if logisticPCA via rpy2 is available)

Data: expects R .rda dataframes in data/ (same filenames as authors).
"""

import os, time
import numpy as np
import pyreadr

from nbmf_mm import BernoulliNMF_MM

# ---------------- helpers ----------------

def create_folder(path: str):
    os.makedirs(path, exist_ok=True)

def build_split_masks(shape, prop_train=0.7, prop_val=0.15, seed=12345):
    M, N = shape
    rng = np.random.default_rng(seed)
    # sample entries uniformly at random
    idx = np.arange(M * N)
    rng.shuffle(idx)
    n_train = int(round(prop_train * idx.size))
    n_val   = int(round(prop_val   * idx.size))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_mask = np.zeros(M * N, dtype=np.uint8); train_mask[train_idx] = 1
    val_mask   = np.zeros(M * N, dtype=np.uint8); val_mask[val_idx]     = 1
    test_mask  = np.zeros(M * N, dtype=np.uint8); test_mask[test_idx]   = 1

    return train_mask.reshape(M, N), val_mask.reshape(M, N), test_mask.reshape(M, N)

def get_perplexity(Y: np.ndarray, Yhat: np.ndarray, mask: np.ndarray, eps=1e-9) -> float:
    V = np.clip(Yhat, eps, 1.0 - eps)
    M = mask.astype(Y.dtype)
    ll = (M * Y) * np.log(V) + (M * (1.0 - Y)) * np.log(1.0 - V)
    nobs = float(M.sum())
    return float(-ll.sum() / max(nobs, 1.0))

def model_fit_nbmf(X_df, train_mask, K, alpha, beta, *, init='random',
                   orientation='beta-dir', max_iter=2000, tol=1e-5, seed=12345):
    Y = X_df.to_numpy().astype(np.float64, copy=False)
    model = BernoulliNMF_MM(
        n_components=int(K),
        alpha=alpha, beta=beta,
        orientation=orientation,
        init=init,
        max_iter=max_iter, tol=tol,
        random_state=seed, verbose=0
    )
    t0 = time.perf_counter()
    W = model.fit_transform(Y, mask=train_mask)
    H = model.components_
    dur = time.perf_counter() - t0
    Yhat = model.inverse_transform(W)
    return W, H, Yhat, dur, np.array(model.objective_history_, dtype=float), model.n_iter_

# (optional) logistic PCA via rpy2
def model_fit_logpca(X_df, train_mask, K, max_iter=2000, seed=12345):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, packages
        numpy2ri.activate()
        logisticPCA = packages.importr("logisticPCA")
        base = packages.importr("base")
    except Exception:
        return None  # gracefully skip

    Y = X_df.to_numpy().astype(np.float64, copy=False)
    # use only training entries for logisticPCA by imputing 0.5 elsewhere (neutral)
    Y_imp = Y.copy()
    Y_imp[train_mask == 0] = 0.5
    t0 = time.perf_counter()
    fit = logisticPCA.logisticPCA(Y_imp, k=int(K), main_effects=False, m=int(max_iter), quiet=True)
    dur = time.perf_counter() - t0
    # fitted natural parameter Θ; convert to mean via sigmoid
    theta = np.array(fit.rx2("Theta"), dtype=float)
    Yhat = 1.0 / (1.0 + np.exp(-theta))
    # crude factor proxies for saving viz (not needed for perplexity)
    W = np.zeros((Y.shape[0], int(K))); H = np.zeros((int(K), Y.shape[1]))
    return W, H, Yhat, dur, np.array([], dtype=float), int(max_iter)

# ---------------- main ----------------

if __name__ == "__main__":
    np.random.seed(12345)

    data_dir = "data/"
    out_dir  = "outputs/"
    datasets = ["animals", "paleo", "lastfm"]
    prop_train, prop_val = 0.70, 0.15

    max_iter = 2000
    tol = 1e-5
    n_init = 10

    # EXACT grids from authors:
    list_nfactors = [2, 4, 8, 16]                # K grid (paper code)  ← verified in main_script.py
    list_alpha = np.linspace(1, 3, 11)           # prior grid            ← verified in main_script.py
    list_beta  = list_alpha

    # NBMF‑EM (α=β=1) and NBMF‑MM (grid), plus logPCA
    models = ["NBMF-EM", "NBMF-MM", "logPCA"]

    for dataset in datasets:
        X_df = pyreadr.read_r(os.path.join(data_dir, f"{dataset}.rda"))[dataset]
        M, N = X_df.shape
        train_mask, val_mask, test_mask = build_split_masks(X_df.shape, prop_train=prop_train, prop_val=prop_val, seed=12345)
        np.savez(os.path.join(data_dir, f"{dataset}_split.npz"), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        ds_out = os.path.join(out_dir, dataset) + "/"
        create_folder(ds_out)

        # ---------- NBMF‑EM (α=β=1) ----------
        best_val = np.inf; best_tuple = None
        val_pplx_em = np.zeros((len(list_nfactors), 1, 1))
        for ik, K in enumerate(list_nfactors):
            W, H, Yhat, dur, hist, iters = model_fit_nbmf(
                X_df, train_mask, K, alpha=1.0, beta=1.0,
                init="random", orientation="beta-dir",  # authors’ orientation
                max_iter=max_iter, tol=tol, seed=12345
            )
            pplx = get_perplexity(X_df.to_numpy(), Yhat, mask=val_mask)
            val_pplx_em[ik, 0, 0] = pplx
            if pplx < best_val:
                np.savez(os.path.join(ds_out, "NBMF-EM_model.npz"),
                         W=W, H=H, Y_hat=Yhat,
                         hyper_params=(int(K), 1.0, 1.0), time=dur, loss=hist, iters=iters)
                best_val = pplx; best_tuple = (int(K), 1.0, 1.0)
        np.savez(os.path.join(ds_out, "NBMF-EM_val.npz"),
                 val_pplx=val_pplx_em, list_hyper=(list_nfactors, [1.0], [1.0]))

        # ---------- NBMF‑MM (grid over α,β) ----------
        nk, na, nb = len(list_nfactors), len(list_alpha), len(list_beta)
        val_pplx = np.zeros((nk, na, nb))
        best_val = np.inf; best_tuple = None
        for ik, K in enumerate(list_nfactors):
            for ia, a in enumerate(list_alpha):
                for ib, b in enumerate(list_beta):
                    W, H, Yhat, dur, hist, iters = model_fit_nbmf(
                        X_df, train_mask, K, alpha=float(a), beta=float(b),
                        init="random", orientation="beta-dir",
                        max_iter=max_iter, tol=tol, seed=12345
                    )
                    pplx = get_perplexity(X_df.to_numpy(), Yhat, mask=val_mask)
                    val_pplx[ik, ia, ib] = pplx
                    if pplx < best_val:
                        np.savez(os.path.join(ds_out, "NBMF-MM_model.npz"),
                                 W=W, H=H, Y_hat=Yhat,
                                 hyper_params=(int(K), float(a), float(b)), time=dur, loss=hist, iters=iters)
                        best_val = pplx; best_tuple = (int(K), float(a), float(b))
        np.savez(os.path.join(ds_out, "NBMF-MM_val.npz"),
                 val_pplx=val_pplx, list_hyper=(list_nfactors, list_alpha, list_beta))

        # ---------- logPCA ----------
        # Validation: choose best K only (no hyperparameters)
        lpca_best = None
        try_lpca = True
        val_pplx_lpca = np.zeros((len(list_nfactors), 1, 1))
        for ik, K in enumerate(list_nfactors):
            out = model_fit_logpca(X_df, train_mask, K, max_iter=max_iter, seed=12345)
            if out is None:
                try_lpca = False
                break
            W, H, Yhat, dur, hist, iters = out
            pplx = get_perplexity(X_df.to_numpy(), Yhat, mask=val_mask)
            val_pplx_lpca[ik, 0, 0] = pplx
            if (lpca_best is None) or (pplx < lpca_best[0]):
                np.savez(os.path.join(ds_out, "logPCA_model.npz"),
                         W=W, H=H, Y_hat=Yhat,
                         hyper_params=(int(K), None, None), time=dur, loss=hist, iters=iters)
                lpca_best = (pplx, int(K))
        if try_lpca:
            np.savez(os.path.join(ds_out, "logPCA_val.npz"),
                     val_pplx=val_pplx_lpca, list_hyper=(list_nfactors, [None], [None]))

        # ---------- Test with n_init random initializations ----------
        for model_name in ["NBMF-EM", "NBMF-MM"] + (["logPCA"] if try_lpca else []):
            hp = np.load(os.path.join(ds_out, f"{model_name}_model.npz"), allow_pickle=True)["hyper_params"]
            K = int(hp[0]); a = float(hp[1]) if hp[1] is not None else None; b = float(hp[2]) if hp[2] is not None else None

            test_pplx = np.zeros(n_init)
            test_time = np.zeros(n_init)
            test_iter = np.zeros(n_init)

            for i in range(n_init):
                if model_name == "logPCA":
                    out = model_fit_logpca(X_df, train_mask, K, max_iter=max_iter, seed=12345 + i)
                    if out is None:
                        break
                    W, H, Yhat, dur, hist, iters = out
                else:
                    W, H, Yhat, dur, hist, iters = model_fit_nbmf(
                        X_df, train_mask, K, alpha=(a if a is not None else 1.0), beta=(b if b is not None else 1.0),
                        init="random", orientation="beta-dir",
                        max_iter=max_iter, tol=tol, seed=12345 + i
                    )
                pplx = get_perplexity(X_df.to_numpy(), Yhat, mask=test_mask)
                test_pplx[i] = pplx; test_time[i] = dur; test_iter[i] = iters

            np.savez(os.path.join(ds_out, f"{model_name}_test_init.npz"),
                     test_pplx=test_pplx, test_time=test_time, test_iter=test_iter)
