# -*- coding: utf-8 -*-
"""Initialization and projection helpers for NBMF–MM."""
from __future__ import annotations
import numpy as np

__all__ = [
    "project_rows_to_simplex", "project_cols_to_simplex",
    "dirichlet_rows", "dirichlet_cols", "beta_matrix",
    "nndsvd_warm_start"
]

def project_rows_to_simplex(W: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project each row onto the probability simplex (sum=1, >=eps)."""
    W = np.maximum(W, eps)
    s = W.sum(axis=1, keepdims=True)
    zero = (s[:, 0] <= eps)
    if np.any(zero):
        W[zero] = 1.0 / W.shape[1]
        s = W.sum(axis=1, keepdims=True)
    return W / s

def project_cols_to_simplex(H: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project each column onto the probability simplex (sum=1, >=eps)."""
    H = np.maximum(H, eps)
    s = H.sum(axis=0, keepdims=True)
    zero = (s[0, :] <= eps)
    if np.any(zero):
        H[:, zero] = 1.0 / H.shape[0]
        s = H.sum(axis=0, keepdims=True)
    return H / s

def dirichlet_rows(n_rows: int, n_cols: int, rng: np.random.Generator, concentration: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    alpha = np.full(n_cols, max(concentration, eps), dtype=np.float64)
    return project_rows_to_simplex(rng.dirichlet(alpha, size=n_rows), eps=eps)

def dirichlet_cols(n_rows: int, n_cols: int, rng: np.random.Generator, concentration: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    alpha = np.full(n_rows, max(concentration, eps), dtype=np.float64)
    H = rng.dirichlet(alpha, size=n_cols).T
    return project_cols_to_simplex(H, eps=eps)

def beta_matrix(n_rows: int, n_cols: int, rng: np.random.Generator, a: float, b: float, eps: float = 1e-9) -> np.ndarray:
    X = rng.beta(max(a, eps), max(b, eps), size=(n_rows, n_cols))
    return np.clip(X, eps, 1.0 - eps)

def nndsvd_warm_start(X: np.ndarray, k: int, rng: np.random.Generator, eps: float = 1e-9):
    """
    Optional: NNDSVD warm start via scikit-learn (Frobenius). Rows/columns are projected/clipped.
    Returns (W, H) or raises ImportError if sklearn is not installed.
    """
    from sklearn.decomposition import NMF as _SKNMF  # local import to keep sklearn optional
    sk = _SKNMF(n_components=k, init="nndsvd", solver="cd",
                beta_loss="frobenius", max_iter=400,
                random_state=rng.integers(0, 2**31-1))
    W0 = sk.fit_transform(X); H0 = sk.components_
    W = project_rows_to_simplex(W0.astype(float, copy=False), eps=eps)
    H = np.clip(H0.astype(float, copy=False), eps, 1.0 - eps)
    # per-row robust rescale to ≤1
    q = np.quantile(H, 0.99, axis=1, keepdims=True)
    q = np.where(q <= eps, 1.0, q)
    H = np.clip(H / q, eps, 1.0 - eps)
    return W, H
