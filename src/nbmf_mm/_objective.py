# -*- coding: utf-8 -*-
"""Objective utilities for NBMF–MM."""
from __future__ import annotations
import numpy as np

__all__ = ["clip01", "bernoulli_nll", "beta_neglogprior", "perplexity"]

def clip01(V: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Clip to (eps, 1-eps) for stable logs/ratios."""
    return np.clip(V, eps, 1.0 - eps)

def bernoulli_nll(X: np.ndarray, P: np.ndarray, eps: float = 1e-9) -> float:
    """
    Bernoulli negative log-likelihood for mean-parameter P in [0,1].
    L = - Σ_{mn} [ x_{mn} log p_{mn} + (1-x_{mn}) log (1-p_{mn}) ].
    """
    P = clip01(P, eps)
    return float(-(X * np.log(P) + (1.0 - X) * np.log(1.0 - P)).sum())

def beta_neglogprior(Z: np.ndarray, alpha: np.ndarray, beta: np.ndarray, eps: float = 1e-9) -> float:
    """
    Elementwise −log Beta prior for Z ∈ (0,1), broadcasting alpha,beta over rows.
    """
    Z = clip01(Z, eps)
    # alpha,beta expected as shape (K,1) for broadcasting
    return float(-(((alpha - 1.0) * np.log(Z)) + ((beta - 1.0) * np.log(1.0 - Z))).sum())

def perplexity(X: np.ndarray, P: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-9) -> float:
    """
    Perplexity = exp( average NLL per observed entry ).
    Matches the paper’s evaluation routine (NBMF code). See their helpers/functions.py.
    """
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    P = clip01(P, eps)
    nll = -(mask * (X * np.log(P) + (1.0 - X) * np.log(1.0 - P))).sum()
    denom = float(mask.sum())
    return float(np.exp(nll / max(1.0, denom)))
