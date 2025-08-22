# nbmf_mm/_solver.py
import time
from typing import Optional, Tuple, List

import numpy as np


def _validate_toggles(mask_policy: str, simplex_normalizer: str) -> None:
    allowed_mask = {"observed-only", "magron2022-legacy"}
    allowed_norm = {"observed-count", "magron2022-legacy"}
    if mask_policy not in allowed_mask:
        raise ValueError(f"mask_policy must be one of {allowed_mask}, got {mask_policy!r}")
    if simplex_normalizer not in allowed_norm:
        raise ValueError(f"simplex_normalizer must be one of {allowed_norm}, got {simplex_normalizer!r}")


def _ensure_array(x):
    if x is None:
        return None
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _project_simplex_cols_inplace(W: np.ndarray) -> np.ndarray:
    """
    Project each column of W onto the probability simplex (Duchi et al., 2008).
    W is modified in place and returned.
    """
    k, m = W.shape
    for j in range(m):
        v = W[:, j]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * (np.arange(k) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / float(rho + 1)
        W[:, j] = np.maximum(v - theta, 0.0)
    return W


def nbmf_mm_update_beta_dir(
    Y: np.ndarray,
    W: np.ndarray,  # shape (k, m), columns sum to 1
    H: np.ndarray,  # shape (k, n), values in (0, 1)
    mask: Optional[np.ndarray],
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    *,
    mask_policy: str = "observed-only",
    simplex_normalizer: str = "observed-count",
    projection_method: str = "normalize",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One MM iteration for beta-dir orientation. W columns sum to 1, H has Beta prior.

    mask_policy:
      - "observed-only" (default): only observed entries contribute to y and (1-y) terms
      - "magron2022-legacy": treat missing entries as negatives in the (1-y) term

    simplex_normalizer:
      - "observed-count" (default): divide each W column by the #observed entries for that sample
      - "magron2022-legacy": divide by global n (#features), regardless of masking

    projection_method:
      - "normalize" (default): multiplicative step + normalizer (MM-exact)
      - "duchi": multiplicative step + Euclidean projection to simplex (Duchi)
    """
    _validate_toggles(mask_policy, simplex_normalizer)
    m, n = Y.shape

    # --- Masked partitions
    M = np.ones_like(Y) if mask is None else (_ensure_array(mask))
    Y_pos = Y * M
    if mask_policy == "magron2022-legacy":
        # Legacy: missing act as zeros in Y ⇒ ones in (1 - Y)
        Y_neg = 1.0 - Y_pos
    else:
        Y_neg = (1.0 - Y) * M

    # --- Beta prior pseudo-counts
    A = (alpha - 1.0) * np.ones_like(H)
    B = (beta - 1.0) * np.ones_like(H)

    # ======================== H update ========================
    WH = W.T @ H                    # (m, n)
    num = H * (W @ (Y_pos / (WH + eps))) + A
    den = (1.0 - H) * (W @ (Y_neg / (1.0 - WH + eps))) + B
    H_new = num / (num + den + eps)
    H_new = np.clip(H_new, eps, 1.0 - eps)

    # ======================== W update ========================
    HW_T = H_new.T @ W              # (n, m)
    W_new = W * (
        H_new @ (Y_pos.T / (HW_T + eps)) +
        (1.0 - H_new) @ (Y_neg.T / (1.0 - HW_T + eps))
    )

    if projection_method == "duchi":
        # Ignore normalizer choice; project to the simplex instead
        W_new = _project_simplex_cols_inplace(np.maximum(W_new, 0.0))
    else:
        # "normalize": MM-exact normalizer
        if simplex_normalizer == "magron2022-legacy":
            W_new = W_new / float(n)
        else:
            obs_counts = M.sum(axis=1)                  # per-sample observed count (len m)
            obs_counts = np.maximum(obs_counts, 1.0)
            W_new = W_new / obs_counts[np.newaxis, :]
        W_new = W_new / (W_new.sum(axis=0, keepdims=True) + eps)

    return W_new, H_new


def nbmf_mm_solver(
    Y: np.ndarray,
    n_components: int,
    max_iter: int = 500,
    tol: float = 1e-5,
    alpha: float = 1.2,
    beta: float = 1.2,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
    orientation: str = "beta-dir",
    eps: float = 1e-8,
    *,
    mask_policy: str = "observed-only",
    simplex_normalizer: str = "observed-count",
    projection_method: str = "normalize",
) -> Tuple[np.ndarray, np.ndarray, List[float], float, int]:
    """
    Core MM solver for NBMF (beta-dir orientation).
    Returns (W, H, losses, time_elapsed, n_iter) with W shape (m, k), H shape (k, n).
    """
    _validate_toggles(mask_policy, simplex_normalizer)

    Y = np.asarray(Y, dtype=float)
    if np.any((Y < 0) | (Y > 1)):
        raise ValueError("Y must be in [0, 1].")
    m, n = Y.shape
    k = int(n_components)
    M = None if mask is None else (_ensure_array(mask))

    rng = np.random.default_rng(random_state)
    W0 = rng.uniform(0.1, 0.9, size=(m, k)) if W_init is None else np.asarray(W_init, dtype=float).copy()
    H0 = rng.uniform(0.1, 0.9, size=(k, n)) if H_init is None else np.asarray(H_init, dtype=float).copy()

    # Put W on the simplex (rows sum to 1 for (m, k) → columns sum to 1 for (k, m))
    W0 = np.maximum(W0, eps)
    W0 = W0 / (W0.sum(axis=1, keepdims=True) + eps)

    W = W0.T     # (k, m)
    H = np.clip(H0, eps, 1.0 - eps)

    losses: List[float] = []
    prev = np.inf
    t0 = time.time()

    for it in range(max_iter):
        W, H = nbmf_mm_update_beta_dir(
            Y, W, H, M, alpha, beta, eps,
            mask_policy=mask_policy,
            simplex_normalizer=simplex_normalizer,
            projection_method=projection_method,
        )

        # masked negative log-likelihood + prior (for monitoring / stopping)
        Y_hat = (W.T @ H)  # (m, n)
        if M is None:
            ll_term = Y * np.log(Y_hat + eps) + (1.0 - Y) * np.log(1.0 - Y_hat + eps)
            n_obs = Y.size
        else:
            ll_term = M * (Y * np.log(Y_hat + eps) + (1.0 - Y) * np.log(1.0 - Y_hat + eps))
            n_obs = int(np.count_nonzero(M))

        A = (alpha - 1.0)
        B = (beta - 1.0)
        prior = A * np.log(H + eps) + B * np.log(1.0 - H + eps)

        loss = - (ll_term.sum() + prior.sum()) / max(n_obs, 1)
        losses.append(float(loss))

        if verbose and (it % 50 == 0 or it == max_iter - 1):
            print(f"iter={it:4d}  loss={loss:.6f}")

        if prev - loss < tol:
            break
        prev = loss

    elapsed = time.time() - t0
    n_iter = it + 1
    return W.T, H, losses, float(elapsed), int(n_iter)
