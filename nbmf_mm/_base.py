from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from ._solver import nbmf_mm_solver, nbmf_mm_update_beta_dir, _validate_toggles


class NBMFMM(BaseEstimator, TransformerMixin):
    """
    Non-negative Binary Matrix Factorization via Majorization-Minimization.

    Parameters
    ----------
    n_components : int
        Number of latent components.
    alpha, beta : float
        Beta prior hyperparameters (alpha, beta > 0).
    max_iter : int, default=500
    tol : float, default=1e-5
    random_state : int or None
    verbose : int, default=0
    orientation : {"beta-dir","dir-beta"}, default="beta-dir"
        - "beta-dir": H in (0,1), W rows sum to 1 (simplex).
        - "dir-beta": symmetric formulation on X.T.
    mask_policy : {"observed-only","magron2022-legacy"}, default="observed-only"
        Controls how masking is applied in the H-update.
        * "observed-only" (paper-correct): only observed entries contribute to both y and (1-y) terms.
        * "magron2022-legacy": treat missing entries as negatives in (1-y) term (matches Magron's script).
    simplex_normalizer : {"observed-count","magron2022-legacy"}, default="observed-count"
        Controls the W-step normalizer under masking.
        * "observed-count" (paper-correct): divide per row by # observed entries.
        * "magron2022-legacy": divide by global n (features), regardless of masking.
    """

    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 1.2,
        beta: float = 1.2,
        max_iter: int = 500,
        tol: float = 1e-5,
        W_init: Optional[np.ndarray] = None,
        H_init: Optional[np.ndarray] = None,
        init: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        orientation: str = "beta-dir",
        mask_policy: str = "observed-only",
        simplex_normalizer: str = "observed-count",
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.W_init = W_init
        self.H_init = H_init
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.orientation = orientation
        self.mask_policy = mask_policy
        self.simplex_normalizer = simplex_normalizer

        _validate_toggles(self.mask_policy, self.simplex_normalizer)

    # ---- Helpers

    def _fit_beta_dir(self, X, mask):
        W, H, losses, elapsed, n_iter = nbmf_mm_solver(
            X, self.n_components,
            max_iter=self.max_iter, tol=self.tol,
            alpha=self.alpha, beta=self.beta,
            W_init=self.W_init, H_init=self.H_init,
            mask=mask, random_state=self.random_state, verbose=self.verbose,
            orientation="beta-dir",
            mask_policy=self.mask_policy,
            simplex_normalizer=self.simplex_normalizer,
        )
        self.components_ = H  # shape (k, n)
        self.embedding_ = W   # shape (m, k) a.k.a. W
        self.n_iter_ = n_iter
        self.training_time_ = elapsed
        self.loss_curve_ = losses
        return self

    def _transform_beta_dir(self, X, mask, n_steps=50, eps=1e-8):
        # Given fixed H, update W only (beta-dir orientation).
        X = np.asarray(X, dtype=float)
        m, n = X.shape
        k = self.components_.shape[0]
        H = self.components_

        # Initialize W on simplex
        rng = np.random.default_rng(self.random_state)
        W = rng.uniform(0.1, 0.9, size=(m, k))
        W = np.maximum(W, eps)
        W = W / (W.sum(axis=1, keepdims=True) + eps)

        M = None if mask is None else (mask.toarray() if hasattr(mask, "toarray") else np.asarray(mask))

        # Run a few multiplicative steps
        Wk = W.T  # (k, m)
        for _ in range(n_steps):
            # one W-update with fixed H
            Y = X
            Mloc = np.ones_like(Y) if M is None else M
            Y_pos = Y * Mloc
            if self.mask_policy == "magron2022-legacy":
                Y_neg = 1.0 - Y_pos
            else:
                Y_neg = (1.0 - Y) * Mloc
            HW_T = H.T @ Wk
            Wk = Wk * (
                H @ (Y_pos.T / (HW_T + eps)) +
                (1.0 - H) @ (Y_neg.T / (1.0 - HW_T + eps))
            )
            if self.simplex_normalizer == "magron2022-legacy":
                Wk = Wk / float(n)
            else:
                obs_counts = Mloc.sum(axis=1)
                obs_counts = np.maximum(obs_counts, 1.0)
                Wk = Wk / obs_counts[np.newaxis, :]
            Wk = Wk / (Wk.sum(axis=0, keepdims=True) + eps)

        return Wk.T

    # ---- sklearn API

    def fit(self, X, y=None, mask=None):
        X = check_array(X, accept_sparse=False, dtype=float, order="C")
        if self.orientation == "beta-dir":
            return self._fit_beta_dir(X, mask)
        elif self.orientation == "dir-beta":
            # symmetric formulation on transposed data
            self._fit_beta_dir(X.T, None if mask is None else mask.T)
            # swap naming to keep attributes consistent with input X
            self.components_ = self.embedding_.T
            self.embedding_ = self.components_.T
            return self
        else:
            raise ValueError('orientation must be "beta-dir" or "dir-beta"')

    def transform(self, X, mask=None):
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        X = check_array(X, accept_sparse=False, dtype=float, order="C")
        if self.orientation != "beta-dir":
            raise NotImplementedError("transform currently implemented for orientation='beta-dir'")
        return self._transform_beta_dir(X, mask)

    def inverse_transform(self, W):
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        return np.dot(W, self.components_)

    def score(self, X, mask=None):
        """Average log-likelihood per observed entry (nats)."""
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        X = check_array(X, accept_sparse=False, dtype=float, order="C")
        W = self.transform(X, mask=mask)
        X_hat = self.inverse_transform(W)
        eps = 1e-8
        if mask is None:
            log_term = X * np.log(X_hat + eps) + (1.0 - X) * np.log(1.0 - X_hat + eps)
            n_obs = X.size
        else:
            M = mask.toarray() if hasattr(mask, "toarray") else np.asarray(mask)
            log_term = M * (X * np.log(X_hat + eps) + (1.0 - X) * np.log(1.0 - X_hat + eps))
            n_obs = int(np.count_nonzero(M))
        return float(np.sum(log_term)) / float(max(n_obs, 1))

    def perplexity(self, X, mask=None):
        """Traditional perplexity = exp(-average NLL per observed entry)."""
        return float(np.exp(-self.score(X, mask=mask)))
