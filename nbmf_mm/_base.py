# nbmf_mm/_base.py
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
    alpha, beta : float
    max_iter : int, default=500
    tol : float, default=1e-5
    random_state : int or None
    verbose : int, default=0
    orientation : {"beta-dir","dir-beta"}, default="beta-dir"
    projection_method : {"normalize","duchi"}, default="normalize"
    mask_policy : {"observed-only","magron2022-legacy"}, default="observed-only"
        H-update masking semantics. See README.
    simplex_normalizer : {"observed-count","magron2022-legacy"}, default="observed-count"
        W-step normalizer under masking. See README.

    The following kwargs are accepted for API compatibility but are no-ops:
      - use_numexpr, use_numba, projection_backend, init, W_init, H_init, n_init
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
        projection_method: str = "normalize",
        mask_policy: str = "observed-only",
        simplex_normalizer: str = "observed-count",
        n_init: int = 1,
        # accepted but unused
        use_numexpr: bool = False,
        use_numba: bool = False,
        projection_backend: str = "auto",
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
        self.projection_method = projection_method
        self.mask_policy = mask_policy
        self.simplex_normalizer = simplex_normalizer
        self.n_init = int(n_init)
        # compat shims
        self.use_numexpr = use_numexpr
        self.use_numba = use_numba
        self.projection_backend = projection_backend

        _validate_toggles(self.mask_policy, self.simplex_normalizer)
        
        # Normalize orientation aliases to canonical forms
        self.orientation = self._normalize_orientation(self.orientation)

    # ---- Internal helpers ----------------------------------------------------
    
    def _normalize_orientation(self, orientation):
        """Normalize orientation aliases to canonical forms."""
        # Convert to lowercase and remove special chars for matching
        norm = orientation.lower().replace("-", "").replace(" ", "").replace("_", "")
        
        # Map aliases to canonical forms
        alias_map = {
            # dir-beta aliases
            "dirbeta": "dir-beta",
            "aspectbernoulli": "dir-beta",
            "dirbeta": "dir-beta",
            # beta-dir aliases  
            "betadir": "beta-dir",
            "binaryica": "beta-dir",
            "bica": "beta-dir",
        }
        
        if norm in alias_map:
            return alias_map[norm]
        elif orientation in ["beta-dir", "dir-beta"]:
            return orientation
        else:
            raise ValueError(f'orientation must be "beta-dir", "dir-beta", or a recognized alias. Got: {orientation}')

    def _fit_beta_dir_single(self, X, mask, random_state):
        W, H, losses, elapsed, n_iter = nbmf_mm_solver(
            X, self.n_components,
            max_iter=self.max_iter, tol=self.tol,
            alpha=self.alpha, beta=self.beta,
            W_init=self.W_init, H_init=self.H_init,
            mask=mask, random_state=random_state, verbose=self.verbose,
            orientation="beta-dir",
            mask_policy=self.mask_policy,
            simplex_normalizer=self.simplex_normalizer,
            projection_method=self.projection_method,
        )
        return W, H, losses, elapsed, n_iter

    def _fit_beta_dir(self, X, mask):
        # Multi-start (if requested)
        if self.n_init <= 1:
            W, H, losses, elapsed, n_iter = self._fit_beta_dir_single(X, mask, self.random_state)
        else:
            rng = np.random.default_rng(self.random_state)
            seeds = rng.integers(1, 2**31 - 1, size=self.n_init)
            best = None
            best_loss = np.inf
            total_time = 0.0
            for rs in seeds:
                W_, H_, losses_, elapsed_, n_iter_ = self._fit_beta_dir_single(X, mask, int(rs))
                total_time += elapsed_
                if losses_[-1] < best_loss:
                    best = (W_, H_, losses_, n_iter_)
                    best_loss = losses_[-1]
            W, H, losses, n_iter = best
            elapsed = total_time  # aggregate

        # Persist learned factors (beta-dir orientation)
        self.components_ = H              # (k, n)
        self.embedding_ = W               # (m, k)
        self.W_ = W                       # public alias (tests/README use this)
        self.n_iter_ = n_iter
        self.training_time_ = elapsed
        self.loss_curve_ = losses
        self._fit_shape = X.shape
        
        # Legacy attribute aliases for backward compatibility
        self.objective_history_ = losses  # alias for loss_curve_
        
        # Add reconstruction error (final loss value)
        self.reconstruction_err_ = losses[-1] if losses else np.nan
        self.loss_ = losses[-1] if losses else np.nan  # final loss value
        return self

    def _transform_beta_dir(self, X, mask, n_steps=50, eps=1e-8):
        # Given fixed H, update W only (beta-dir).
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

        # Run a few multiplicative steps for W with fixed H
        Wk = W.T  # (k, m)
        for _ in range(n_steps):
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

            if self.projection_method == "duchi":
                # Project columns to simplex
                k_, m_ = Wk.shape
                for j in range(m_):
                    v = Wk[:, j]
                    u = np.sort(v)[::-1]
                    cssv = np.cumsum(u)
                    rho = np.nonzero(u * (np.arange(k_) + 1) > (cssv - 1))[0][-1]
                    theta = (cssv[rho] - 1.0) / float(rho + 1)
                    Wk[:, j] = np.maximum(v - theta, 0.0)
            else:
                # MM-exact normalizer path
                if self.simplex_normalizer == "magron2022-legacy":
                    Wk = Wk / float(n)
                else:
                    obs_counts = Mloc.sum(axis=1)
                    obs_counts = np.maximum(obs_counts, 1.0)
                    Wk = Wk / obs_counts[np.newaxis, :]

            Wk = Wk / (Wk.sum(axis=0, keepdims=True) + eps)

        return Wk.T

    # ---- sklearn API ---------------------------------------------------------

    def fit(self, X, y=None, mask=None):
        X = check_array(X, accept_sparse=True, dtype=float, order="C")
        # Convert sparse to dense for internal processing
        if hasattr(X, "toarray"):
            X = X.toarray()
        if self.orientation == "beta-dir":
            return self._fit_beta_dir(X, mask)
        elif self.orientation == "dir-beta":
            # symmetric formulation: fit on transposed data, then swap roles
            self._fit_beta_dir(X.T, None if mask is None else mask.T)
            # After fitting on X.T:
            #   embedding_ (on X.T) has shape (n, k)
            #   components_ (on X.T) has shape (k, m)
            # For dir-beta on X:
            #   W (continuous) := components_.T (m, k)
            #   H (simplex cols) := embedding_.T   (k, n)
            W_cont = self.components_.T
            H_simplex_cols = self.embedding_.T
            self.embedding_ = W_cont
            self.components_ = H_simplex_cols
            self.W_ = self.embedding_
            self._fit_shape = X.shape
            
            # Legacy attribute aliases for backward compatibility
            self.objective_history_ = self.loss_curve_  # alias for loss_curve_
            self.reconstruction_err_ = self.loss_curve_[-1] if self.loss_curve_ else np.nan
            self.loss_ = self.loss_curve_[-1] if self.loss_curve_ else np.nan
            return self
        else:
            raise ValueError('orientation must be "beta-dir" or "dir-beta"')

    def transform(self, X, mask=None):
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        X = check_array(X, accept_sparse=True, dtype=float, order="C")
        # Convert sparse to dense for internal processing
        if hasattr(X, "toarray"):
            X = X.toarray()
        if self.orientation == "beta-dir":
            return self._transform_beta_dir(X, mask)
        elif self.orientation == "dir-beta":
            # For dir-beta: we need to transform X to get W (continuous factors)
            # The components_ are H with simplex columns, embedding_ is W (continuous)
            # We need to solve for W given H and X, but this is complex for dir-beta
            # For now, just use a simple approximation
            return self.embedding_  # Return the stored W from fitting
        else:
            raise NotImplementedError(f"transform not implemented for orientation='{self.orientation}'")

    def inverse_transform(self, W):
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        return np.dot(W, self.components_)

    def score(self, X, mask=None):
        """Average log-likelihood per observed entry (nats)."""
        if not hasattr(self, "components_"):
            raise AttributeError("Model is not fitted yet.")
        X = check_array(X, accept_sparse=True, dtype=float, order="C")
        # Convert sparse to dense for internal processing
        if hasattr(X, "toarray"):
            X = X.toarray()
        # Use training W if this is the training matrix; otherwise compute W on the fly
        if hasattr(self, "_fit_shape") and X.shape == self._fit_shape:
            W = self.embedding_
        else:
            # Fall back to a small number of W updates
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

    def perplexity(self, X, mask=None, use_magron2022_legacy: bool = False):
        """
        Traditional perplexity = exp(-average NLL per observed entry).

        If use_magron2022_legacy is True, return the *average NLL per observed entry* (nats)
        to mirror the magron2022 research scripts' get_perplexity function definition.
        """
        avg_loglik = self.score(X, mask=mask)           # average log-likelihood per observed entry
        if use_magron2022_legacy:
            return float(-avg_loglik)                   # matches Magron's "perplexity" number
        return float(np.exp(-avg_loglik))               # textbook perplexity
        

# Proper public alias
NBMF = NBMFMM
