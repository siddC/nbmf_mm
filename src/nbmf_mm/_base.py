# src/nbmf_mm/_base.py
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from ._solver import nbmf_mm_solver
from ._utils import check_is_fitted


class NBMFMM(BaseEstimator, TransformerMixin):
    """
    Nonnegative Binary Matrix Factorization via Majorization–Minimization.

    Implements the NBMF–MM algorithm from:
      P. Magron & C. Févotte (2022), “A Majorization–Minimization Algorithm for
      Nonnegative Binary Matrix Factorization,” IEEE SPL.

    Parameters
    ----------
    n_components : int, default=10
        Number of components (latent dimension k).
    alpha, beta : float, default=1.2
        Beta prior parameters (≥ 1) for the factor with Beta entries.
    max_iter : int, default=2000
    tol : float, default=1e-5
    W_init : ndarray, shape (n_samples, n_components), optional
    H_init : ndarray, shape (n_components, n_features), optional
    random_state : int or None, default=None
    verbose : int, default=0
    orientation : {"beta-dir","dir-beta"}, default="beta-dir"
        - "beta-dir"  (Binary ICA):       W rows simplex, H ∈ (0,1)
        - "dir-beta"  (Aspect Bernoulli): W ∈ (0,1),     H columns simplex
    """

    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 1.2,
        beta: float = 1.2,
        max_iter: int = 2000,
        tol: float = 1e-5,
        W_init: Optional[np.ndarray] = None,
        H_init: Optional[np.ndarray] = None,
        init=None,  # kept for compatibility
        random_state: Optional[int] = None,
        verbose: int = 0,
        orientation: str = "beta-dir",
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
        self.orientation = self._normalize_orientation(orientation)

    # ------------------------------ API ---------------------------------

    def fit(self, X, y=None, mask=None):
        """Fit NBMF–MM to binary (or [0,1]) data X."""
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if hasattr(X, "toarray"):  # sparse
            X = X.toarray()

        if not np.all((X >= 0.0) & (X <= 1.0)):
            raise ValueError("X must be binary or in [0, 1].")

        W, H, losses, t, n_iter = nbmf_mm_solver(
            Y=X,
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            alpha=self.alpha,
            beta=self.beta,
            W_init=self.W_init,
            H_init=self.H_init,
            mask=mask,
            random_state=self.random_state,
            verbose=self.verbose,
            orientation=self.orientation,
        )

        self.W_ = W
        self.components_ = H
        self.loss_curve_ = list(losses)
        self.objective_history_ = self.loss_curve_  # alias
        self.loss_ = self.loss_curve_[-1] if self.loss_curve_ else np.inf
        self.n_iter_ = n_iter
        self.reconstruction_err_ = self.loss_  # alias

        return self

    def fit_transform(self, X, y=None, mask=None):
        """Fit the model and return W."""
        self.fit(X, y=y, mask=mask)
        return self.W_

    def transform(self, X, mask=None, n_iter: int = 50):
        """
        Given fixed H, estimate W for new data X.

        - In "beta-dir" mode (Binary ICA), W rows lie on the simplex.
          We enforce this with the same λ-normalized update used in fit,
          starting from a simplex-initialized W to keep WH in [0,1].
        - In "dir-beta" mode (Aspect Bernoulli), W has Beta entries in (0,1);
          we do not project W to a simplex.
        """
        check_is_fitted(self, ["components_"])
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if hasattr(X, "toarray"):
            X = X.toarray()

        m, _ = X.shape
        k = self.n_components
        H = self.components_

        eps = 1e-8

        # Initialize W
        if self.orientation == "beta-dir":
            # Simplex init prevents negative denominators in the first step
            W = np.random.rand(m, k)
            W = W / (W.sum(axis=1, keepdims=True) + eps)
        else:
            # Beta entries in (0,1)
            W = np.random.uniform(0.1, 0.9, size=(m, k))

        # Build masked views once
        if mask is not None and hasattr(mask, "toarray"):
            mask = mask.toarray()
        if mask is None:
            Y_obs = X
            Z_obs = 1.0 - X
            Y_T = X.T
            Z_T = Z_obs.T
        else:
            Y_obs = X * mask
            Z_obs = (1.0 - X) * mask
            Y_T = Y_obs.T
            Z_T = Z_obs.T

        # Iterate a few steps
        for _ in range(n_iter):
            W_T = W.T  # (k, m)
            HW_T = H.T @ W_T  # (n, m)

            if self.orientation == "beta-dir":
                # λ-normalized (simplex) W update
                F = H @ (Y_T / (HW_T + eps)) + (1.0 - H) @ (Z_T / (1.0 - HW_T + eps))
                W_raw = W_T * F
                lam = W_raw.sum(axis=0, keepdims=True)
                W_T = W_raw / (lam + eps)
                W = W_T.T  # rows sum to 1 after transpose
            else:
                # "dir-beta": W is Beta – elementwise update in (0,1), no simplex
                A = (self.alpha - 1.0)
                B = (self.beta - 1.0)
                num_W = W_T * (H @ (Y_T / (HW_T + eps))) + A
                den_W = (1.0 - W_T) * (H @ (Z_T / (1.0 - HW_T + eps))) + B
                W_T = num_W / (num_W + den_W + eps)
                W_T = np.clip(W_T, eps, 1.0 - eps)
                W = W_T.T

        # Final guardrails
        if self.orientation == "beta-dir":
            s = W.sum(axis=1, keepdims=True)
            W = W / (s + eps)  # rows sum to 1 (non-negative)
        else:
            W = np.clip(W, eps, 1.0 - eps)

        return W

    def inverse_transform(self, W):
        """Return reconstruction in [0,1]."""
        check_is_fitted(self, ["components_"])
        W = check_array(W, dtype=np.float64)
        Xhat = W @ self.components_
        return np.clip(Xhat, 0.0, 1.0)

    def score(self, X, mask=None):
        """
        Average log-likelihood per observed entry (higher is better, ≤ 0).
        """
        check_is_fitted(self, ["components_"])
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Predict probabilities
        Xhat = self.inverse_transform(self.transform(X, mask=mask, n_iter=10))
        eps = 1e-8

        if mask is not None and hasattr(mask, "toarray"):
            mask = mask.toarray()

        if mask is None:
            Y_obs = X
            Z_obs = 1.0 - X
            n_obs = X.size
        else:
            Y_obs = X * mask
            Z_obs = (1.0 - X) * mask
            n_obs = int(mask.sum())

        log_lik = Y_obs * np.log(Xhat + eps) + Z_obs * np.log(1.0 - Xhat + eps)
        return float(np.sum(log_lik) / max(n_obs, 1))

    def perplexity(self, X, mask=None):
        """
        Perplexity = exp(- average log-likelihood per observed entry).
        Lower is better; minimum is 1.0 when predictions are perfect.
        """
        s = self.score(X, mask=mask)
        # score is ≤ 0, so perplexity ≥ 1
        return float(np.exp(-s))

    # -------------------------- helpers ---------------------------------

    def _normalize_orientation(self, orientation: str) -> str:
        """Normalize orientation parameter to standard form."""
        mapping = {
            "beta-dir": "beta-dir",
            "dir-beta": "dir-beta",
            "Beta-Dir": "beta-dir",
            "Dir-Beta": "dir-beta",
            "Dir Beta": "dir-beta",
            "binary ICA": "beta-dir",
            "Binary ICA": "beta-dir",
            "bICA": "beta-dir",
            "Aspect Bernoulli": "dir-beta",
        }
        if orientation in mapping:
            return mapping[orientation]
        raise ValueError(
            f"Unknown orientation: {orientation}. "
            f"Must be one of {list(mapping.keys())}"
        )


# Backwards-compatible alias
NBMF = NBMFMM
