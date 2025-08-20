import numpy as np
import pytest

from nbmf_mm._mm_exact import mm_step_beta_dir

def _dirichlet_rows(rng, M, K):
    W = rng.gamma(shape=1.0, scale=1.0, size=(M, K))
    W /= W.sum(axis=1, keepdims=True)
    return W

# Detect whether NBMF exposes init_W/init_H; if not, skip this strict test.
try:
    from nbmf_mm import NBMF as _NBMF_
    _init_vars = _NBMF_.__init__.__code__.co_varnames
    _has_init_injection = ("init_W" in _init_vars) and ("init_H" in _init_vars)
except Exception:
    _has_init_injection = False

@pytest.mark.skipif(not _has_init_injection, reason="NBMF does not expose init_W/init_H; skipping strict one-step parity test")
def test_strict_one_step_parity_when_init_injection_supported():
    from nbmf_mm import NBMF  # import here to avoid side effects at module import

    r = np.random.default_rng(123)
    M, N, K = 20, 25, 4
    Y = (r.random((M, N)) < 0.3).astype(float)

    alpha, beta = 1.2, 1.2
    W0 = _dirichlet_rows(r, M, K)
    H0 = np.clip(r.random((K, N)), 1e-6, 1 - 1e-6)

    # One paper-exact step from (W0, H0)
    W1_ref, H1_ref = mm_step_beta_dir(Y, W0.copy(), H0.copy(), alpha, beta)

    # Run your estimator for exactly one iteration starting from exactly (W0, H0)
    model = NBMF(
        n_components=K,
        orientation="beta-dir",
        alpha=alpha, beta=beta,
        random_state=0, n_init=1,
        max_iter=1, tol=0.0,
        projection_method="normalize",
        # These OPTIONAL kwargs are required for strict one-step parity:
        init_W=W0, init_H=H0,
    ).fit(Y)

    # Compare factors and reconstruction after one iteration
    W1 = model.W_
    H1 = model.components_
    Xhat_ref = W1_ref @ H1_ref
    Xhat = W1 @ H1

    assert np.allclose(W1, W1_ref, atol=1e-12)
    assert np.allclose(H1, H1_ref, atol=1e-12)
    assert np.allclose(Xhat, Xhat_ref, atol=1e-12)
