import numpy as np
import pytest

from nbmf_mm import NBMF

rng = np.random.default_rng(0)

def _toy_data(m=60, n=80, p=0.25):
    return (rng.random((m, n)) < p).astype(float)

def test_fit_shapes_dir_beta():
    X = _toy_data()
    model = NBMF(n_components=8, orientation="dir-beta", max_iter=200, tol=1e-6,
                 use_numexpr=False, use_numba=False, projection_backend="numpy")
    model.fit(X)
    assert model.W_.shape == (X.shape[0], 8)
    assert model.components_.shape == (8, X.shape[1])
    assert len(model.objective_history_) >= 1
    assert np.isfinite(model.objective_history_[-1])

def test_fit_shapes_beta_dir():
    X = _toy_data()
    model = NBMF(n_components=6, orientation="beta-dir", max_iter=150, tol=1e-6,
                 use_numexpr=False, use_numba=False, projection_backend="numpy")
    model.fit(X)
    assert model.W_.shape == (X.shape[0], 6)
    assert model.components_.shape == (6, X.shape[1])

def test_objective_not_increasing():
    X = _toy_data()
    model = NBMF(n_components=5, max_iter=120, tol=1e-7,
                 use_numexpr=False, use_numba=False, projection_backend="numpy")
    model.fit(X)
    hist = np.asarray(model.objective_history_, dtype=float)
    assert hist[-1] <= hist[0] + 1e-8

def test_transform_inverse_shapes():
    X = _toy_data()
    model = NBMF(n_components=7, max_iter=150, tol=1e-6,
                 use_numexpr=False, use_numba=False, projection_backend="numpy").fit(X)
    W = model.transform(X)
    Xhat = model.inverse_transform(W)
    assert W.shape == (X.shape[0], 7)
    assert Xhat.shape == X.shape
    assert np.all((Xhat >= 0.0) & (Xhat <= 1.0))

def test_mask_training():
    X = _toy_data()
    mask = (rng.random(X.shape) < 0.9).astype(float)
    model = NBMF(n_components=6, max_iter=120, tol=1e-6,
                 use_numexpr=False, use_numba=False, projection_backend="numpy").fit(X, mask=mask)
    s = model.score(X, mask=mask)
    perp = model.perplexity(X, mask=mask)
    assert np.isfinite(s)
    assert perp >= 1.0

def test_projection_variants():
    X = _toy_data()
    # Duchi + NumPy
    m1 = NBMF(n_components=5, projection_method="duchi", projection_backend="numpy",
              max_iter=50, use_numexpr=False, use_numba=False).fit(X)
    assert m1.W_.shape[0] == X.shape[0]
    # Legacy normalize
    m2 = NBMF(n_components=5, projection_method="normalize",
              max_iter=50, use_numexpr=False, use_numba=False).fit(X)
    # Check rows/cols are proper simplexes
    if m2.orientation == "beta-dir":
        assert np.allclose(m2.W_.sum(axis=1), 1.0, atol=1e-6)
    else:
        assert np.allclose(m2.components_.sum(axis=0), 1.0, atol=1e-6)

def test_sparse_inputs():
    sp = pytest.importorskip("scipy.sparse")
    X = _toy_data()
    Xs = sp.csr_matrix(X)
    mask = (rng.random(X.shape) < 0.8).astype(float)
    Ms = sp.csr_matrix(mask)
    m = NBMF(n_components=4, max_iter=40, tol=1e-6, use_numexpr=False, use_numba=False).fit(Xs, mask=Ms)
    assert m.W_.shape == (X.shape[0], 4)
