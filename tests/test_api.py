import numpy as np
import pytest

from nbmf_mm import NBMF

rng = np.random.default_rng(0)

def _toy_data(m=60, n=80, p=0.25):
    return (rng.random((m, n)) < p).astype(float)

def test_fit_shapes_dir_beta():
    X = _toy_data()
    model = NBMF(
        n_components=8,
        orientation="dir-beta",
        max_iter=200,
        tol=1e-6,
        use_numexpr=False,
        use_numba=False,
        projection_backend="numpy",
        random_state=0,
    )
    model.fit(X)
    assert model.W_.shape == (X.shape[0], 8)
    assert model.components_.shape == (8, X.shape[1])
    assert len(model.objective_history_) >= 1
    assert np.isfinite(model.objective_history_[-1])

def test_fit_shapes_beta_dir():
    X = _toy_data()
    model = NBMF(
        n_components=6,
        orientation="beta-dir",
        max_iter=150,
        tol=1e-6,
        use_numexpr=False,
        use_numba=False,
        projection_backend="numpy",
        random_state=0,
    )
    model.fit(X)
    assert model.W_.shape == (X.shape[0], 6)
    assert model.components_.shape == (6, X.shape[1])

def test_objective_not_increasing_with_normalize_projection():
    """
    MM guarantees monotone decrease for the MAP objective (NLL + Beta prior)
    when the simplex step is multiplicative + L1 renormalization ("normalize").
    """
    X = _toy_data()
    model = NBMF(
        n_components=5,
        max_iter=120,
        tol=1e-7,
        use_numexpr=False,
        use_numba=False,
        projection_method="normalize",   # MM-faithful
        projection_backend="numpy",
        random_state=0,
    )
    model.fit(X)
    hist = np.asarray(model.objective_history_, dtype=float)
    # Non-increasing over iterations; allow tiny numerical jitter
    assert np.all(hist[1:] <= hist[:-1] + 1e-8)
    assert hist[-1] <= hist[0] + 1e-10

def test_duchi_projection_runs_and_is_finite():
    """
    With Euclidean simplex projection ("duchi") we do not strictly assert
    monotonicity (projection is outside the original MM majorizer). Instead,
    assert the objective is finite and reconstructions are in-bounds.
    """
    X = _toy_data()
    model = NBMF(
        n_components=5,
        max_iter=120,
        tol=1e-7,
        use_numexpr=False,
        use_numba=False,
        projection_method="duchi",
        projection_backend="numpy",
        random_state=0,
    )
    model.fit(X)
    hist = np.asarray(model.objective_history_, dtype=float)
    assert np.isfinite(hist).all()
    W = model.W_
    H = model.components_
    Xhat = model.inverse_transform(W)
    assert Xhat.shape == X.shape
    assert np.all((Xhat >= 0.0) & (Xhat <= 1.0))

def test_transform_inverse_shapes():
    X = _toy_data()
    model = NBMF(
        n_components=7,
        max_iter=150,
        tol=1e-6,
        use_numexpr=False,
        use_numba=False,
        projection_backend="numpy",
        random_state=0,
    ).fit(X)
    W = model.transform(X)
    Xhat = model.inverse_transform(W)
    assert W.shape == (X.shape[0], 7)
    assert Xhat.shape == X.shape
    assert np.all((Xhat >= 0.0) & (Xhat <= 1.0))

def test_mask_training():
    X = _toy_data()
    mask = (rng.random(X.shape) < 0.9).astype(float)
    model = NBMF(
        n_components=6,
        max_iter=120,
        tol=1e-6,
        use_numexpr=False,
        use_numba=False,
        projection_backend="numpy",
        random_state=0,
    ).fit(X, mask=mask)
    s = model.score(X, mask=mask)
    perp = model.perplexity(X, mask=mask)
    assert np.isfinite(s)
    assert perp >= 1.0

def test_projection_variants_simplex_property():
    X = _toy_data()
    # Duchi + NumPy
    m1 = NBMF(
        n_components=5,
        projection_method="duchi",
        projection_backend="numpy",
        max_iter=50,
        use_numexpr=False,
        use_numba=False,
        random_state=0,
    ).fit(X)
    # Legacy normalize
    m2 = NBMF(
        n_components=5,
        projection_method="normalize",
        max_iter=50,
        use_numexpr=False,
        use_numba=False,
        random_state=0,
    ).fit(X)
    # Check simplex constraints
    if m1.orientation == "beta-dir":
        assert np.allclose(m1.W_.sum(axis=1), 1.0, atol=1e-6)
    else:
        assert np.allclose(m1.components_.sum(axis=0), 1.0, atol=1e-6)
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
    m = NBMF(
        n_components=4,
        max_iter=40,
        tol=1e-6,
        use_numexpr=False,
        use_numba=False,
        random_state=0,
    ).fit(Xs, mask=Ms)
    assert m.W_.shape == (X.shape[0], 4)

def test_objective_not_increasing_with_normalize_projection_beta_dir():
    """Monotone MAP descent under 'normalize' for the other orientation."""
    X = _toy_data()
    model = NBMF(
        n_components=5,
        orientation="beta-dir",
        max_iter=120,
        tol=1e-7,
        use_numexpr=False,
        use_numba=False,
        projection_method="normalize",
        projection_backend="numpy",
        random_state=0,
    )
    model.fit(X)
    hist = np.asarray(model.objective_history_, dtype=float)
    assert np.all(hist[1:] <= hist[:-1] + 1e-8)
