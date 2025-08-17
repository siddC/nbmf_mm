# tests/test_api_basic.py
import numpy as np
from nbmf_mm import NBMF

def test_api_shapes_and_bounds():
    rng = np.random.default_rng(0)
    M, N, K = 30, 20, 5
    X = (rng.random((M, N)) < 0.25).astype(float)

    model = NBMF(
        n_components=K,
        alpha=1.5, beta=1.2,
        orientation="Dir-Beta",   # default, case-insensitive alias
        max_iter=300, tol=1e-6, random_state=0
    )
    W = model.fit_transform(X)
    H = model.components_
    Xhat = model.inverse_transform(W)

    # shapes
    assert W.shape == (M, K)
    assert H.shape == (K, N)
    assert Xhat.shape == (M, N)

    # bounds and constraints
    assert np.all(W >= 0) and np.all(W <= 1)
    # For Dir-Beta, columns of H sum to 1 (within eps)
    col_sums = H.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-7)

    assert np.all(Xhat >= 0.0) and np.all(Xhat <= 1.0)

    # attributes populated
    assert isinstance(model.reconstruction_err_, float)
    assert model.n_iter_ >= 1
    assert len(model.objective_history_) == model.n_iter_
