# tests/test_reproducibility.py
import numpy as np
from nbmf_mm import NBMF

def test_seed_reproducibility_and_variation():
    rng = np.random.default_rng(99)
    X = (rng.random((30, 18)) < 0.35).astype(float)

    kwargs = dict(n_components=4, alpha=1.2, beta=1.2, max_iter=250, tol=1e-6, orientation="dir-beta")

    m1 = NBMF(random_state=123, **kwargs).fit(X)
    m2 = NBMF(random_state=123, **kwargs).fit(X)
    m3 = NBMF(random_state=456, **kwargs).fit(X)

    # Same seed → very close final objective & reconstruction
    assert abs(m1.reconstruction_err_ - m2.reconstruction_err_) < 1e-8
    Xhat1 = m1.inverse_transform(m1.W_)
    Xhat2 = m2.inverse_transform(m2.W_)
    assert np.allclose(Xhat1, Xhat2, atol=1e-8)

    # Different seed → usually different objective (not guaranteed, but likely)
    # So we check that at least one distance is noticeable or component matrices differ.
    Xhat3 = m3.inverse_transform(m3.W_)
    diff = np.linalg.norm(Xhat1 - Xhat3)
    assert diff > 1e-6 or abs(m1.reconstruction_err_ - m3.reconstruction_err_) > 1e-6
