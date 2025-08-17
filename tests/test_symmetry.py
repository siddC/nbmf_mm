# tests/test_symmetry.py
import numpy as np
from nbmf_mm import BernoulliNMF_MM

def test_orientation_symmetry_dirbeta_vs_betadir_transpose():
    rng = np.random.default_rng(7)
    X = (rng.random((25, 30)) < 0.2).astype(float)
    K = 5

    # Fit Dir‑Beta on X (default orientation)
    m_db = BernoulliNMF_MM(
        n_components=K, alpha=1.3, beta=1.7,
        orientation="Aspect Bernoulli",
        max_iter=300, tol=1e-6, random_state=7
    ).fit(X)
    Xhat_db = m_db.inverse_transform(m_db.W_)

    # Fit Beta‑Dir on X^T
    m_bd = BernoulliNMF_MM(
        n_components=K, alpha=1.3, beta=1.7,
        orientation="binary ICA",
        max_iter=300, tol=1e-6, random_state=7
    ).fit(X.T)
    Xhat_bd_T = m_bd.inverse_transform(m_bd.W_).T

    # Reconstructions should be close
    assert np.allclose(Xhat_db, Xhat_bd_T, atol=5e-3, rtol=5e-3)
