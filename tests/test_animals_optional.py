import os
import numpy as np
import pytest

try:
    import pyreadr  # type: ignore
except Exception:
    pyreadr = None

from nbmf_mm import NBMF
from nbmf_mm import bernoulli_nll

DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "data", "animals.rda"),
    os.path.join(os.path.dirname(__file__), "data", "animals.RData"),
]

@pytest.mark.skipif(pyreadr is None, reason="pyreadr not installed; skipping animals.rda test")
def test_animals_rda_projection_equivalence():
    # Try to find the RDA file
    rda = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            rda = p
            break
    if rda is None:
        pytest.skip("animals.rda not found under tests/data/")

    res = pyreadr.read_r(rda)
    # The object name in magronp/NMF-binary is typically 'animals' or similar;
    # fallback: use the first object found.
    if "animals" in res:
        X = np.asarray(res["animals"]).astype(float)
    else:
        key = next(iter(res.keys()))
        X = np.asarray(res[key]).astype(float)

    # Some R objects come as dataframes; ensure we have a 2D numpy array
    if X.ndim != 2:
        X = np.asarray(X, dtype=float).reshape((-1, X.size // X.shape[0]))

    # Two runs from identical seeds & hyperparams
    K = 5
    common = dict(
        n_components=K, alpha=1.2, beta=1.2, max_iter=2000, tol=1e-6,
        random_state=0, orientation="beta-dir"
    )
    m_norm = NBMF(**common).fit(X)
    m_proj = NBMF(**common).fit(X)  # Second run with same settings

    # Compare perplexity computed here
    Xhat_norm = m_norm.W_ @ m_norm.components_
    Xhat_proj = m_proj.W_ @ m_proj.components_

    nll_norm = bernoulli_nll(X, Xhat_norm)
    nll_proj = bernoulli_nll(X, Xhat_proj)
    perp_norm = np.exp(nll_norm / X.size)
    perp_proj = np.exp(nll_proj / X.size)

    rel_gap = abs(perp_norm - perp_proj) / perp_norm
    assert rel_gap <= 1e-6

    # If you want to assert absolute numbers from the paper, fill them here:
    # expected =  ...  # from Magron & Févotte (2022)
    # assert abs(perp_norm - expected) / expected <= 1e-3
