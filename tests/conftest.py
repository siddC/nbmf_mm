import numpy as np
import pytest

def rng(seed=0):
    return np.random.default_rng(seed)

@pytest.fixture(scope="session")
def tiny_animals():
    """
    A tiny *synthetic* "animals-like" binary matrix for smoke tests.
    Shape 32x18 with 3 latent groups.  This is not the paper dataset;
    it just gives a stable target for basic assertions without R deps.
    """
    r = rng(42)
    M, N, K = 32, 18, 3
    # Construct cluster assignments
    z = r.integers(0, K, size=M)
    # Prototypes (probabilities) for K groups across N features
    protos = np.array([
        0.85 * (r.random(N) < 0.5),   # group 0: many 1s
        0.15 * (r.random(N) < 0.5),   # group 1: many 0s
        0.50 * (r.random(N) < 0.5),   # group 2: mixed
    ])
    P = np.vstack([protos[zi] for zi in z])
    # Sample binary matrix
    X = (r.random((M, N)) < P).astype(float)
    return X
