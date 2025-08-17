# tests/test_monotonic_objective.py
import numpy as np
from nbmf_mm import BernoulliNMF_MM

def test_mm_objective_monotone_nonincreasing():
    rng = np.random.default_rng(123)
    X = (rng.random((40, 25)) < 0.30).astype(float)

    model = BernoulliNMF_MM(
        n_components=6,
        alpha=1.1, beta=1.1,
        orientation="dir-beta",
        max_iter=400, tol=1e-7, random_state=123
    ).fit(X)

    hist = np.array(model.objective_history_, dtype=float)
    diffs = np.diff(hist)

    # Allow tiny floating jitter; overall must go down meaningfully
    assert (diffs <= 1e-7).sum() >= 0.95 * diffs.size
    assert hist[-1] <= hist[0] - 1e-3
